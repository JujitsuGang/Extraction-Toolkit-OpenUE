import transformers as trans
import torch
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.models.auto.configuration_auto import AutoConfig
from transformers import AutoTokenizer
from openue.data.utils import get_labels_ner, get_labels_seq, OutputExample
from typing import Dict

class BertForRelationClassification(trans.BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = trans.BertModel(config)
        self.relation_classification = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        label_ids_seq=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # import pdb; pdb.set_trace()
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0, :]
        relation_output = self.relation_classification(cls_output)
        relation_output_sigmoid = torch.sigmoid(relation_output)

        if label_ids_seq is None:
            return (relation_output_sigmoid, relation_output, cls_output)
        else:
            loss = self.loss_fn(relation_output, label_ids_seq)

            return (loss, relation_output_sigmoid, relation_output, cls_output)

    def add_to_argparse(parser):
        parser.add_argument("--model_type", type=str, default="bert")



class BertForNER(trans.BertPreTrainedModel):

    def __init__(self, config, **model_kwargs):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.bert = trans.BertModel(config)

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.token_classification = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        # labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        label_ids_seq=None,
        label_ids_ner=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # batch_size * 107 * hidden_size
        sequence_poolout_output = self.dropout(outputs[0])
        # batch_size * 107 * 6
        logits = self.token_classification(sequence_poolout_output)

        if label_ids_ner is None:
            return logits ,outputs[1]

        loss_fct = CrossEntropyLoss()
        # Only keep active parts of the loss
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, label_ids_ner.view(-1), torch.tensor(loss_fct.ignore_index).type_as(label_ids_ner)
            )
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, self.num_labels), label_ids_ner.view(-1))

        # if not return_dict:
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    
    
    def add_to_argparse(parser):
        parser.add_argument("--model_type", type=str, default="bert")


class Inference(pl.LightningModule):
    """
        input the text, 
        return the triples
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        # init the labels
        self._init_labels()    
        self._init_models()
        
        
        self.mode = "event" if "event" in args.task_name else "triple"
        self.start_idx = self.tokenizer("[relation0]", add_special_tokens=False)['input_ids'][0]
        
        if self.mode == "event":
            self.process = self.event_process
        else:
            self.process = self.normal_process
        
    
    def _init_labels(self):
        self.labels_ner = get_labels_ner()
        self.label_map_ner: Dict[int, str] = {i: label for i, label in enumerate(self.labels_ner)}
        self.num_labels_ner = len(self.labels_ner)

        # 读取seq的label
        self.labels_seq = get_labels_seq(self.args)
        self.label_map_seq: Dict[int, str] = {i: label for i, label in enumerate(self.labels_seq)}
        self.num_labels_seq = len(self.labels_seq)

    
    
    def _init_models(self):
        model_name_or_path = self.args.seq_model_name_or_path
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=self.num_labels_seq,
            label2id={label: i for i, label in enumerate(self.labels_seq)},
        )
        
        self.model_seq = BertForRelationClassification.from_pretrained(
            model_name_or_path,
            config=config,
        )

        model_name_or_path = self.args.ner_model_name_or_path
        # 读取待训练的ner模型
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=self.num_labels_ner,
            id2label=self.label_map_ner,
            label2id={label: i for i, label in enumerate(self.labels_ner)},
        )
        self.model_ner = BertForNER.from_pretrained(
            model_name_or_path,
            config=config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
        )

    def forward(self, inputs):
        """
        两种方案，一种直接所有relation搞起来，一种使用动态batch size, 针对出现的relation进行forward
        首先通过model_seq获得输入语句的类别标签，batch中每一个样本中含有的关系，
        之后选择大于阈值(0.5)的关系，将其输入取出来得到[batch_size*num_relation, seq_length]的输入向量，以及每一个样本对应的关系数量，
        将其增加了关系类别embedding之后，输入到model_ner中，得到input_ids中每一个token的类别，之后常规的实体识别。
        
        """
        # for k, v in inputs.items():
        #     if isinstance(v, torch.Tensor):
        #         inputs[k] = v.to(self.device)

        inputs_seq = {'input_ids': inputs['input_ids'],
                    'token_type_ids': inputs['token_type_ids'],
                    'attention_mask': inputs['attention_mask'],
                    }

        with torch.no_grad():
            outputs_seq = self.model_seq(**inputs_seq)

            batch_size = inputs_seq['input_ids'].shape[0]
            num_relations = len(self.label_map_seq.keys())
            max_length = inputs_seq['input_ids'].shape[1]

            # [batch_size, num_relation]
            relation_output_sigmoid = outputs_seq[0]

            # 多关系预测
            mask_relation_output_sigmoid = relation_output_sigmoid > 0.5
            # # 这个0.5是超参数，超参数
            # 如果没有关系那就选一个最大概率的关系抽取。
            for i in range(batch_size):
                if torch.sum(mask_relation_output_sigmoid[i]) == 0:
                    max_relation_idx = torch.max(relation_output_sigmoid[i], dim=0)[1].item()
                    mask_relation_output_sigmoid[i][max_relation_idx] = 1

            mask_relation_output_sigmoid = mask_relation_output_sigmoid.long()
            # mask_output [batch_size*num_relation] 表示哪一个输入是需要的
            mask_output = mask_relation_output_sigmoid.view(-1)

            # relation 特殊表示，需要拼接 input_ids :[SEP relation]  attention_mask: [1 1] token_type_ids:[1 1]
            # relation_index shape : [batch_size, num_relations]
            relation_index = torch.arange(self.start