""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import logging
import os
from tqdm import tqdm
from dataclasses import dataclass
from enum import Enum
from re import DEBUG, sub
from shutil import Error
from typing import List, Optional, Union, Dict

import numpy as np

import jsonlines

from transformers import PreTrainedTokenizer, is_torch_available, BatchEncoding
from transformers.utils.dummy_pt_objects import DebertaForQuestionAnswering

logger = logging.getLogger(__name__)


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def get_entities(seq, suffix=False):
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i-1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def f1_score(y_true, y_pred, average='micro', suffix=False):
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score


def accuracy_score(y_true, y_pred):
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]

    nb_correct = sum(y_t==y_p for y_t, y_p in zip(y_true, y_pred))
    nb_true = len(y_true)

    score = nb_correct / nb_true

    return score


def precision_score(y_true, y_pred, average='micro', suffix=False):
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)

    score = nb_correct / nb_pred if nb_pred > 0 else 0

    return score


def recall_score(y_true, y_pred, average='micro', suffix=False):
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_true = len(true_entities)

    score = nb_correct / nb_true if nb_true > 0 else 0

    return score


@dataclass
class InputExample:
    text_id: str
    words: str
    triples: List

@dataclass
class OutputExample:
    h: str
    r: str
    t: str

@dataclass
class InputFeatures:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids_seq: Optional[List[int]] = None
    label_ids_ner: Optional[List[int]] = None
    words: str = None

@dataclass
class InputFeatures_Interactive:
    input_ids: List[int] = None
    attention_mask: List[int] = None
    token_type_ids: List[int] = None
    triples: List[List[int]] = None



class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data.dataset import Dataset

    class OpenUEDataset(Dataset):

        features: List[InputFeatures]
        pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

        def __init__(
            self,
            data_dir: str,
            labels_seq: List,
            labels_ner: List,
            tokenizer: PreTrainedTokenizer,
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
            task='seq'

        ):
        
            with open(f"{data_dir}/rel2id.json", "r") as file:
                rel2id = json.load(file)
            # Load data features from cache or dataset file
            cached_examples_file = os.path.join(
                data_dir, "cached_{}_{}.examples".format(mode.value, tokenizer.__class__.__name__),
            )

            if task == 'seq':
                cached_features_file = os.path.join(
                    data_dir, "cached_{}_{}_seq".format(mode.value, tokenizer.__class__.__name__),
                )
            elif task == 'ner':
                cached_features_file = os.path.join(
                    data_dir, "cached_{}_{}_ner".format(mode.value, tokenizer.__class__.__name__),
                )
            elif task == 'interactive':
                cached_features_file = os.path.join(
                    data_dir, "cached_{}_{}_interactive".format(mode.value, tokenizer.__class__.__name__),
                )

            # features是否存在
            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                # examples是否存在
                if os.path.exists(cached_examples_file) and not overwrite_cache:
                    logger.info(f"Loading example from dataset file at {data_dir}")
                    examples = torch.load(cached_examples_file)
                else:
                    logger.info(f"Creating example from cached file {cached_examples_file}")
                    examples = read_examples_from_file(data_dir, mode)
                    torch.save(examples, cached_examples_file)

                logger.info(f"Creating features from dataset file at {data_dir}")
                if task == 'seq':
                    self.features = convert_examples_to_seq_features(
                        examples,
                        # labels,
                        labels_seq=labels_seq,
                        labels_ner=labels_ner,
                        max_seq_length=max_seq_length,
                        tokenizer=tokenizer,
                        cls_token_at_end=bool(model_type in ["xlnet"]),
                        cls_token=tokenizer.cls_token,
                        cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                        sep_token=tokenizer.sep_token,
                        sep_token_extra=False,
                        pad_on_left=bool(tokenizer.padding_side == "left"),
                        pad_token=tokenizer.pad_token_id,
                        pad_token_segment_id=tokenizer.pad_token_type_id,
                        pad_token_label_id=self.pad_token_label_id,
                    )
                elif task == 'ner':
                    self.features = convert_examples_to_ner_features(
                        examples,
                        labels_seq=labels_seq,
                        labels_ner=labels_ner,
                        max_seq_length=max_seq_length,
                        tokenizer=tokenizer,
                        cls_token_at_end=bool(model_type in ["xlnet"]),
                        cls_token=tokenizer.cls_token,
                        cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                        sep_token=tokenizer.sep_token,
                        sep_token_extra=False,
                        pad_on_left=bool(tokenizer.padding_side == "left"),
                        pad_token=tokenizer.pad_token_id,
                        pad_token_segment_id=tokenizer.pad_token_type_id,
                        pad_token_label_id=self.pad_token_label_id,
                    )
                elif task == 'interactive':
                    self.features = convert_examples_to_interactive_features(
                        examples,
                        labels_seq=labels_seq,
                        labels_ner=labels_ner,
                        max_seq_length=max_seq_length,
                        tokenizer=tokenizer,
                        rel2id=rel2id
                    )

                logger.info(f"Saving features into cached file {cached_features_file}")
                torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]

import json
def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[InputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, f"{mode}.json")

    examples = []

    with open(file_path, "r+", encoding="utf8") as f:

        text_id = 0
        
        for line in f.readlines():
            item = eval(line)
            text = item['text']
            triples = []
            for triple in item['spo_list']:
                triples.append([triple['subject'], triple['predicate'], triple['object']])

            examples.append(InputExample(text_id=text_id, words=text, triples=triples))
            text_id = text_id + 1

    return examples


def convert_examples_to_seq_features(
    examples: List[InputExample],
    labels_seq: List[str],
    labels_ner: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    features = []
    label2id = {label: i for i, label in enumerate(labels_seq)}
    cnt = 0

    for (ex_index, example) in tqdm(enumerate(examples), total=len(examples)):
        inputs = tokenizer(
            preprocess(example.words),
            add_special_tokens=True,
            # return_overflowing_tokens=True,
            truncation="longest_first",
            max_length=max_seq_length,
        )
        label_ids_seq = []
        for triple in example.triples:
            label_ids_seq.append(label2id[triple[1]])
        if len(label_ids_seq) == 0:
            cnt += 1
            continue
        label_ids_seq = torch.sum(torch.nn.functional.one_hot(torch.tensor(label_ids_seq), num_classes=len(labels_seq)), dim=0).float()
        # the relation may show more than once, [1,2,1,0] -> [1,1,1,0]
        label_ids_seq[label_ids_seq>0] = 1

        features.append(InputFeatures(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                          token_type_ids=inputs['token_type_ids'], label_ids_seq=label_ids_seq))

    logger.warning(f"total {cnt} samples error in the dataset!")
    return features



def preprocess(text):
    return text.lower()
    return " ".join([_ for _ in text]).lower()

def convert_examples_to_ner_features(
    examples: List[InputExample],
    labels_seq: List[str],
    labels_ner: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    # 将relation ids转化为特殊字符对应的ids,避免了relation 表示和原来的词表进行冲突
    start_idx = tokenizer("[relation0]", add_special_tokens=False)['input_ids'][0]
    label_map_seq = {label: i for i, label in enumerate(labels_seq)}
    seq_label2ids = {label: i+start_idx for i, label in enumerate(labels_seq)}
    label_map_ner = {label: i for i, label in enumerate(labels_ner)}

    features = []
    counter = 0

    def find_word_in_texts(word_ids, texts_ids):
        length = len(word_ids)
        for i, W in enumerate(texts_ids):
            if (i+length) >= len(texts_ids): break
            if texts_ids[i: i + length] == word_ids:
                return i, i + length
        return None, None

    for (ex_index, example) in enumerate(examples):
        # 用bert分词，转换为token
        # text = example.text
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        

        text = example.words
        
        tmp_triples = {}
        for triple in example.triples:
            subject = preprocess(triple[0])
            relation = triple[1]
            object_ = preprocess(triple[2])
            if relation not in tmp_triples:
                tmp_triples[relation] = [[subject, relation, object_]]
            else:
                tmp_triples[relation].append([subject, relation, object_])
        
        my_triples = []
        for k, v in tmp_triples.items():
            my_triples.append([[v[i][0] for i in range(len(v))], k, [v[i][2] for i in range(len(v))]])
        
        
        hard_to_solve = 0
        #  triple [[subject list], r, [object list]]
        for triple in my_triples:
            subject_list = triple[0]
            relation = triple[1]
            object_list = triple[2]
            
            # same entity map as subject and object            
            if set(subject_list) & set(object_list) :
                hard_to_solve += 1

            # cls w1 w2 .. sep w3 w4 sep 000000000
            # token_type
            # 000000000000000 1111111111
            # 转换为id，加上cls以及seq等
            # {"input_ids":[], "token_type_ids":[], "attention_mask":[]}
            inputs = tokenizer(
                preprocess(text),
                add_special_tokens=True,
                max_length=max_seq_length-2,
                truncation="longest_first"
            )

            inputs['token_type_ids'] = tokenizer.create_token_type_ids_from_sequences(inputs['input_ids'][1:-1],
                                                                                       [seq_label2ids[relation]])
            # label_map_seq[relation] 加入关系信息, 使用seq_label2ids
            inputs['input_ids'] = inputs['input_ids'] + [seq_label2ids[relation], tokenizer.sep_token_id]
            inputs['attention_mask'] = inputs['attention_mask'] + [1, 1]

            # 添加split_text文本的标签
            # B-SUB I-SUB / B-OBJ I-OBJ
            split_text_ids = inputs['input_ids']
            # ["O", "B-SUB", "I-SUB", "B-OBJ", "I-OBJ", "Relation"]
            # 默认所有位置都为'O'
            label_ner = ['O' for i in range(len(split_text_ids))]

            # 标注subject
            continue_flag = False
            for subject in subject_list:
                subject_ids = tokenizer.encode(subject, add_special_tokens=False)
                [start_idx, end_idx] = find_word_in_texts(subject_ids, split_text_ids)
                if start_idx is None:
                    # logger.info('语料有问题(subject)！%d', ex_index)
             