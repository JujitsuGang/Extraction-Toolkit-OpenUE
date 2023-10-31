import os
from .base_data_module import BaseDataModule
from .processor import get_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

import logging

logger = logging.getLogger(__name__)
from .utils import get_labels_ner, get_labels_seq, openue_data_collator_seq, openue_data_collator_ner, openue_data_collator_interactive


collator_set = {"ner": openue_data_collator_ner, "seq": openue_data_collator_seq, "interactive": openue_data_collator_interactive}

class REDataset(BaseDataModule):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.prepare_data()
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        self.num_labels = len(get_labels_ner()) if args.task_name == "ner" else len(get_labels_seq(args))
        self.collate_fn = collator_set[args.task_name]
        
        num_relations = len(get_labels_seq(args))

        # 默认加入特殊token来表示关系
        add_flag = False
        for i in range(num_relations):
            if f"[relation{i}]" not in self.tokenizer.get_added_vocab():
                add_flag = True
                break
        
        if add_flag:
            relation_tokens = [f"[relation{i}]" for i in range(num_relations)]
            num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': relation_tokens})
            logger.info(f"add total special tokens: {num_added_tokens} \n {relation_tokens}")

    def setup(self, s