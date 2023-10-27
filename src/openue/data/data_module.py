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
        self.num_labels = le