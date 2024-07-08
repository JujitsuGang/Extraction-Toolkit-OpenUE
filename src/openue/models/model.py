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
        self.num_labels