
from logging import debug
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
# Hide lines below until Lab 5
import wandb
import numpy as np
from openue.data.utils import OutputExample

from openue.models.model import Inference
# Hide lines above until Lab 5

from .base import BaseLitModel
from .metric import compute_f1, acc, compute_metrics, seq_metric
from transformers.optimization import (
    get_linear_schedule_with_warmup,
)
from transformers import AutoConfig, AutoTokenizer
from functools import partial
from openue.models import BertForRelationClassification, BertForNER
from openue.data import get_labels_ner

from functools import partial

class RELitModel(BaseLitModel):
    def __init__(self, args, data_config):
        super().__init__(args, data_config)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.num_tokens = data_config['num_tokens']
        label_map_ner = get_labels_ner()
        self.eval_fn = partial(compute_metrics,label_map_ner=label_map_ner)
        self.best_f1 = 0
        
        self._init_model()

    def forward(self, x):
        return self.model(x)

    def _init_model(self):
        #TODO put the parameters from the data_config to the config, maybe use the __dict__?
        config = AutoConfig.from_pretrained(self.args.model_name_or_path)
        config.num_labels = self.data_config['num_labels']
        self.model = BertForNER.from_pretrained(self.args.model_name_or_path, config=config)
        self.model.resize_token_embeddings(self.data_config['num_tokens'])
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        self.config = config

    def _init_label_embedding(self):
        #TODO put the right meaning into the [relation{i}]
        pass

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        loss, logits = self.model(**batch)
        self.log("Train/loss", loss)
        return loss