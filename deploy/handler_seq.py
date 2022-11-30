from abc import ABC
import enum
import json
import logging
import os
import ast
from posixpath import realpath
import torch
import transformers
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
)
from transformers.models.bert.configuration_bert import BertConfig
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s",transformers.__version__)


from model import BertForNER, BertForRelationClassification


class BertForSEQHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super().__init__()
        self.initialized = False
	
    def initialize(self, ctx):
