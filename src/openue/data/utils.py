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
        tag: current chun