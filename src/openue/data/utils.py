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
                        cls_token_segment_id=2 if m