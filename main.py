"""Experiment-running framework."""
import argparse
import importlib

import numpy as np
import torch
import pytorch_lightning as pl
import openue.lit_models as lit_models
import yaml
import time
from openue.lit_models import MyTrainer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In order to ensure reproducible experiments, we must set random seeds.


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
	
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--litmodel_class", type=str, default="SEQLitModel")
    parser.add_argument("--data_class", type=str, default="REDataset")
    parser.add_argument("--model_class", type=str, default="bert.BertForSequenceClassification")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"openue.data.{temp_args.data_class}")
    model_class = _import_class(f"openue.models.{temp_args.model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argum