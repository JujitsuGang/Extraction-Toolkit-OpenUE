from .data import REDataset
from .lit_models import RELitModel, SEQLitModel
from .models import BertForNER, BertForRelationClassification

import importlib
import argparse


def _import_class(module_and_class_name: str) -> type:

    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
	
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    # trainer_parser = pl.Trainer.add_argparse_args(parser)
    # trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access