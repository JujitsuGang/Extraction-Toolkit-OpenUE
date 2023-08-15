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


def _