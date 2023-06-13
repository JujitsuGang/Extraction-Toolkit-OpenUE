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


# In order to ensure reproducible exp