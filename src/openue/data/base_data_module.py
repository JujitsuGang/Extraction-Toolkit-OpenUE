"""Base DataModule class."""
from pathlib import Path
from typing import Dict
import argparse
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class Config(dict):
    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, val):
        self[name] = val


BATCH_SIZE = 8
NUM_WORKERS = 8


class BaseDataModule(pl.LightningDataModule):
    """
    Base DataModule.
    Learn more at ht