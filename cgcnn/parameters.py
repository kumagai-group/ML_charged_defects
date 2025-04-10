# -*- coding: utf-8 -*-
#  Copyright (c) 2024 Kumagai group.
from dataclasses import dataclass, field
from typing import List

from monty.json import MSONable
from torch import optim


@dataclass
class BondFeaturizerParams:
    cutoff_radius: float = 8.0  # default of cgcnn
    max_num_neighbors: int = 12  # default of cgcnn
    etas: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5])  # Default values from cgcnndefect
    R_offsets: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0, 4.0, 5.0])  # Default values from cgcnndefect

    @property
    def num_bond_features(self):
        return len(self.etas) * len(self.R_offsets)

    @property
    def B(self):
        return self.num_bond_features  # = 15


@dataclass
class HyperParams(MSONable):
    num_cnn: int
    embedding_dim: int
    hidden_dim: int
    num_epoch: int
    learning_rate: float
    dropout_ratio: float
    # self.etas = etas if etas else [0.5, 1.0, 1.5]
    # self.R_offsets = R_offset if R_offset else [1.0, 2.0, 3.0, 4.0, 5.0]

    def __post_init__(self):
        self.milestones: List[int] = [100]
        self.optimizer = optim.Adam

    @property
    def A(self):
        return self.embedding_dim

    def check_if_exists(self, col):
        d = {f"hyperparams.{name}": self.__getattribute__(name)
             for name in self.__dict__}
        return col.exists(d)


@dataclass
class DataParams:
    batch_size: int
    val_ratio: float
    test_ratio: float
    random_seed: int

    @property
    def model_name(self):
        return f"model_v{self.val_ratio}_t{self.test_ratio}_s{self.random_seed}"
