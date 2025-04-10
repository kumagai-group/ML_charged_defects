# -*- coding: utf-8 -*-
#  Copyright (c) 2024 Kumagai group.
from dataclasses import dataclass
import random
from typing import List

import numpy as np
import torch


@dataclass
class DataSet:
    random_seed: int
    training_size: int
    validation_size: int
    test_size: int
    training_set: List[List[int, str]]
    validation_set: List[List[int, str]]
    test_set: List[List[int, str]]


@dataclass
class DatasetSampler:
    val_ratio: float
    test_ratio: float
    random_seed: int

    def __post_init__(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

    @property
    def train_ratio(self) -> float:
        return 1. - (self.val_ratio + self.test_ratio)

    def train_val_test_indices(self, total_size: int):
        train_size = int(total_size * self.train_ratio)
        val_size = int(total_size * self.val_ratio)
        train_val_size = train_size + val_size

        randomized_indices = list(range(total_size))
        np.random.shuffle(randomized_indices)

        train_indices = randomized_indices[: train_size]
        val_indices = randomized_indices[train_size: train_val_size]
        test_indices = randomized_indices[train_val_size:]

        return train_indices, val_indices, test_indices
