# -*- coding: utf-8 -*-
#  Copyright (c) 2024 Kumagai group.
import random
from dataclasses import dataclass
from typing import List, Tuple, Callable

import numpy as np
import torch
from monty.json import MontyDecoder
from torch.utils.data import Dataset, DataLoader, SequentialSampler

from cgcnn.material import Material
from cgcnn.normalizer import DefectDistributions


class MaterialsDataset(Dataset):
    def __init__(self, materials: List[Material]):
        self.materials = materials

    @classmethod
    def from_col(cls, col, formula: List[str] = None):
        decoder = MontyDecoder()
        pd = decoder.process_decoded
        materials = []
        query = {"seed": {"$exists": False}}
        query |= {"formula": {"$in": formula}} if formula else {}
        for doc in col.find(query, {"_id": 0}):
            materials.append(pd(doc))
        return cls(materials)

    def __len__(self):
        return len(self.materials)

    def __getitem__(self, index):
        return self.materials[index]

    @property
    def size(self):
        return len(self)

    def apply_normalization(self, normalizer, plt=None):
        for m in self.materials:
            m.target_vals = normalizer.normed_target_vals(m.target_vals,
                                                          m.charges)


TrainValTestIndices = Tuple[List[int], List[int], List[int]]


@dataclass
class DatasetSplitter:
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

    def train_val_test_indices(self, total_size: int) -> TrainValTestIndices:
        train_size = int(total_size * self.train_ratio)
        val_size = int(total_size * self.val_ratio)

        randomized_indices = list(range(total_size))
        np.random.shuffle(randomized_indices)

        train_indices = randomized_indices[:train_size]
        val_indices = randomized_indices[train_size:train_size + val_size]
        test_indices = randomized_indices[train_size + val_size:]

        return train_indices, val_indices, test_indices
