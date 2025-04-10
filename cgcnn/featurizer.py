# -*- coding: utf-8 -*-
#  Copyright (c) 2024 Kumagai group.
import itertools
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from pymatgen.core import Element
from torch import Tensor

from cgcnn.cgcnn_module import CgcnnFeatures


dir_ = Path(__file__).parent
with open(dir_ / "atom_init.json") as f:
    elem_embed = json.load(f)


def element_featurize(element: Element) -> List[float]:
    return elem_embed[str(element.Z)]


class ElementFeaturizer:

    def __init__(self, type_: str = ""):
        self.type = type_
        if type_ == "Z":
            self.num_feature = 1
        else:
            self.num_feature = 92

    def featurize(self, element: Element) -> List[float]:
        if self.type == "Z":
            return [element.Z]
        else:
            return element_featurize(element)  # 92


class BondFeaturizer(ABC):
    def __init__(self, cutoff_radius, max_num_neighbors):
        self.cutoff = cutoff_radius
        self.max_num_neighbors = max_num_neighbors

    @abstractmethod
    def apply(self, structure) -> Tuple[Tensor, List[list]]:
        pass

    def _get_bond_info(self, structure):
        distances, bond_indices = [], []
        for nbr in get_sorted_neighbors(structure, self.cutoff):
            if len(nbr) < self.max_num_neighbors:
                raise ValueError("Not enough neighbors to build graph. "
                                 "Increase radius.")
            distances.append(
                [n[1] for n in nbr[:self.max_num_neighbors]])
            bond_indices.append(
                [n[2] for n in nbr[:self.max_num_neighbors]])
        return torch.tensor(distances), bond_indices

    @property
    @abstractmethod
    def num_bond_features(self):
        pass

    @property
    def B(self):
        return self.num_bond_features


def get_sorted_neighbors(structure, cutoff_radius):
    all_neighbors = structure.get_all_neighbors(cutoff_radius,
                                                include_index=True)
    return [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_neighbors]


class GaussianBasis(BondFeaturizer):
    def __init__(self, cutoff_radius, max_num_neighbors, etas: List[float],
                 R_offset: List[float]):
        super().__init__(cutoff_radius, max_num_neighbors)
        self.etas = etas
        self.R_offsets = R_offset

    @property
    def num_bond_features(self) -> int:
        return len(self.etas) * len(self.R_offsets)

    def apply(self, structure) -> Tuple[Tensor, List[list]]:
        """
        R_ij : np.ndarray (N_atom, N_neighbor)
            The interatomic distances as a Tensor
        """
        R_ij, bond_indices = self._get_bond_info(structure)

        eta_R_pairs = torch.tensor(list(itertools.product(self.etas,
                                                          self.R_offsets)))
        etas, Rs = eta_R_pairs[:, 0], eta_R_pairs[:, 1]

        # broadcast
        R_ij = R_ij.unsqueeze(-1)  # サイズ: [24, 12, 1]
        etas = etas[None, None, :]  # サイズ: [1, 1, 15]
        Rs = Rs[None, None, :]  # サイズ: [1, 1, 15]

        return torch.exp(-etas * ((R_ij - Rs) ** 2)), bond_indices


class DistanceBinsBasis(BondFeaturizer):
    def __init__(self, cutoff_radius, max_num_neighbors,
                 min_max=None, num_bins=10):
        super().__init__(cutoff_radius, max_num_neighbors)
        min_max = min_max if min_max is not None else [0.7, 5.2]

        self.pts = np.linspace(min_max[0], min_max[1],
                               num=num_bins+1, endpoint=True)
        self.num_bins = num_bins

    @property
    def num_bond_features(self):
        return self.num_bins

    def apply(self, structure):
        R_ij, bond_indices = self._get_bond_info(structure)

        one_hot_shape = R_ij.shape + (self.num_bins,)  # 元の形状 + 区間数
        one_hot_encoded = torch.zeros(one_hot_shape, dtype=torch.int)

        for i in range(self.num_bins):
            mask = (self.pts[i] <= R_ij) & (R_ij < self.pts[i + 1])  # 条件に合う要素を探す
            one_hot_encoded[..., i][mask] = 1

        return one_hot_encoded, bond_indices


class CgcnnFeaturizer:
    def __init__(self,
                 bond_featurizer: BondFeaturizer,
                 element_featurizer: ElementFeaturizer):
        self.bond_featurizer = bond_featurizer
        self.elem_featurizer = element_featurizer

    def __call__(self, structure) -> CgcnnFeatures:
        return self.get_features(structure)

    def get_features(self, structure) -> CgcnnFeatures:
        site_features = [self.elem_featurizer.featurize(site.specie)
                         for site in structure]
        bond_features, bond_indices = self.bond_featurizer.apply(structure)

        return CgcnnFeatures(
            site_features=torch.tensor(site_features, dtype=torch.float),
            bond_features=bond_features.float(),
            bond_indices=torch.tensor(bond_indices))
