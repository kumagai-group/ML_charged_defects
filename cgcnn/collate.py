# -*- coding: utf-8 -*-
#  Copyright (c) 2024 Kumagai group.
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import torch

from cgcnn.cgcnn_module import CgcnnFeatures
from cgcnn.featurizer import ElementFeaturizer, BondFeaturizer, CgcnnFeaturizer
from cgcnn.material import Material


@dataclass
class BatchedCgcnnInfo:
    cgcnn_features: List[CgcnnFeatures]
    # [material][prop][site, weight]
    target_infos: List[List[List[str]]]  # [[["MgO", "O1", 0], ...]
    target_site_indices: List[List[int]]
    site_charges: List[List[int]]  # If expanded_charges are not related, set 0.
    target_vals: List[List[float]] = None
    is_gpu: bool = False

    @property
    def num_atoms(self) -> List[int]:  # in each structure
        return [fea.N for fea in self.cgcnn_features]

    @property
    def num_target_sites(self) -> List[int]:
        return [len(t) for t in self.target_site_indices]

    @property
    def batched_cgcnn_features(self):
        features = self.cgcnn_features
        site_features = torch.cat([f.site_features for f in features], dim=0)
        bond_features = torch.cat([f.bond_features for f in features], dim=0)
        bond_indices = []
        prev_n_atom = 0
        for features_by_mat, n_atom in zip(self.cgcnn_features,
                                           self.num_atoms):
            bond_indices.append(features_by_mat.bond_indices + prev_n_atom)
            prev_n_atom += n_atom

        bond_indices = torch.cat(bond_indices, dim=0)

        return CgcnnFeatures(site_features, bond_features, bond_indices)

    @property
    def target_infos_in_batch(self) -> List[List[str]]:
        return [t for target_infos in self.target_infos for t in target_infos]

    @property
    def site_charges_in_batch(self) -> List[int]:
        return [c for site_charges in self.site_charges for c in site_charges]

    @property
    def target_site_indices_in_batch(self) -> List[int]:
        result = []
        prev_n_atom = 0
        for indices, n_atom in zip(self.target_site_indices, self.num_atoms):
            result.extend([site_idx + prev_n_atom for site_idx in indices])
            prev_n_atom += n_atom

        return result

    @property
    def target_vals_in_batch(self) -> List[float]:
        return np.concatenate(self.target_vals).tolist()


class MaterialsCollater:
    def __init__(self,
                 bond_featurizer: BondFeaturizer,
                 element_featurizer: ElementFeaturizer):
        self.bond_featurizer = bond_featurizer
        self.element_featurizer = element_featurizer

    def collate_materials(self, materials: List[Material]):
        featurizer = CgcnnFeaturizer(self.bond_featurizer,
                                     self.element_featurizer)
        features = [featurizer(m.structure) for m in materials]

        return BatchedCgcnnInfo(
            cgcnn_features=features,
            target_infos=[m.target_info for m in materials],
            site_charges=[m.charges for m in materials],
            target_site_indices=[m.target_site_indices for m in materials],
            target_vals=[m.target_vals for m in materials])

