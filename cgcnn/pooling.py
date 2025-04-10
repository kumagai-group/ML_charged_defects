# -*- coding: utf-8 -*-
#  Copyright (c) 2024 Kumagai group.
from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn


class Pooling(ABC, nn.Module):

    @abstractmethod
    def __call__(self, embedded_site_features):
        pass


class SitePooling(Pooling):
    def __init__(self, target_site_indices: List[int]):
        super().__init__()
        self.target_site_indices = target_site_indices

    def __call__(self, embedded_site_features):
        return torch.stack(
            [embedded_site_features[i] for i in self.target_site_indices])


class AveragePooling(Pooling):
    def __init__(self, num_atoms: List[int], num_target_sites: List[int]):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_target_sites = num_target_sites

    def __call__(self, embedded_site_features):
        features = []
        prev_natom = 0
        for natom, nsites in zip(self.num_atoms, self.num_target_sites):
            ave_site_fea = embedded_site_features[prev_natom:natom+prev_natom].mean(dim=0)
            A = len(ave_site_fea)
            features.extend(
                ave_site_fea.expand(nsites, A)
                )
            prev_natom += natom
        return torch.stack(features)



