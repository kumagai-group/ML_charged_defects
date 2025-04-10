# -*- coding: utf-8 -*-
#  Copyright (c) 2024 Kumagai group.
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import torch
from monty.json import MSONable
from scipy.optimize import minimize
from vise.util.mix_in import ToJsonFileMixIn

from cgcnn.material import Material


@dataclass
class DefectNormalizer(MSONable, ToJsonFileMixIn):
    shift: float
    mean: float
    std: float

    def normed_target_vals(self,
                           target_vals: List[float],
                           charges: List[int]) -> List[float]:
        target_vals = torch.tensor(target_vals)
        charges = torch.tensor(charges)
        result = (target_vals - self.mean + self.shift * charges) / self.std
        return result.tolist()

    def denormed_target_vals(self,
                             target_vals: torch.tensor,
                             charges: List[int]) -> List[float]:
        charges = torch.tensor(charges)
        return (target_vals * self.std + self.mean - self.shift * charges).tolist()


@dataclass
class Distribution:
    target_vals: List[float]

    @property
    def mean(self):
        return float(np.mean(self.target_vals))

    def shift(self, x, std) -> "Distribution":
        return Distribution([(v + x) / std for v in self.target_vals])

    def __call__(self, *args, **kwargs):
        return self.target_vals


@dataclass
class DefectDistributions:
    distribution_by_charge: Dict[int, Distribution]

    def _hist_by_charge(self, charge, bins):
        return np.histogram(
            self.distribution_by_charge[charge].target_vals, bins=bins)

    def _mean(self, charge) -> float:
        return self.distribution_by_charge[charge].mean

    @property
    def _means(self) -> Dict[int, float]:
        return {charge: self._mean(charge)
                for charge in self.distribution_by_charge.keys()}

    @property
    def _mean_all(self):
        return float(np.mean(self.vals))

    @property
    def _std_all(self):
        return float(np.std(self.vals))

    @property
    def vals(self):
        return [v for dist in self.distribution_by_charge.values()
                for v in dist()]

    @property
    def _max(self):
        return np.max(self.vals)

    @property
    def _min(self):
        return np.min(self.vals)

    @property
    def _shift_for_min_dist(self) -> Tuple[float, "DefectDistributions"]:

        def total_dist_after_shift(x) -> float:
            return self.shifted_distributions(shift=x).total_distances

        result = minimize(total_dist_after_shift, x0=0.0)
        optimal_shift = float(result.x[0])
        return optimal_shift, self.shifted_distributions(optimal_shift)

    @classmethod
    def from_materials(cls, materials: List[Material]) -> "DefectDistributions":
        target_by_charge = defaultdict(list)
        for m in materials:
            for charge, target in zip(m.charges, m.target_vals):
                target_by_charge[charge].append(target)
        result = {k: Distribution(v) for k, v in target_by_charge.items()}
        return cls(result)

    def shifted_distributions(self, shift=0.0, mean=0.0, std=1.0
                              ) -> "DefectDistributions":
        return DefectDistributions(
            {charge: d.shift(x=charge * shift - mean, std=std)
             for charge, d in self.distribution_by_charge.items()})

    @property
    def total_distances(self) -> float:
        result = 0.0
        for (m1, m2) in combinations(self._means.values(), 2):
            result += abs(m1 - m2)
        return result

    @property
    def min_dist_distributions(self) \
            -> Tuple["DefectDistributions", DefectNormalizer]:
        optimal_shift, dists = self._shift_for_min_dist
        new_dists = dists.shifted_distributions(mean=dists._mean_all,
                                                std=dists._std_all)
        normalizer = DefectNormalizer(optimal_shift, dists._mean_all, dists._std_all)
        return new_dists, normalizer

    def plot(self, plt, num_bins=51) -> None:
        plt.title("Energy distribution")

        bins = np.linspace(self._min, self._max, num_bins)
        colors = ["blue", "green", "red"]

        for charge, color in zip(self.distribution_by_charge.keys(), colors):
            counts, bin_edges = self._hist_by_charge(charge, bins)
            plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges),
                    align="edge", edgecolor="black", color=color, alpha=0.5)

        plt.xlabel("Energy")
        plt.ylabel("Count")


def make_normalizer(dist: DefectDistributions):
    _, normalizer = dist.min_dist_distributions
    return normalizer
