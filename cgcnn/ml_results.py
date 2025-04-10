# -*- coding: utf-8 -*-
#  Copyright (c) 2024 Kumagai group.
from collections import defaultdict
from dataclasses import dataclass
from typing import List

import torch
from monty.json import MSONable, MontyDecoder
from numpy import sqrt
from tabulate import tabulate
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchmetrics import R2Score
from torchmetrics.functional import mean_absolute_error as mae, \
    mean_squared_error as mse

from cgcnn.parameters import HyperParams

decoder = MontyDecoder()
pd = decoder.process_decoded

batch_size = 32


@dataclass
class Accuracy(MSONable):
    formula: str
    site: str
    charge: int
    predicted: float
    actual: float

    def match(self, formula, site, charge):
        return self.formula == formula and self.site == site and self.charge == charge

    @property
    def error(self):
        return self.actual - self.predicted

    @property
    def abs_error(self):
        return abs(self.actual - self.predicted)

    def __str__(self):
        return f'{self.formula} {self.site} {self.charge}: pred={self.predicted}, actual={self.actual}'


@dataclass
class MLResult(MSONable):
    hyperparams: HyperParams
    train_mae: float
    val_mae: float
    test_mae: float


def color_by_charge(charge: int):
    if charge == 0:
        return "b"
    elif charge == 1:
        return "g"
    elif charge == 2:
        return "r"
    else:
        raise ValueError("charge must be 0, 1, or 2")


@dataclass
class DetailedMLResult(MSONable):
    hyperparams: HyperParams
    train_accuracies: List[Accuracy]
    val_accuracies: List[Accuracy]
    test_accuracies: List[Accuracy]

    def query(self, dataset, formula, site, charge):
        accuracies = self.__getattribute__(f"{dataset}_accuracies")
        for accuracy in accuracies:
            if accuracy.match(formula, site, charge):
                return accuracy

        raise ValueError(f"No match for {formula} {site} {charge}")

    def __str__(self):
        data = [["", "# data", "MAE (eV)", "RMSE (eV)", "R2"]]
        for type_ in ["train", "val", "test"]:
            attr = f"{type_}_accuracies"
            x = self.__getattribute__(attr)
            try:
                r2 = self.get_stat(type_, "R2", 4)
            except ValueError:
                r2 = None
            data.append([type_, len(x),
                         self.get_stat(type_, "mae", 4),
                         self.get_stat(type_, "rmse", 4), r2])

        if len(self.charge_set) > 1:
            for c in self.charge_set:
                length = len([1 for a in self.test_accuracies if a.charge == c])
                data.append([f"test_charge{c}", length,
                             self.get_stat(type_, "mae", 4, charges=[c]),
                             self.get_stat(type_, "rmse", 4, charges=[c]),
                             self.get_stat(type_, "R2", 4, charges=[c])
                             ])

        table = tabulate(data, tablefmt="simple")
        return "\n".join([str(self.hyperparams), table])

    def outliers(self, threshold: float):
        result = []
        for accuracy in self.test_accuracies:
            if accuracy.abs_error > threshold:
                result.append(accuracy)
        return result

    def outliers_str(self, threshold: float):
        result = []
        for a in self.outliers(threshold):
            result.append(a.__str__())
        return "\n".join(result)

    def ml_result(self):
        return MLResult(self.hyperparams,
                        self.get_stat("train", "mae", 4),
                        self.get_stat("val", "mae", 4),
                        self.get_stat("test", "mae", 4))

    @property
    def charge_set(self) -> set:
        charges = ([a.charge for a in self.train_accuracies]
                   + [a.charge for a in self.val_accuracies]
                   + [a.charge for a in self.test_accuracies])
        return set(charges)

    def get_stat(self,
                 data_name: str,
                 stat_type_name: str,
                 digit=None,
                 charges: List[int] = None,
                 exclude: List[str] = None):
        attr = f"{data_name}_accuracies"
        pred, actual = [], []
        for a in self.__getattribute__(attr):
            if charges and a.charge not in charges:
                continue
            if exclude and a.formula in exclude:
                print(f"{a.formula} is excluded.")
                continue
            pred.append(a.predicted)
            actual.append(a.actual)

        pred, actual = torch.tensor(pred), torch.tensor(actual)

        stat_type = mae if stat_type_name == "mae" \
            else mse if stat_type_name == "rmse" \
            else R2Score() if stat_type_name == "R2" else None
        if stat_type is None:
            raise ValueError
        result = float(stat_type(pred, actual))
        result = sqrt(result) if stat_type_name == "rmse" else result
        return result if digit is None else round(result, digit)

    def test_mae(self, exclude=None):
        return self.get_stat("test", "mae", exclude=exclude)

    def get_pred_actual(self, data_name: str, charge: int):
        predicted, actual = [], []
        for accuracy in self.__getattribute__(f"{data_name}_accuracies"):
            if accuracy.charge == charge:
                predicted.append(accuracy.predicted)
                actual.append(accuracy.actual)
        return predicted, actual

    def _values(self, only_test: bool):
        result = [a.predicted for a in self.test_accuracies] + [a.actual for a in self.test_accuracies]
        if only_test is False:
            result.extend([a.predicted for a in self.train_accuracies]
                          + [a.actual for a in self.train_accuracies])
        return result

    def _min(self, only_test: bool):
        return min(self._values(only_test))

    def _max(self, only_test: bool):
        return max(self._values(only_test))

    def _range_min(self, only_test: bool):
        return self._min(only_test) * 1.1 - self._max(only_test) * 0.1

    def _range_max(self, only_test: bool):
        return self._max(only_test) * 1.1 - self._min(only_test) * 0.1

    def plot_parity(self, plt, size=30, title=None, only_test=True, legend=False):
        plt.figure(figsize=(8, 8))
        if only_test:
            plot_info = ["test"], ["o"], [0.5]
        else:
            plot_info = ["train", "test"], ["x", "o"], [0.3, 0.6]

        for data, marker, alpha in zip(*plot_info):
            for charge in self.charge_set:
                pred, actual = self.get_pred_actual(data, charge=charge)
                plt.scatter(actual, pred, alpha=alpha,
                            label=f"{data}_charge{charge}",
                            marker=marker, s=size,
                            edgecolors='none',
                            color=color_by_charge(charge))

        range_ = [self._range_min(only_test), self._range_max(only_test)]
        plt.xlim(range_)
        plt.ylim(range_)
        plt.plot(range_, range_, color="black", linestyle="--")
        plt.xlabel("Actual values (eV)")
        plt.ylabel("Predicted values (eV)")
        if title:
            plt.title(title)
        if legend:
            plt.legend()

        plt.grid(True)


def accuracies_to_dict(accuracies: List[Accuracy]) -> dict:
    # d["MgO"]["O1"][0]
    d = defaultdict(lambda: defaultdict(dict))
    for x in accuracies:
        d[x.formula][x.site][x.charge] = (x.predicted, x.actual)
    return dict(d)


@dataclass
class MLResults(MSONable):
    seed: int
    num_data: int
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    ml_results: List[MLResult] = None
    best_ml_results: DetailedMLResult = None

    def __post_init__(self):
        if self.ml_results is None:
            self.ml_results = []

    def append_ml_result(self, detailed_ml_result: DetailedMLResult, col):
        """ Return if the appended accuracy is the best."""
        new_ml_result = detailed_ml_result.ml_result()
        self.ml_results.append(new_ml_result)

        q = {f"seed": self.seed}
        col.update_one(q, {"$push": {"ml_results": new_ml_result.as_dict()}})

        is_best = (self.ml_results[-1].test_mae
                   == min(lr.test_mae for lr in self.ml_results))
        if is_best:
            print("The best hyperparameter is updated.")
            self.best_ml_results = detailed_ml_result
            col.update_one(q, {"$set": {"best_ml_results": detailed_ml_result.as_dict()}})

        return is_best

    def loaders(self, dataset, collate_fn):
        loader_kwargs = dict(dataset=dataset,
                             batch_size=batch_size,
                             collate_fn=collate_fn)

        train_sampler = SubsetRandomSampler(self.train_indices)
        train_loader = DataLoader(sampler=train_sampler, **loader_kwargs)

        val_sampler = SubsetRandomSampler(self.val_indices)
        val_loader = DataLoader(sampler=val_sampler, **loader_kwargs)

        test_sampler = SubsetRandomSampler(self.test_indices)
        test_loader = DataLoader(sampler=test_sampler, **loader_kwargs)

        return train_loader, val_loader, test_loader

    @classmethod
    def find_doc(cls, col, seed: int) -> "MLResults":
        doc = col.find_one({"seed": seed}, {})
        if doc is None:
            raise ValueError(f'No doc for seed {seed}.')
        return pd(doc)

