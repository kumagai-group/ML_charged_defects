# -*- coding: utf-8 -*-
#  Copyright (c) 2024 Kumagai group.
from typing import List

import torch

from cgcnn.ml_results import Accuracy


class EvalAccuracy:

    def __init__(self, model, normalizer):
        self.model = model
        self.denorm = normalizer.denormed_target_vals

    def __call__(self, loader) -> List[Accuracy]:
        self.model.eval()

        result = []
        with torch.no_grad():
            for batch in loader:
                predictions = self.model(batch).cpu()
                targets = torch.tensor(batch.target_vals_in_batch)
                charges = batch.site_charges_in_batch

                predictions = self.denorm(predictions, charges)
                targets = self.denorm(targets, charges)

                for p, a, (f, s, c) in zip(predictions, targets,
                                              batch.target_infos_in_batch):
                    result.append(Accuracy(formula=f, site=s, charge=c,
                                           predicted=p, actual=a))
        return result