# -*- coding: utf-8 -*-
#  Copyright (c) 2024 Kumagai group.
from dataclasses import dataclass
from typing import List

from monty.json import MSONable
from pymatgen.core import Structure
from vise.util.mix_in import ToJsonFileMixIn


@dataclass
class Material(MSONable, ToJsonFileMixIn):
    formula: str
    structure: Structure
    target_site_names: List[str]  # ["Va_O1", ...
    target_site_indices: List[int]
    target_vals: List[float]
    charges: List[int]

    @property
    def num_atoms(self) -> int:
        return len(self.structure)

    @property
    def target_info(self) -> List[List[str]]:
        return [[self.formula, name, charge]
                for name, charge in zip(self.target_site_names, self.charges)]

