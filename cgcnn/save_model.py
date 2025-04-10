# -*- coding: utf-8 -*-
#  Copyright (c) 2024 Kumagai group.
from pathlib import Path

import torch

from cgcnn.normalizer import DefectNormalizer


def save_model(model, normalizer: DefectNormalizer, filename, dirname="models") -> Path:
    torch_path = Path(dirname) / f"{filename}.pth"
    torch.save(model, torch_path)
    print(f"Model saved to {torch_path}")

    normalizer_path = Path(dirname) / f"{filename}.json"
    normalizer.to_json_file(normalizer_path)
    print(f"Normalizer saved to {normalizer_path}")
    return torch_path
