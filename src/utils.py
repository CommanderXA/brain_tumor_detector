from enum import Enum

import torch

class MRIClasses(Enum):
    Normal = 0
    Tumor = 1

def label_from_num(x: float) -> MRIClasses:
    if x >= 0.5:
        return MRIClasses.Tumor
    else:
        return MRIClasses.Normal


def label_to_num(x: str) -> float:
    if x[0] == "tumor":
        return torch.tensor([[1.]])
    else:
        return torch.tensor([[0.]])


def prediction_to_class(x: float) -> int:
    if x >= 0.5:
        return 1
    else:
        return 0
