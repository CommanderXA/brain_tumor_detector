from enum import Enum

import torch

from config import Config


class MRIClasses(Enum):
    Normal = 0
    Tumor = 1


def label_from_num(x: float) -> MRIClasses:
    if x >= 0.5:
        return MRIClasses.Tumor
    else:
        return MRIClasses.Normal


def label_to_num(x: str) -> torch.Tensor:
    if x == "tumor":
        x = 1.
    else:
        x = 0.
    
    y = torch.Tensor([x])
    return y


def prediction_to_class(x: float) -> int:
    if x >= 0.5:
        return 1.
    else:
        return 0.
