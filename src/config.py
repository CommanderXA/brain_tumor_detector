import torch
from omegaconf import DictConfig


class Config:
    cfg = None
    device = torch.device("cpu")
    log = None

    @classmethod
    def set_cfg(cls, cfg: DictConfig) -> None:
        cls.cfg = cfg

    @classmethod
    def set_device(cls) -> None:
        cls.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def set_log(cls, log) -> None:
        cls.log = log
