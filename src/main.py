import os
import logging
from datetime import datetime

import hydra
from omegaconf import DictConfig

import torch
from torch.nn import BCELoss
from torch.optim import SGD
from torchvision import transforms
from torch.utils.data import DataLoader

from config import Config
from model import Model
from dataset import MRIDataset
from train import train


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    # setup
    log = logging.getLogger(__name__)
    Config.set_device()
    Config.set_cfg(cfg)
    Config.set_log(log)
    Config.log.info(f"Performing setup")
    model = Model().to(Config.device)
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=Config.cfg.hyperparams.lr)

    # datasets
    Config.log.info(f"Loading Dataset")
    train_dataset = MRIDataset(
        annotation_file=os.path.join(
            Config.cfg.files.root_dir,
            Config.cfg.files.train + Config.cfg.files.file_ext
        ),
        root_dir=Config.cfg.files.data_dir,
        transform=transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((256, 256))
        ])
    )
    val_dataset = MRIDataset(
        annotation_file=os.path.join(
            Config.cfg.files.root_dir,
            Config.cfg.files.val + Config.cfg.files.file_ext
        ),
        root_dir=Config.cfg.files.data_dir,
        transform=transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((256, 256))
        ])
    )
    test_dataset = MRIDataset(
        annotation_file=os.path.join(
            Config.cfg.files.root_dir,
            Config.cfg.files.test + Config.cfg.files.file_ext
        ),
        root_dir=Config.cfg.files.data_dir,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    # dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=3)
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=True, num_workers=3)

    # training
    if Config.cfg.core.train == True:
        Config.log.info(f"Starting training")
        train(
            Config.cfg.hyperparams.epochs,
            model,
            train_dataloader,
            val_dataloader,
            criterion,
            optimizer
        )

        Config.log.info(f"Training finished")

        # model save
        if not os.path.isdir("../models/"):
            os.mkdir("../models/")

        torch.save(
            model, f"../models/model_{datetime.now().strftime('%Y-%m-%d_%H:%M')}.pt"
        )
    
    # for i, data in enumerate(train_dataloader):
    #     sample, target = data
    #     sample = sample.to(Config.device)
    #     print(sample.shape)
    #     model(sample)


if __name__ == "__main__":
    main()
