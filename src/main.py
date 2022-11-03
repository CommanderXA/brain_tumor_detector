import os
import logging
from datetime import datetime

import hydra
from omegaconf import DictConfig

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader

from config import Config
from model import Model
from dataset import MRIDataset
from train import train
from inference import forward


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    # setup
    log = logging.getLogger(__name__)
    Config.setup(cfg, log)
    Config.log.info(f"Performing setup")
    model = Model().to(Config.device)
    Config.log.info(
        "Model has: " + str(model.parameters_count()) + " parameters")
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=Config.cfg.hyperparams.lr)

    if Config.cfg.core.train == True:

        # datasets
        Config.log.info(f"Loading Dataset")
        train_dataset = MRIDataset(
            annotation_file=os.path.join(
                Config.cfg.files.root_dir,
                Config.cfg.files.train + Config.cfg.files.file_ext
            ),
            root_dir=Config.cfg.files.data_dir,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
        )
        val_dataset = MRIDataset(
            annotation_file=os.path.join(
                Config.cfg.files.root_dir,
                Config.cfg.files.val + Config.cfg.files.file_ext
            ),
            root_dir=Config.cfg.files.data_dir,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
        )
        test_dataset = MRIDataset(
            annotation_file=os.path.join(
                Config.cfg.files.root_dir,
                Config.cfg.files.test + Config.cfg.files.file_ext
            ),
            root_dir=Config.cfg.files.data_dir,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
        )

        # dataloaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=Config.cfg.hyperparams.batch_size, shuffle=True, num_workers=3)
        val_dataloader = DataLoader(
            val_dataset, batch_size=Config.cfg.hyperparams.batch_size, shuffle=True, num_workers=3)

        # training
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

    if Config.cfg.core.inference is not False:
        prediction = forward(model)
        Config.log.info(
            f"Image {Config.cfg.core.inference} is classified as: {prediction}")


if __name__ == "__main__":
    main()
