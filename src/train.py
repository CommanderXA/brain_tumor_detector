import os

import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MRIDataset
from config import Config
from model import Model


def train(
    epochs: int,
    model: Model,
):
    writer = SummaryWriter()
    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=Config.cfg.hyperparams.lr)

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
    test_dataloader = DataLoader(
        test_dataset, batch_size=Config.cfg.hyperparams.batch_size, shuffle=True, num_workers=3)

    min_val_loss = 1000
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        # training
        model.train()
        with tqdm(train_dataloader) as epoch_data:
            for sample, target in epoch_data:
                epoch_data.set_description(f"Epoch: {epoch}/{epochs}")

                sample = sample.to(Config.device)
                target = target.to(Config.device)

                out = model(sample)
                total += target.size(0)
                correct += (torch.round(out) == target).sum().item()

                # compute loss
                loss = criterion(out, target)
                loss.backward()
                epoch_loss += loss.item()/len(train_dataloader)

                # optimize
                optimizer.step()
                optimizer.zero_grad()

                epoch_data.set_postfix(
                    loss=loss.item(), accuracy=f"{(correct/total*100.):.2f}%")

        # writing Accuracy and Loss of training to the TensorBoard
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", correct/total, epoch)

        # validation
        epoch_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            model.eval()
            with tqdm(val_dataloader) as epoch_val_data:
                for sample, target in epoch_val_data:
                    epoch_val_data.set_description(f"Validation: ")

                    sample = sample.to(Config.device)
                    target = target.to(Config.device)
                    out = model(sample)

                    total += target.size(0)
                    correct += (torch.round(out) == target).sum().item()

                    # compute loss
                    loss = criterion(out, target)
                    epoch_loss += loss.item() / len(val_dataloader)

                    epoch_val_data.set_postfix(
                        loss=loss.item(), accuracy=f"{(correct/total*100.):.2f}%")

        # writing Accuracy and Loss of validation to the TensorBoard
        writer.add_scalar("Loss/val", epoch_loss, epoch)
        writer.add_scalar("Accuracy/val", correct/total, epoch)

        # save the best model with accordance to the validation loss
        if min_val_loss > epoch_loss:
            min_val_loss = epoch_loss
            torch.save(model, os.path.join(
                Config.cfg.files.models_dir, Config.training_model_name))

    # testing
    correct = 0
    epoch_loss = 0
    total = 0
    model.eval()
    with torch.no_grad():
        model.eval()
        with tqdm(test_dataloader) as epoch_data:
            for sample, target in epoch_data:
                epoch_data.set_description(f"Testing: ")

                sample = sample.to(Config.device)
                target = target.to(Config.device)

                out = model(sample)

                total += target.size(0)
                correct += (torch.round(out) == target).sum().item()

                # compute loss
                loss = criterion(out, target)
                epoch_loss += loss.item() / len(test_dataloader)

                epoch_data.set_postfix(
                    loss=loss.item(), accuracy=f"{(correct/total*100.):.2f}%")

        # writing Accuracy and Loss of validation to the TensorBoard
        writer.add_scalar("Loss/test", epoch_loss, epoch)
        writer.add_scalar("Accuracy/test", correct/total, epoch)

    writer.close()
