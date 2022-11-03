import torch
import torch.optim as optim
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from model import Model


def train(
    epochs: int,
    model: Model,
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    criterion: BCELoss,
    optimizer: optim.Adam
):
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        model.train()
        with tqdm(dataloader) as epoch_data:
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
                epoch_loss += loss.item()/len(dataloader)

                # optimize
                optimizer.step()
                optimizer.zero_grad()

                epoch_data.set_postfix(
                    loss=loss.item(), accuracy=f"{(correct/total*100.):.2f}%")

        total = 0
        with torch.no_grad():
            val_correct = 0
            model.eval()
            with tqdm(val_dataloader) as epoch_val_data:
                for sample, target in epoch_val_data:
                    epoch_val_data.set_description(f"Validation: ")

                    sample = sample.to(Config.device)
                    target = target.to(Config.device)

                    out = model(sample)

                    total += target.size(0)
                    val_correct += (torch.round(out) == target).sum().item()

                    # compute loss
                    loss = criterion(out, target)

                    epoch_data.set_postfix(
                        loss=loss.item(), accuracy=f"{(val_correct/total*100.):.2f}%")
