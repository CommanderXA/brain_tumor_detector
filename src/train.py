import torch
import torch.optim as optim
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from model import Model
from utils import label_to_num, prediction_to_class


def train(
    epochs: int,
    model: Model,
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    criterion: BCELoss,
    optimizer: optim.Adam
):
    for epoch in range(epochs):
        correct = 0
        model.train()
        with tqdm(dataloader) as epoch_data:
            for sample, target in epoch_data:
                epoch_data.set_description(f"Epoch: {epoch}/{epochs}")

                sample = sample.to(Config.device)
                target = label_to_num(target).to(Config.device)

                out = model(sample)
                if (prediction_to_class(out) == target).item():
                    correct += 1

                # compute loss
                loss = criterion(out, target)
                loss.backward()

                # optimize
                optimizer.zero_grad()
                optimizer.step()

                epoch_data.set_postfix(
                    loss=loss.item(), accuracy=correct/len(dataloader)*100.)

        with torch.no_grad():
            val_correct = 0
            model.eval()
            with tqdm(val_dataloader) as epoch_val_data:
                for sample, target in epoch_val_data:
                    epoch_val_data.set_description(f"Validation: ")

                    sample = sample.to(Config.device)
                    target = label_to_num(target).to(Config.device)

                    out = model(sample)
                    if prediction_to_class(out) == target:
                        val_correct += 1

                    # compute loss
                    loss = criterion(out, target)

                    epoch_val_data.set_postfix(
                        loss=loss.item(), accuracy=val_correct/len(val_dataloader)*100.)
