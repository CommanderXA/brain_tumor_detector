import torch
import torch.optim as optim
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from model import Model


def train(
    epochs: int,
    model: Model,
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    criterion: BCELoss,
    optimizer: optim.Adam
):
    writer = SummaryWriter()
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        # training
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
