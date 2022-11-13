import torch
from torchvision import transforms
from PIL import Image

from config import Config
from model import Model
from utils import label_from_num


def get_image(image, transform=None) -> torch.Tensor:
    # data_path = Config.cfg.core.inference
    image = Image.open(image)
    if transform:
        image = transform(image)
    return image.unsqueeze(0)


def forward(model: Model, image) -> tuple[str, float]:
    image = get_image(
        image, 
        transform=transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])).to(Config.device)

    model.eval()  # sets the model to evaluation mode
    out = model(image)
    prediction = label_from_num(out[0])
    return prediction.name, out[0]
