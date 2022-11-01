import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class MRIDataset(Dataset):
    """MRI dataset of Human brains"""

    def __init__(self, annotation_file: str, root_dir: str, transform=None) -> None:
        """
        Args:
            annotation_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.labels = pd.read_csv(annotation_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> any:
        data_path = os.path.join(self.root_dir, self.labels.iloc[index, 1])
        image = Image.open(data_path)
        label = self.labels.iloc[index, 2]
        if self.transform:
            image = self.transform(image)
        return image, label
