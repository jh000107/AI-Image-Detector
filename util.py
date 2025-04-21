import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ContrastiveImageDataset(Dataset):
    def __init__(self, df, is_train=True, transform=None):
        self.df = df
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['file_name']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.is_train:
            label = self.df.iloc[idx]['label']
            return image, label
        else:
            return image, -1

        return image, label


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, base_transform):
        self.transform = base_transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]