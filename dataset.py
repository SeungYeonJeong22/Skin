import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class SkinConditionDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(os.path.join(root_dir, split))
        self.image_paths = self.df['image_path']
        self.label = self.df['label']
        if self.label.dtype == 'O':
            self.label_to_int = {label: idx for idx, label in enumerate(self.label.unique())}
            self.int_to_label = {idx: label for label, idx in self.label_to_int.items()}         

        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_name).convert('RGB')
        # label = self.label.iloc[idx]
        
        if self.label.dtype == 'O':
            label = self.df.iloc[idx, 1]
            label = self.label_to_int[label]
        else:
            label = self.label.iloc[idx].astype('int64')

        
        if self.transform:
            image = self.transform(image)

        image = image.float()
        
        return image, label
