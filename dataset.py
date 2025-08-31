import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import json

class SkinConditionDataset(Dataset):
    def __init__(self, dataframe=None, root_dir=None, label2int=None, mode='train', transform=None):
        self.root_dir = root_dir
        self.df = dataframe
        self.mode = mode

        # ISIC2019 dataset에서 Unknown label 제거
        if self.mode != 'train':
            self.df = self.df[self.df['label'] != 'Unknown']
            self.df = self.df.reset_index(drop=True)

        self.image_paths = self.df['image_path']
        self.label = self.df['label']
        self.num_classes = self.label.nunique()
        
        if self.label.dtype != 'O':
            self.label = self.label.astype('str')

        root_dir_path = root_dir.split("/data/")[0]
        dataset_name = root_dir.split("/data/")[1].split('/')[0]
        # with open(f'{root_dir_path}/data/label/{dataset_name}_int2label.json', 'r') as f:
        #     mapping = json.load(f)
        self.label_to_int = label2int
        #     self.int_to_label = {int(k): v for k, v in mapping['int_to_label'].items()}

        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.mode, self.image_paths[idx])
        img_name = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        label = self.df.iloc[idx, 1]
        label = str(label)
        label = self.label_to_int[label]
        
        if self.transform:
            image = self.transform(image)

        return image, label, img_name
