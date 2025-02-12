from torch.utils.data import DataLoader, random_split

from models.efficientnet.enet import enet
from models.vit.vit import vit
from models.convnext.convnext import convnext

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from utils.utils import seed, current_time

from dataset import SkinConditionDataset
from train_val import train_val

import wandb
import argparse
import os
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Skin Condition Classification')
    parser.add_argument('--model', type=str, default='efficientb0', help='Model name (e.g., efficientb0, efficientb1)')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--dataset', type=str, default='./data/AICamp-2023-Skin-Conditions_Dataset', help='Batch size')
    args = parser.parse_args()
    
    seed(42)
    root_path = args.dataset

    print(f"Dataset: {root_path}")

    batch_size = 16
    weight_decay = 0.01
    num_workers = 2
    num_epochs = 50
    lr = 0.001
    early_stopping_patience = 10
    model_save_root_path='./save_weights'

    train_csv_path = os.path.join(root_path, 'train.csv')
    df = pd.read_csv(train_csv_path)
    labels = df['label'].unique()
    num_classes = len(labels)    

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f'Using device: {device}')

    if args.model.lower().__contains__('vit'):
        model, feature_extractor = vit(num_classes=num_classes)
    elif args.model.lower().__contains__('efficient'):
        model = enet(model_name=args.model, num_classes=num_classes)
    elif args.model.lower().__contains__('convnext'):
        model = convnext(num_classes=num_classes)
        
    model = model.to(device)
    model_name = current_time() + "_" + args.model + ".pt"
    print("model_name : ", model_name)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=1e-5, patience=5)    

    params = {
        'dataset':args.dataset, 
        'model_name':model_name,
        'batch_size':batch_size,
        'num_epochs':num_epochs,
        'num_workers':num_workers,
        'optimizer':optimizer,
        'weight_decay':weight_decay,
        'loss_func':loss_func,
        'lr':lr,
        'lr_scheduler':lr_scheduler,
        'model_save_root_path':model_save_root_path,
        'early_stopping_patience':early_stopping_patience
    }


    wandb.init(project="skin-condition-classification", name=model_name, config={
        "batch_size": params['batch_size'],
        "num_epochs": params['num_epochs'],
        "num_workers": params['num_workers'],
        "weight_decay": params['weight_decay'],
        "learning_rate": params['lr'],
        "early_stopping_patience": params['early_stopping_patience']
    })
    config = wandb.config

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(20),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        # transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        # transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = SkinConditionDataset(root_dir=root_path, split="train.csv", transform=train_transform)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    print(f'Train size: {train_size}')
    print(f'Validation size: {val_size}')

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model, loss_hist, metric_hist = train_val(model, params, train_loader, val_loader, device=device)