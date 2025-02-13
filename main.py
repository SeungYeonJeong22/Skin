from torch.utils.data import DataLoader, random_split

from models.efficientnet.enet import enet
from models.vit.vit import vit
from models.convnext.convnext import convnext
from models.custom.fusion import FusionModel

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torchvision import transforms

from utils.utils import seed, current_time, get_activation
from test import test

from dataset import SkinConditionDataset
from train_val import train_val

import wandb
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Skin Condition Classification')
    parser.add_argument('--model', '-m', type=str, default='efficientb0', help='Model name (e.g., efficientb0, efficientb1)')
    parser.add_argument('--batch', '-b', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='Batch size')
    parser.add_argument('--dataset', '-ds', type=str, default='./data/AICamp-2023-Skin-Conditions_Dataset', help='Batch size')
    args = parser.parse_args()
    
    seed(42)
    dataset = args.dataset

    print(f"Dataset: {dataset}")

    batch_size = args.batch
    weight_decay = 0.01
    num_workers = 2
    num_epochs = 50
    lr = args.learning_rate
    early_stopping_patience = 5
    model_save_root_path='./save_weights'

    train_csv_path = os.path.join(dataset, 'train.csv')
    train_df = pd.read_csv(train_csv_path)
    labels = train_df['label'].unique()
    num_classes = len(labels)    

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f'Using device: {device}')
    activation = {}

    # 모델 선택 & Grad-CAM에 쓰기 위한 마지막 레이어 선택
    if args.model.lower().__contains__('vit'):
        model, feature_extractor = vit(num_classes=num_classes)
        layer_name = 'vit.encoder.layer[-1].output'
        model.vit.encoder.layer[-1].output.register_forward_hook(get_activation(activation, layer_name))
    elif args.model.lower().__contains__('efficient'):
        model = enet(model_name=args.model, num_classes=num_classes)
        layer_name = '_blocks[-1]'
        model._blocks[-1].register_forward_hook(get_activation(activation, layer_name))
    elif args.model.lower().__contains__('convnext'):
        model = convnext(num_classes=num_classes)
        layer_name = 'stages[-1]'
        model.stages[-1].register_forward_hook(get_activation(activation, layer_name))
    elif args.model.lower().__contains__('fusion'):
        model = FusionModel(num_classes=num_classes)
        layer_name = 'fusion_conv'
        model.fusion_conv.register_forward_hook(get_activation(activation, layer_name))
    else:
        raise ValueError("Unsupported model type")
        
    model = model.to(device)
    model_name = current_time() + "_" + args.model + ".pt"
    print("model_name : ", model_name)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=1e-5, patience=5)    
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    print("learning_rate: ", lr)
    print("lr_scheduler: ", lr_scheduler)
    print("optimizer: ", optimizer)

    params = {
        'dataset':dataset, 
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

    dataset = SkinConditionDataset(root_dir=dataset, split_file="train.csv")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    print(f'Train size: {train_size}')
    print(f'Validation size: {val_size}')

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


    # ViT는 증강 x
    if args.model.lower().__contains__('vit'):
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model, loss_hist, metric_hist = train_val(model, params, train_loader, val_loader, device=device, activation=activation, layer_name=layer_name)
    test(args, params, device=device)