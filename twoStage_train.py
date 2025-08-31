import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import pandas as pd

from models.twoStage import Stage1Classifier, SwinBaseFeatureExtractor, Stage2FineTuner
from dataset import SkinConditionDataset
from train_val import train_val

def train_stage1(train_df, val_df, dataset_root, save_path, num_classes=23, device='cpu'):
    model = Stage1Classifier(num_classes=num_classes).to(device)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = SkinConditionDataset(train_df, dataset_root, mode='train', transform=train_transform)
    val_dataset = SkinConditionDataset(val_df, dataset_root, mode='valid', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    int_to_label = val_dataset.int_to_label

    params = {
        'num_epochs': 30,
        'loss_func': nn.CrossEntropyLoss(),
        'optimizer': torch.optim.AdamW(model.parameters(), lr=1e-4),
        'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
            torch.optim.AdamW(model.parameters(), lr=1e-4), mode='min'),
        'model_save_root_path': os.path.dirname(save_path),
        'model_name': 'stage1',
        'save_model_path': os.path.basename(save_path),
        'early_stopping_patience': 5
    }

    model, _, _ = train_val(model, params, train_loader, val_loader, device=device)
    torch.save(model.state_dict(), save_path)
    print(f"Stage 1 model saved to {save_path}")

    return int_to_label

def train_stage2(train_df, val_df, dataset_root, save_path, minority_class_ids, int_to_label, device='cpu'):
    train_df = train_df[train_df['label'].isin(minority_class_ids)].reset_index(drop=True)
    val_df = val_df[val_df['label'].isin(minority_class_ids)].reset_index(drop=True)

    swin = SwinBaseFeatureExtractor()
    model = Stage2FineTuner(swin, num_classes=len(minority_class_ids)).to(device)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = SkinConditionDataset(train_df, dataset_root, mode='train', transform=train_transform)
    val_dataset = SkinConditionDataset(val_df, dataset_root, mode='valid', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    params = {
        'num_epochs': 30,
        'loss_func': nn.CrossEntropyLoss(),
        'optimizer': torch.optim.AdamW(model.parameters(), lr=1e-4),
        'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
            torch.optim.AdamW(model.parameters(), lr=1e-4), mode='min'),
        'model_save_root_path': os.path.dirname(save_path),
        'model_name': 'stage2',
        'save_model_path': os.path.basename(save_path),
        'early_stopping_patience': 5
    }

    model, _, _ = train_val(model, params, train_loader, val_loader, device=device)
    torch.save(model.state_dict(), save_path)
    print(f"Stage 2 model saved to {save_path}")

def twoStageTraining(train_df, val_df, dataset_root, save_dir, minority_class_ids, num_classes=23, device='cpu'):
    os.makedirs(os.path.join(save_dir, 'twoStages'), exist_ok=True)

    stage1_path = os.path.join(save_dir, "twoStages", "stage1.pth")
    stage2_path = os.path.join(save_dir, "twoStages", "stage2.pth")

    int_to_label = train_stage1(train_df, val_df, dataset_root, stage1_path, num_classes=num_classes, device=device)
    train_stage2(train_df, val_df, dataset_root, stage2_path, minority_class_ids, int_to_label=int_to_label, device=device)

    return stage1_path, stage2_path, minority_class_ids