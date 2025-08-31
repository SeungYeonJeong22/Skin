from models.models import *

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torchvision import transforms
from sklearn.model_selection import train_test_split

from utils.utils import seed, current_time, FocalLoss, device_settings
from utils.get_model import get_model
from test import test

from dataset import SkinConditionDataset
from train_val import train_val
from twoStage_train import twoStageTraining
from twoStage_test import twoStageTest

import argparse
import os
import pandas as pd
import json

try:
    import wandb
except ImportError:
    import subprocess
    import sys
    print("📦 wandb 패키지가 없어서 설치합니다...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Skin Condition Classification')
    parser.add_argument('--batch_size', '-bs', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.00001, help='Learning Rate')
    parser.add_argument('--wandb_flag', '-wf', type=str, default='false', help='wandb_flag')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.01, help='weight_decay')
    parser.add_argument('--num_workers', '-nw', type=int, default=4, help='num_workers')
    parser.add_argument('--num_epochs', '-ne', type=int, default=50, help='num_epochs')
    parser.add_argument('--early_stopping_patience', '-es', type=int, default=5, help='early_stopping_patience')
    parser.add_argument('--model', '-m', type=str, default='efficientb0', help='Model name')
    parser.add_argument('--dataset', '-ds', type=str, default='./data/ISIC_2019', help='Dataset Path')
    parser.add_argument('--training_record_flag', '-tr', type=str, default='false', help='Is Traning Record?')
    parser.add_argument('--tag', '-t', type=str, default="0.0.11", help='Model Taggin Number')

    args = parser.parse_args()
    wandb_flag = args.wandb_flag.lower() == 'true'

    seed(42)
    dataset = args.dataset
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    num_workers = args.num_workers
    num_epochs = args.num_epochs
    lr = args.learning_rate
    early_stopping_patience = args.early_stopping_patience
    model_save_root_path = './save_weights'
    int2label_path = os.path.join(args.dataset, "int2label.json")
    model_name = args.model
    training_record_flag = args.training_record_flag.lower()
    tag = args.tag

    with open(int2label_path, 'r') as f:
        label_dicts = json.load(f)
        int2label = label_dicts['int_to_label']
        label2int = label_dicts['label_to_int']

    os.makedirs(model_save_root_path, exist_ok=True)

    train_df = pd.read_csv(os.path.join(dataset, 'train.csv'))
    val_df = pd.read_csv(os.path.join(dataset, 'valid.csv'))
    print("Train Size : ", len(train_df))
    print("Validation Size : ", len(val_df))        

    labels = train_df['label'].unique()
    num_classes = len(labels)

    device, n_gpus, batch_size, num_workers = device_settings(args)

    save_model_path = current_time() + "_" + model_name + ".pt"
    print("save_model_path : ", save_model_path)

    if model_name == "2stage":
        minority_class_ids = train_df['label'].value_counts().nsmallest(5).index.tolist()

        stage1_path, stage2_path, minority_class_ids = twoStageTraining(
            train_df=train_df,
            val_df=val_df,
            dataset_root=dataset,
            save_dir=model_save_root_path,
            minority_class_ids=minority_class_ids,
            num_classes=num_classes,
            device=device
        )

        twoStageTest(
            val_df=val_df,
            dataset_root=dataset,
            stage1_path=stage1_path,
            stage2_path=stage2_path,
            minority_class_ids=minority_class_ids,
            device=device
        )
    else:
        model, layer_name = get_model(model_name, num_classes)
        model = model.to(device)

        if n_gpus > 1:
            model = nn.DataParallel(model)
            print(" Applied DataParallel")

        # loss_func = nn.CrossEntropyLoss()
        loss_func = FocalLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=1e-5, patience=5)

        params = {
            'dataset': dataset,
            'model_name': model_name,
            'save_model_path': save_model_path,
            'int2label': int2label_path,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'num_workers': num_workers,
            'optimizer': optimizer,
            'weight_decay': weight_decay,
            'loss_func': loss_func,
            'lr': lr,
            'lr_scheduler': lr_scheduler,
            'model_save_root_path': model_save_root_path,
            'early_stopping_patience': early_stopping_patience,
            'tag': tag
        }

        if wandb_flag:
            wandb.init(project="skin-condition-classification", name=save_model_path, config=params)

        train_dataset = SkinConditionDataset(dataframe=train_df, root_dir=dataset, label2int=label2int, mode='train')
        val_dataset = SkinConditionDataset(dataframe=val_df, root_dir=dataset, label2int=label2int, mode='valid') if val_df is not None else None

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

        train_dataset.transform = train_transform
        if val_dataset:
            val_dataset.transform = val_transform

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_dataset else None

        model, loss_hist, metric_hist = train_val(model, params, train_loader, val_loader, int2label=int2label, device=device, layer_name=layer_name, wandb_flag=wandb_flag, training_record_flag=training_record_flag)
        test(args, params, device=device, wandb_flag=wandb_flag)