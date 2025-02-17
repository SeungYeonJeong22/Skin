import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.utils import accuracy, seed, current_time
from dataset import SkinConditionDataset
from models.enet import enet
from models.vit import vit
from models.convnext import convnext
from torchvision import transforms
from models.fusion import FusionModel
from models.sejin import Sejin


import argparse

import os
import pandas as pd
import shutil

from tqdm import tqdm

loss_func = nn.CrossEntropyLoss()

# function to evaluate the model on the test set
def evaluate_test_set(model, model_name, test_dl, int_to_label, device='cpu'):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    len_data = len(test_dl.dataset)
    results = []

    with torch.no_grad():
        for xb, yb, img_name in tqdm(test_dl, desc="Processing Test batches"):
            xb = xb.to(device)
            yb = yb.to(device)
            output = model(xb)

            if model_name.lower().__contains__('vit'):
                logits = output.logits if hasattr(output, 'logits') else output
            else:
                logits = output            

            # loss_b = loss_func(logits, yb)
            acc_b = accuracy(logits, yb)

            # test_loss += loss_b.item() * xb.size(0)
            test_acc += acc_b * xb.size(0)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            for i in range(len(xb)):
                results.append({
                    'image': "/".join(img_name[i].split("/")[-2:]),
                    'true_label': int_to_label[yb[i].item()],
                    'predicted_label': int_to_label[preds[i].item()],
                    'probability': f"{probs[i][preds[i]].item():.4f}"
                })

    # test_loss /= len_data
    test_acc /= len_data

    test_acc *= 100

    # return test_loss, test_acc, results
    return test_acc, results


def test(args, params, device='cpu'):
    root_path = params['dataset']
    model_save_root_path = params['model_save_root_path']
    model_name = params['model_name']
    saved_model_path = os.path.join(model_save_root_path, model_name)
    num_workers = params['num_workers']
    root_path = params['dataset']
    root_path = params['dataset']

    test_csv_path = os.path.join(root_path, 'test.csv')
    df = pd.read_csv(test_csv_path)
    labels = df['label'].unique()
    num_classes = len(labels)

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = SkinConditionDataset(root_dir=root_path, split_file='test.csv', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=num_workers)

    test_size = int(len(test_dataset))
    print("Test Size : ", test_size)

    saved_model = saved_model_path.split('/')[-1]
    saved_model_time = "_".join(saved_model.split('.')[0].split("_")[:2])
    saved_model_name = "".join(saved_model.split('.')[0].split("_")[2:])
    
    # 모델 선택 & Grad-CAM에 쓰기 위한 마지막 레이어 선택
    if args.model.lower().__contains__('vit'):
        model, feature_extractor = vit(num_classes=num_classes)
    elif args.model.lower().__contains__('efficient'):
        model = enet(model_name=args.model, num_classes=num_classes)
    elif args.model.lower().__contains__('convnext'):
        model = convnext(num_classes=num_classes)
    elif args.model.lower().__contains__('fusion'):
        model = FusionModel(num_classes=num_classes)
    elif args.model.lower().__contains__('sejin'):
        base_model_name = 'convnext_tiny'
        sub_model_name = 'swin_tiny_patch4_window7_224'
        model = Sejin(base_model_name=base_model_name, sub_model_name=sub_model_name, num_classes=num_classes)
    else:
        raise ValueError("Unsupported model type")     

    model.load_state_dict(torch.load(f'./save_weights/{saved_model}'))
    model = model.to(device)

    test_acc, results = evaluate_test_set(model, saved_model_name, test_loader, test_dataset.int_to_label, device=device)
    print(f'Test Accuracy: {test_acc:.4f}')

    if test_acc < 30:
        print("Test accuracy is below threshold. Moving model to 'low_performance_models' directory.")
        os.makedirs('low_performance_models', exist_ok=True)
        shutil.move(saved_model_path, os.path.join('low_performance_models', os.path.basename(saved_model_path)))
    else:
        os.makedirs('results', exist_ok=True)
        summary = {
            'Time': saved_model_time,
            'Model': saved_model_name,
            'Model Path': saved_model,
            'Test Accuracy': f"{test_acc:.4f}"
        }
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(f'results/{saved_model_time}_{saved_model_name}_results.csv', index=False)

        results_df = pd.DataFrame(results)
        results_df.to_csv(f'results/{saved_model_time}_{saved_model_name}_results.csv', mode='a', index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Skin Condition Classification')
    parser.add_argument('--saved_model', '-m', type=str, default='0211_1413_efficient_b1.pt', help='Model path (e.g., 0211_1413_efficient_b1.pt)')
    parser.add_argument('--dataset', '-ds', type=str, default='./data/AICamp-2023-Skin-Conditions_Dataset', help='Root directory')
    parser.add_argument('--batch', '-b', type=int, default=16, help='Batch size')
    args = parser.parse_args()

    batch_size = args.batch
    weight_decay = 0.01
    num_workers = 2
    num_epochs = 50
    lr = 0.001
    early_stopping_patience = 5
    model_save_root_path='./save_weights'

    params = {
        'dataset':args.dataset, 
        'model_name':args.saved_model,
        'batch_size':batch_size,
        'num_epochs':num_epochs,
        'num_workers':num_workers,
        'model_save_root_path':model_save_root_path,
    }

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")    

    test(args, params, device=device)