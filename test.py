from models.models import *

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.utils import eval_metric, device_settings
from utils.get_model import get_model
from dataset import SkinConditionDataset
from torchvision import transforms
from torch.serialization import add_safe_globals
from torch.serialization import safe_globals
import onnxruntime as rt

import argparse
import numpy as np

import os
import pandas as pd

from tqdm import tqdm
import wandb
import json

loss_func = nn.CrossEntropyLoss()

# function to evaluate the model on the test set
def evaluate_test_set(model, model_name, df=None, test_dl=None, dataset_name=None, int2label=None, device='cpu', wandb_flag=True):
    model.eval()
    acc = 0.0
    precision = 0.0
    recall = 0.0
    f1_score = 0.0
    total_weight = 0.0
    test_acc = 0.0
    len_data = len(test_dl.dataset)
    results = []

    print("dataset_name : ", dataset_name)

    with torch.no_grad():
        for xb, yb, img_name in tqdm(test_dl, desc="Processing Test batches"):
            xb = xb.to(device)
            yb = yb.to(device)
            output = model(xb)

            if model_name.lower() == 'vit':
                logits = output.logits if hasattr(output, 'logits') else output
            elif model_name.lower().__contains__("swin_transformer"):
                logits = output
                if output.ndim == 4:
                    output = output.permute(0, 3, 1, 2) # [B, C, H, W]
                    logits = torch.nn.functional.adaptive_avg_pool2d(output, 1).flatten(1)
            else:
                logits = output

            # ISIC 데이터 셋 처리 (class 가중치)
            if dataset_name.__contains__("ISIC_2019"):
                prob, max_prob, pred, target, acc_b = eval_metric(logits, yb, len(int2label), isic_flag=True)
            else:
                prob, max_prob, pred, target, acc_b = eval_metric(logits, yb, len(int2label))

            acc += float(acc_b) * len(xb)
            # precision += float(precision_b) * len(xb)
            # recall += float(recall_b) * len(xb)
            # f1_score += float(f1_score_b) * len(xb)

            if np.isnan(acc):
                pass

            try:
                for i in range(len(xb)):
                    # image = ("/".join(img_name[i].split("/")[-2:]))
                    # image = os.path.basename(img_name[i])
                    # true_label = int2label[str(yb[i].item())]
                    # predicted_label = int2label[str(pred[i].item())]
                    # probability = f"{prob[i][pred[i]].item():.4f}"

                    results.append({
                        'image': os.path.basename(img_name[i]),
                        'true_label': int2label[str(yb[i].item())],
                        'predicted_label' : int2label[str(pred[i].item())],
                        'probability' :f"{prob[i][pred[i]].item():.4f}"
                    })

            except Exception as e:
                print(i,  " yb[i].item() : ", yb[i].item())


    test_acc = (acc / len_data) * 100
    test_precision = (precision / len_data) * 100
    test_recall = (recall / len_data) * 100
    test_f1_score = (f1_score / len_data) * 100

    print(f"Acc: {test_acc}\nPrecision: {test_precision}\nRecall: {test_recall}\nF1: {test_f1_score}")

    if wandb_flag:
        wandb.log({'test_acc': test_acc})

    return test_acc, test_precision, test_recall, test_f1_score, results


def test(args, params, device='cpu', wandb_flag=True):
    dataset = params['dataset']
    model_save_root_path = params['model_save_root_path']
    model_name = params['model_name']
    save_model_path = params['save_model_path']
    batch_size = params['batch_size']
    tag = params['tag']
    saved_model_path = os.path.join(model_save_root_path, save_model_path)
    int2label_path = os.path.join(args.dataset, "int2label.json")

    num_workers = params['num_workers']
    learning_rate = params['lr']
    with open(int2label_path, 'r') as f:
        label_dicts = json.load(f)
        int2label = label_dicts['int_to_label']
        label2int = label_dicts['label_to_int']

    test_csv_path = os.path.join(dataset, 'test.csv')

    test_df = pd.read_csv(test_csv_path)
    test_df = test_df[test_df['label'] != 'Unknown']
    test_df = test_df.reset_index(drop=True)

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    labels = test_df['label'].unique()
    num_classes = len(labels)    

    test_size = int(len(test_df))
    print("Test Size : ", test_size)

    test_dataset = SkinConditionDataset(dataframe=test_df, root_dir=dataset, label2int=label2int, mode='test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    saved_model = saved_model_path.split('/')[-1]
    saved_model_time = "_".join(saved_model.split('.')[0].split("_")[:2])

    # 원본
    model, layer_name = get_model(model_name, num_classes)  
    model.load_state_dict(torch.load(f'./save_weights/{saved_model}'))
    model = model.to(device)

    
    os.makedirs("./onnx_models", exist_ok=True)
    torch.onnx.export(
        model,
        torch.randn(1, 3, 224, 224, device=device),
        f"./onnx_models/{saved_model}_{tag}.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
    )


        
    model = model.to(device)

    test_acc, test_precision, test_recall, test_f1_score, results = evaluate_test_set(model, model_name, test_dl=test_loader, dataset_name=test_loader.dataset.root_dir, int2label=int2label, device=device, wandb_flag=wandb_flag)
    # print(f'Test Accuracy: {test_acc:.4f}')

    os.makedirs('results', exist_ok=True)
    summary = {
        'Time': saved_model_time,
        'Dataset': dataset,
        'Model': model_name,
        'Initial LR': learning_rate,
        'Model Path': saved_model,
        'Test Accuracy': f"{test_acc:.4f}",
        'Test Precision': f"{test_precision:.4f}",
        'Test Recall': f"{test_recall:.4f}",
        'Test F1_Score': f"{test_f1_score:.4f}"
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f'results/{saved_model_time}_{model_name}_results.csv', index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'results/{saved_model_time}_{model_name}_results.csv', mode='a', index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Skin Condition Classification')
    parser.add_argument('--model', '-m', type=str, default='efficientb0', help='Model name(e.g., 0211_1413_efficient_b1.pt)')
    parser.add_argument('--save_model_path', '-mp', type=str, default='0211_1413_efficient_b1.pt', help='Model name(e.g., 0211_1413_efficient_b1.pt)')
    parser.add_argument('--dataset', '-ds', type=str, default='./data/ISIC_2019', help='Dataset Path')
    parser.add_argument('--batch_size', '-bs', type=int, default=32, help='Batch size')
    parser.add_argument('--wandb_flag', '-wf', type=str, default='false', help='wandb_flag')
    parser.add_argument('--tag', '-t', type=str, default='0.0.11', help='Model Tagging Number')

    args = parser.parse_args()

    if args.wandb_flag.lower() == 'true':
        wandb_flag=True
    else:
        wandb_flag=False

    batch_size = args.batch_size
    tag = args.tag
    weight_decay = 0.01
    num_workers = 2
    num_epochs = 50
    lr = 0.001
    early_stopping_patience = 5
    model_save_root_path='./save_weights'
    int2label_path = os.path.join(args.dataset, "int2label.json")

    device, n_gpus, batch_size, num_workers = device_settings()

    params = {
        'dataset':args.dataset, 
        'model_name':args.model,
        'save_model_path':args.save_model_path,
        'label_path':int2label_path,
        'batch_size':batch_size,
        'lr':lr,
        'num_epochs':num_epochs,
        'num_workers':num_workers,
        'model_save_root_path':model_save_root_path,
        'tag':tag
    }

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")    

    test(args, params, device=device, wandb_flag=wandb_flag)