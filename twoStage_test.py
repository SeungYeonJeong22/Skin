import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import classification_report

from models.twoStage import TwoStageModel
from dataset import SkinConditionDataset
from utils.utils import eval_metric

def twoStageTest(val_df, dataset_root, stage1_path, stage2_path, minority_class_ids, device='cpu'):
    model = TwoStageModel(
        num_classes=23,
        minority_class_fn=lambda preds: torch.isin(preds, torch.tensor(minority_class_ids, device=preds.device)),
        minority_class_ids=minority_class_ids
    ).to(device)

    model.stage1_model.load_state_dict(torch.load(stage1_path, map_location=device))
    model.stage2_model.load_state_dict(torch.load(stage2_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = SkinConditionDataset(val_df, dataset_root, mode='valid', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    int_to_label = val_dataset.int_to_label
    results = []
    acc = 0.0
    total = 0

    with torch.no_grad():
        for xb, yb, img_name in tqdm(val_loader, desc="Processing Test batches"):
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)

            prob, max_prob, pred, target, acc_b = eval_metric(logits, yb, len(int_to_label))
            acc += float(acc_b) * len(xb)
            total += len(xb)

            for i in range(len(xb)):
                image = "/".join(img_name[i].split("/")[-2:])
                true_label = int_to_label[yb[i].item()]
                predicted_label = int_to_label[pred[i].item()]
                probability = f"{prob[i][pred[i]].item():.4f}"

                results.append({
                    'image': image,
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'probability': probability
                })

    test_acc = acc / total * 100
    print(f"Test Accuracy: {test_acc:.2f}%")

    os.makedirs('results', exist_ok=True)
    base_name = os.path.basename(stage1_path).split('.')[0]
    summary_path = os.path.join('results', f'{base_name}_results.csv')

    summary_df = pd.DataFrame([{
        'Model Type': 'TwoStageModel',
        'Stage1 Path': stage1_path,
        'Stage2 Path': stage2_path,
        'Test Accuracy': f"{test_acc:.4f}"
    }])
    results_df = pd.DataFrame(results)

    summary_df.to_csv(summary_path, index=False)
    results_df.to_csv(summary_path, mode='a', index=False)

    print(f"Results saved to {summary_path}")