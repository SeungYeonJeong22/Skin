import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.utils import accuracy, seed, current_time
from dataset import SkinConditionDataset
from models.efficientnet.enet import enet
from models.vit.vit import vit
from torchvision import transforms

import argparse

import os
import pandas as pd

loss_func = nn.CrossEntropyLoss()

# function to evaluate the model on the test set
def evaluate_test_set(model, model_name, test_dl, int_to_label, device='cpu'):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    len_data = len(test_dl.dataset)
    results = []

    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            output = model(xb)

            if model_name.lower().__contains__('vit'):
                logits = output.logits if hasattr(output, 'logits') else output
            else:
                logits = output            

            loss_b = loss_func(logits, yb)
            acc_b = accuracy(logits, yb)

            test_loss += loss_b.item() * xb.size(0)
            test_acc += acc_b * xb.size(0)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            for i in range(len(xb)):
                results.append({
                    'image': test_dl.dataset.df.iloc[i]['image_path'],
                    'true_label': int_to_label[yb[i].item()],
                    'predicted_label': int_to_label[preds[i].item()],
                    'probability': f"{probs[i][preds[i]].item():.4f}"
                })

    test_loss /= len_data
    test_acc /= len_data

    test_acc *= 100

    print('Test accuracy: {:.2f}'.format(test_acc))
    return test_loss, test_acc, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Skin Condition Classification')
    parser.add_argument('--saved_model', '-mp', type=str, default='0211_1413_efficient_b1.pt', help='Model path (e.g., 0211_1413_efficient_b1.pt)')
    parser.add_argument('--dataset', '-rd', type=str, default='./data/AICamp-2023-Skin-Conditions_Dataset', help='Root directory')
    parser.add_argument('--batch', '-b', type=int, default=16, help='Batch size')
    args = parser.parse_args()

    seed(42)
    root_path = args.dataset

    test_csv_path = os.path.join(root_path, 'test.csv')
    df = pd.read_csv(test_csv_path)
    labels = df['label'].unique()
    num_classes = len(labels)        

   

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    test_dataset = SkinConditionDataset(root_dir=root_path, split='test.csv', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    test_size = int(0.8 * len(test_dataset))
    print("Test Size : ", test_size)

    saved_model = args.saved_model
    saved_model_time = "_".join(saved_model.split('.')[0].split("_")[:2])
    saved_model_name = "".join(saved_model.split('.')[0].split("_")[2:])
    
    if saved_model_name.lower().__contains__('vit'):
        model, feature_extractor = vit(num_classes=num_classes)
    elif saved_model_name.lower().__contains__('efficient'):
        model = enet(model_name=saved_model_name, num_classes=num_classes)

    model.load_state_dict(torch.load(f'./save_weights/{saved_model}'))
    model = model.to(device)

    test_loss, test_acc, results = evaluate_test_set(model, saved_model_name, test_loader, test_dataset.int_to_label, device=device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

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
    results_df.to_csv(f'results/{saved_model_time}_{saved_model_name}_results.csv', mode='a', header=False, index=False)