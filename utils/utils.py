import torch
import numpy as np
import random
import torchvision.transforms as transforms
import os
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import cv2


def current_time():
    return datetime.now().strftime("%m%d_%H%M")

# check the directory to save weights.pt
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except Exception as e:
        print('Error')
createFolder('./models')

# fix seed for reproducibility
def seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True # non-deterministic
    torch.backends.cudnn.benchmark = False 
    np.random.seed(seed)
    random.seed(seed)
    

# calculate the accuracy per mini-batch
def accuracy(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects / target.size(0)


# get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']
    

# calculate the loss and accuracy per epoch
def loss_epoch(model, model_name, loss_func, dataset_dl, opt=None, device='cpu'):
    running_loss = 0.0
    running_acc = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in tqdm(dataset_dl, desc="Processing batches"):
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        if model_name.lower().__contains__('vit'):
            logits = output.logits if hasattr(output, 'logits') else output
        else:
            logits = output

        loss_b = loss_func(logits, yb)
        acc_b = accuracy(logits, yb)
        
        if opt is not None:
            opt.zero_grad()
            loss_b.backward()
            opt.step()

        running_loss += loss_b.item() * xb.size(0)
        running_acc += acc_b * xb.size(0)

    loss = running_loss / len_data
    acc = running_acc / len_data
    return loss, acc


def draw_plot(num_epochs, loss_hist, metric_hist):
    os.makedirs('imgs/plots', exist_ok=True)
    # Plot train-val loss
    plt.title('Train-Val Loss')
    plt.plot(range(1, num_epochs+1), loss_hist['train'], label='train')
    plt.plot(range(1, num_epochs+1), loss_hist['val'], label='val')
    plt.ylabel('Loss')
    plt.xlabel('Training Epochs')
    plt.legend()
    plt.savefig('imgs/plots/train_val_loss.png')

    # plot train-val accuracy
    plt.title('Train-Val Accuracy')
    plt.plot(range(1, num_epochs+1), metric_hist['train'], label='train')
    plt.plot(range(1, num_epochs+1), metric_hist['val'], label='val')
    plt.ylabel('Accuracy')
    plt.xlabel('Training Epochs')
    plt.legend()
    plt.savefig('imgs/plots/train_val_accuracy.png')

# Grad-CAM
def grad_cam(model, img_tensor, target_layer, class_idx=None):
    model.eval()
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # Forward pass
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    handle_forward.remove()
    handle_backward.remove()

    if class_idx is None:
        class_idx = torch.argmax(output, dim=1).item()

    # Backward pass
    model.zero_grad()
    class_score = output[0, class_idx]
    class_score.backward()

    gradients = gradients[0]
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    features = features[0][0]
    for i in range(features.shape[0]):
        features[i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(features, dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

def apply_heatmap(img, heatmap, alpha=0.6, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    superimposed_img = heatmap * alpha + img
    return superimposed_img

def save_grad_cam(img_path, model, img_tensor, target_layer, output_path, class_idx=None):
    heatmap = grad_cam(model, img_tensor, target_layer, class_idx)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    superimposed_img = apply_heatmap(img, heatmap)
    cv2.imwrite(output_path, superimposed_img)