import torch
import numpy as np
import random
import os
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score

from warnings import filterwarnings
filterwarnings('ignore', category=UserWarning, module='torch')
filterwarnings('ignore', category=FutureWarning, module='torch')

try:
    import torchmetrics
except ImportError:
    import subprocess
    import sys
    print("📦 timm 패키지가 없어서 설치합니다...")    
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchmetrics"])
    import torchmetrics



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
    

# get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


# OOD 감지 함수 (Softmax 확률 기반)
def is_unknown_sample(logits, top_k=3, threshold=0.6):
    probs = torch.softmax(logits, dim=1)
    
    topk_probs, _ = torch.topk(probs, k=top_k, dim=1)
    topk_sum = topk_probs.sum(dim=1)
    
    mask = topk_sum < threshold
    return mask # True: Unknown, False: Known


# # 가중치를 반영한 정확도 계산 함수 (ISIC 2019 데이터셋 적용)
# def weighted_accuracy(pred, target, img_names, test_gt):
#     # Unknown(8) 클래스 제거하여 평가
#     valid_mask = target != 8  # Unknown(8) 제외
#     pred = pred[valid_mask]
#     target = target[valid_mask]

#     if len(target) == 0:  # 모든 샘플이 Unknown이면 0 반환
#         return 0.0, 0

#     # `torchmetrics.Accuracy`를 올바른 디바이스로 이동
#     metric_acc = torchmetrics.Accuracy(task="multiclass", num_classes=8, average="weighted", top_k=1).to(output.device)

#     acc = metric_acc(pred, target).item()  # GPU에서 CPU로 가져와서 float 값 반환
#     batch_weight_sum = len(target)  # Unknown을 제외한 샘플 수를 가중치로 사용

#     # return acc * batch_weight_sum, batch_weight_sum  # (정확도 * 가중치, 총 가중치)
#     return acc


# # calculate the accuracy per mini-batch
# def accuracy(output, target):
#     pred = output.argmax(1, keepdim=True)
#     corrects = pred.eq(target.view_as(pred)).sum().item()
#     return corrects / target.size(0)


def eval_metric(output, target, num_classes, isic_flag=False):
    prob = torch.softmax(output, dim=1)
    max_prob, pred = torch.max(prob, dim=1)
    # pred = torch.softmax(output, dim=1).argmax(1, keepdim=True).squeeze(1)
    target = target.view_as(pred)
    
    # ISIC Dataset
    if isic_flag:
        # unknown post process
        # unknown_mask = is_unknown_sample(output, top_k=1, threshold=0.3)
        # pred[unknown_mask] = 8
        pass

        # print(f"Unknown mask 적용된 샘플 수: {unknown_mask.sum().item()}")
        # print("Unique predicted labels:", torch.unique(pred))
        # print(f"Predicted class distribution: {torch.bincount(pred)}")
        
        # metric_acc = torchmetrics.Accuracy(task="multiclass", num_classes=9, average="weighted", top_k=1).to(output.device)
        # metric_precision = torchmetrics.Precision(task = "multiclass", num_classes=9, average="weighted", top_k=1).to(output.device)
        # metric_recall = torchmetrics.Recall(task = "multiclass", num_classes=9, average="weighted", top_k=1).to(output.device)
        # metric_f1 = torchmetrics.F1Score(task = "multiclass", num_classes=9, average="weighted", top_k=1).to(output.device)

    # else:
    metric_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average="weighted").to(output.device)
    # metric_precision = torchmetrics.Precision(task = "multiclass", num_classes=num_classes, average="weighted").to(output.device)
    metric_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="weighted").to("cpu")
    metric_recall = torchmetrics.Recall(task = "multiclass", num_classes=num_classes, average="weighted").to(output.device)
    metric_f1 = torchmetrics.F1Score(task = "multiclass", num_classes=num_classes, average="weighted").to(output.device) 

    acc = metric_acc(pred, target).item()
    if acc == np.nan:
        pass
    # recall = metric_recall(pred, target).item()
    # f1_score = metric_f1(pred, target).item()

    # metric_precision.update(pred.cpu(), target.cpu())  
    # precision = metric_precision.compute().item()

    sklearn_precision = precision_score(target.cpu().numpy(), pred.cpu().numpy(), average="weighted", zero_division=0)

    # return prob, max_prob, pred, target, acc, precision, recall, f1_score
    return prob, max_prob, pred, target, acc

# calculate the loss and accuracy per epoch
def loss_epoch(model, model_name, epoch, loss_func, dataset_dl, int2label=None, opt=None, device='cpu', mode='train', training_record_flag='false'):
    import torchmetrics
    from tqdm import tqdm
    from torch.cuda.amp import autocast

    running_loss = 0.0
    running_acc = 0.0
    len_data = len(dataset_dl.dataset)

    metric_acc = torchmetrics.Accuracy(task="multiclass", num_classes=dataset_dl.dataset.num_classes, average="weighted", top_k=1).to(device)
    is_train = opt is not None

    record_list = []

    for idx, (xb, yb, img_name) in enumerate(tqdm(dataset_dl, desc=f"Processing {mode} batches")):
        xb = xb.to(device)
        yb = yb.to(device)

        with autocast():
            output = model(xb)

            if model_name.lower() == 'vit':
                logits = output.logits if hasattr(output, 'logits') else output
            elif model_name.lower().__contains__("swin_transformer"):
                logits = output
                if output.ndim == 4:
                    output = output.permute(0, 3, 1, 2)
                    logits = torch.nn.functional.adaptive_avg_pool2d(output, 1).flatten(1)
            else:
                logits = output

            loss_b = loss_func(logits, yb)
            acc_b = metric_acc(logits, yb).item()

        if is_train:
            opt.zero_grad()
            loss_b.backward()
            opt.step()

        running_loss += loss_b.item() * xb.size(0)
        running_acc += acc_b * xb.size(0)

        if training_record_flag and mode == 'train':
            probs = torch.softmax(logits.float(), dim=1).detach().cpu()
            preds = torch.argmax(probs, dim=1)

            for i in range(len(yb)):
                record_list.append({
                    'image': os.path.basename(img_name[i]),
                    'true_label': int2label[str(yb[i].item())],
                    'predicted_label': int2label[str(preds[i].item())],
                    'probability': f"{probs[i, preds[i]].item()}.:4f"
                })

    loss = running_loss / len_data
    acc = running_acc / len_data

    return loss, acc, record_list


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
    img_tensor = img_tensor.unsqueeze(0)  # 배치 차원 추가

    # Feature 및 Gradient 저장을 위한 리스트
    features = []
    gradients = []

    # Forward hook (특징 맵 저장)
    def forward_hook(module, input, output):
        features.append(output)

    # Backward hook (Gradient 저장)
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Hook 등록
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    # Forward propagation
    output = model(img_tensor)
    
    # Hook 제거 (메모리 누수 방지)
    handle_forward.remove()
    handle_backward.remove()

    if class_idx is None:
        class_idx = torch.argmax(output, dim=1).item()

    # Backward propagation
    model.zero_grad()
    class_score = output[0, class_idx]
    class_score.backward()

    # Gradient가 저장되지 않은 경우 예외 처리
    if len(gradients) == 0 or len(features) == 0:
        raise RuntimeError("Gradients or Features are not captured. Check target_layer.")

    # Gradients 및 Feature Map 가져오기
    gradients = gradients[0]  # [batch, channels, height, width]
    features = features[0][0]  # [channels, height, width]

    # Gradient 평균 풀링
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)
    pooled_gradients = pooled_gradients.view(features.shape[0], 1, 1)

    # Feature Map 가중치 적용
    features *= pooled_gradients

    # Heatmap 생성
    heatmap = torch.mean(features, dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)  # ReLU 적용 (음수 값 제거)
    heatmap /= (np.max(heatmap) + 1e-8)  # 정규화 (0~1) + 안정성 확보

    return heatmap

# def apply_heatmap(img, heatmap, alpha=0.6, colormap=cv2.COLORMAP_JET):
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, colormap)
#     superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
#     return superimposed_img

# Heatmap을 이미지에 적용
def apply_heatmap(img, heatmap, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    Grad-CAM heatmap을 원본 이미지에 적용하는 함수.
    
    :param img: 원본 이미지 (H, W, 3) (uint8)
    :param heatmap: Grad-CAM에서 생성된 heatmap (H, W) (float)
    :param alpha: heatmap의 가중치 (0~1)
    :param colormap: 적용할 OpenCV 컬러맵 (기본값: JET)
    :return: heatmap이 적용된 이미지
    """
    # heatmap을 원본 이미지 크기로 변환
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # NaN 값 방지
    heatmap = np.nan_to_num(heatmap)

    # Heatmap을 uint8로 변환 (0~255)
    heatmap = np.uint8(255 * heatmap)

    # 컬러맵 적용
    heatmap = cv2.applyColorMap(heatmap, colormap)

    # 원본 이미지와 혼합 (투명도 alpha 조절)
    if img.dtype != np.uint8:
        img = np.uint8(255 * (img - img.min()) / (img.max() - img.min()))  # 0~255로 변환

    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    return superimposed_img

def save_grad_cam(img_path, model, img_tensor, target_layer, output_path, class_idx=None):
    heatmap = grad_cam(model, img_tensor, target_layer, class_idx)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    superimposed_img = apply_heatmap(img, heatmap)
    cv2.imwrite(output_path, superimposed_img)


# Hook 함수 정의
def get_activation(activation, name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        """
        Focal Loss 구현
        :param alpha: 특정 클래스에 가중치를 주는 하이퍼파라미터 (클래스 불균형 시 유용)
        :param gamma: 어려운 샘플에 대한 가중치를 조정하는 하이퍼파라미터
        :param reduction: 'mean', 'sum', 'none' 중 선택 가능
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        :param inputs: 모델 출력값 (logits 형태) [batch_size, num_classes]
        :param targets: 정답 라벨 (정수형) [batch_size]
        :return: Focal Loss 값
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # 기본 CE Loss 계산
        pt = torch.exp(-ce_loss)  # 예측 확률값 변환

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # Focal Loss 적용

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        

def device_settings(args=None):
    if args:
        batch_size = args.batch_size
        num_workers = args.num_workers
    else:
        batch_size = 32
        num_workers = 4

    # Device setting + DataParallel
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device = torch.device("cuda")
        print(f"Using CUDA ({n_gpus} GPUs): {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        n_gpus = 0
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        n_gpus = 0
        device = torch.device("cpu")
        print("Using CPU")

    if n_gpus > 1:
        batch_size *= n_gpus
        num_workers *= n_gpus

    return device, n_gpus, batch_size, num_workers