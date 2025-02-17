import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.amp import GradScaler  # 혼합 정밀도 스케일러
import torch.nn.functional as F
from tqdm import tqdm
import gc


# 파이썬 가비지 컬렉션 실행: 더 이상 참조되지 않는 객체들을 정리합니다.
gc.collect()

# CUDA 메모리 캐시 비우기: 사용하지 않는 GPU 메모리를 해제합니다.
torch.cuda.empty_cache()


# 1. 모델 정의: ConvNeXt를 베이스라인으로 하여 최종 classification head를 23개 클래스로 변경합니다.
class Sejin(nn.Module):
    def __init__(self, base_model_name, sub_model_name, feature_num=128, num_classes=23, pretrained=True):
        super(Sejin, self).__init__()
        # Base model: classifier head 제거 (num_classes=0)
        self.base_model = timm.create_model(base_model_name, pretrained=pretrained, num_classes=0)
        in_features_base = self.base_model.num_features  # 기본 feature dimension (예: 768 혹은 그 이상)
        # 별도의 head: base model의 feature를 feature_num 차원으로 변환
        self.base_model_head = nn.Linear(in_features_base, feature_num)

        ## Sub model (Swin Transformer): classifier head 제거
        self.sub_model = timm.create_model(sub_model_name, pretrained=pretrained, num_classes=0)
        in_features_sub = self.sub_model.num_features
        ## 별도의 head: sub model의 feature를 feature_num 차원으로 변환
        self.sub_model_head = nn.Linear(in_features_sub, feature_num)

        # 최종 classifier: 두 모델의 feature를 결합 후 분류 (23개 클래스)
        self.classifier = nn.Sequential(
            nn.Linear(feature_num * 2, feature_num),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_num, num_classes)
        )

    def forward(self, x):
        # Base model (ConvNeXt): 보통 4D 텐서 반환, global pooling 적용
        base_features = self.base_model.forward_features(x)  # shape: (B, C, H, W) 예상
        if base_features.dim() == 4:
            base_features = F.adaptive_avg_pool2d(base_features, 1)
            base_features = torch.flatten(base_features, 1)  # (B, in_features_base)
        # 만약 이미 2D라면 그대로 사용
        feat1 = self.base_model_head(base_features)  # (B, feature_num)

        # Sub model (Swin Transformer): 보통 3D 텐서 반환, shape: (B, num_tokens, embed_dim)
        sub_features = self.sub_model.forward_features(x)
        if sub_features.dim() == 4:
            # 토큰 차원(dim=1)에 대해 평균을 내어 (B, embed_dim)로 변환
            sub_features = sub_features.mean(dim=1)
            sub_features = sub_features.mean(dim=1)
        feat2 = self.sub_model_head(sub_features)  # (B, feature_num)

        # 두 모델의 feature 결합 후 최종 분류
        combined_features = torch.cat([feat1, feat2], dim=1)
        output = self.classifier(combined_features)
        return output