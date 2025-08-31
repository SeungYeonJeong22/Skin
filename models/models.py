import subprocess
import sys

try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    print("📦 efficientnet_pytorch 패키지가 없어서 설치합니다...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "efficientnet_pytorch"])
    from efficientnet_pytorch import EfficientNet

try:
    import timm
except ImportError:
    print("📦 timm 패키지가 없어서 설치합니다...")    
    subprocess.check_call([sys.executable, "-m", "pip", "install", "timm"])
    import timm



import torch.nn as nn
# from efficientnet_pytorch import EfficientNet
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch
import torch.nn.functional as F
import gc
from torchvision import models
import torchvision.ops as ops
import collections


# 파이썬 가비지 컬렉션 실행: 더 이상 참조되지 않는 객체들을 정리합니다.
gc.collect()

# CUDA 메모리 캐시 비우기: 사용하지 않는 GPU 메모리를 해제합니다.
torch.cuda.empty_cache()


def enet(model_name='efficientb0', num_classes=23):
    if model_name == 'efficientb0':
        return EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
    elif model_name == 'efficientb1':
        return EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)
    elif model_name == 'efficientb2':
        return EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes)
    elif model_name == 'efficientb3':
        return EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
    elif model_name == 'efficientb4':
        return EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
    elif model_name == 'efficientb5':
        return EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes)
    elif model_name == 'efficientb6':
        return EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_classes)
    elif model_name == 'efficientb7':
        return EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes)
    else:
        raise ValueError('Invalid model name')


def convnext(num_classes=23):
    model = timm.create_model('convnext_base', pretrained=True)
    in_features = model.head.fc.in_features
    model.head.fc = nn.Linear(in_features, num_classes)
    return model


def convnext_tiny(num_classes=23):
    model = timm.create_model('convnext_tiny', pretrained=True)
    in_features = model.head.fc.in_features
    model.head.fc = nn.Linear(in_features, num_classes)
    return model


def vit(num_classes=23):
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=num_classes, ignore_mismatched_sizes=True)
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    return model, feature_extractor


def swin_transformer(num_classes=23):
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
    in_features = model.num_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def swin_transformer_tiny(num_classes=23):
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
    in_features = model.num_features
    model.head = nn.Linear(in_features, num_classes)  # model.head.fc 대신 model.head로 교체
    return model


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
    

class Seungyeon(nn.Module):
    """
    CNN Branch (EfficientNet 또는 ConvNeXt)와 ViT Branch의 중간 특징을 융합하는 모델.
    두 branch의 특징을 각각 adapter를 거쳐 512 차원으로 맞추고,
    공간적으로 결합한 뒤 추가 convolution을 통해 최종 분류를 수행합니다.
    """
    def __init__(self, cnn_type='efficientb0', num_classes=23):
        """
        Args:
            cnn_type (str): 'efficientb0' 또는 'convnext' 등, 사용할 CNN 모델 지정.
            num_classes (int): 분류할 클래스 수.
        """
        super(Seungyeon, self).__init__()
        
        # CNN Branch 초기화 (EfficientNet 또는 ConvNeXt)
        if 'efficient' in cnn_type.lower():
            self.cnn_branch = enet(model_name=cnn_type, num_classes=num_classes)
            # EfficientNet의 경우, 마지막 convolution block의 출력 채널 수 (예시: EfficientNet B0는 약 1280)
            cnn_out_channels = 1280  
        elif 'convnext' in cnn_type.lower():
            self.cnn_branch = convnext(num_classes=num_classes)
            # ConvNeXt의 경우, 마지막 stage의 출력 채널 수 (예시: 1024)
            cnn_out_channels = 1024
        else:
            raise ValueError("지원하지 않는 cnn_type입니다. (efficient 또는 convnext 사용)")
        
        # ViT Branch 초기화  
        # vit() 함수는 (모델, feature_extractor) 튜플을 반환한다고 가정합니다.
        self.vit_branch, self.feature_extractor = vit(num_classes=num_classes)
        # 일반적인 ViT 베이스 모델의 임베딩 차원 (예시: 768)
        vit_feature_dim = 768
        
        # 각 branch의 중간 특징을 512 차원(채널)으로 맞추기 위한 adapter
        self.cnn_adapter = nn.Conv2d(in_channels=cnn_out_channels, out_channels=512, kernel_size=1)
        self.vit_adapter = nn.Linear(vit_feature_dim, 512)
        
        # Fusion Module: 두 branch의 adapter된 특징을 채널 방향으로 결합 후 추가 convolution 적용
        # 최종 결합된 채널 수는 512*2=1024
        self.fusion_conv = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        
        # 최종 분류 헤드: Global Average Pooling 후 Fully Connected layer
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # CNN Branch 처리
        cnn_features = self.cnn_branch.extract_features(x) # [B, cnn_out_channels, H, W]
        cnn_features = self.cnn_adapter(cnn_features)  # [B, 512, H, W]
        
        # ViT Branch 처리
        vit_out = self.vit_branch(x, output_hidden_states=True)
        vit_tokens = vit_out.hidden_states[-1] # shape: [B, N, vit_feature_dim]
        vit_global = vit_tokens.mean(dim=1) # [B, vit_feature_dim]
        vit_features = self.vit_adapter(vit_global) # [B, 512]
        B, _, H, W = cnn_features.size() 
        vit_features = vit_features.unsqueeze(-1).unsqueeze(-1).expand(B, 512, H, W)
        
        # Fusion
        fused_features = torch.cat([cnn_features, vit_features], dim=1) # [B, 1024, H, W]
        fused_features = self.fusion_conv(fused_features) # [B, 512, H, W]
        
        # Classification
        out = self.classifier(fused_features) # [B, num_classes]
        return out    
    

# Moblie Net
def mobilenet(num_classes=23):
    # 사전 학습된 MobileNet V2 모델 불러오기
    model = models.mobilenet_v2(pretrained=True)
    
    # MobileNet V2의 마지막 채널 크기 (일반적으로 1280)
    last_channel = model.last_channel  # 또는 1280로 직접 지정 가능
    
    # classifier를 수정하여 num_classes에 맞게 변경
    # 원래 classifier: Sequential(Dropout(0.2), Linear(1280, 1000))
    model.classifier[1] = nn.Linear(last_channel, num_classes)
    
    return model    


# 세진님 모델 원본
class SkinDiseaseClassifier(nn.Module):
    def __init__(self, base_model_name, sub_model_name, feature_num=128, num_classes=23, pretrained=True):
        super(SkinDiseaseClassifier, self).__init__()
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
            # nn.Dropout(0.5),
            nn.Linear(feature_num, num_classes)
        )

    def forward(self, x):
        # Base model (ConvNeXt): 보통 4D 텐서 반환, global pooling 적용
        base_features = self.base_model.forward_features(x)  # shape: (B, C, H, W) 예상
        if base_features.dim() == 4:
            base_features = F.adaptive_avg_pool2d(base_features, 1)
            base_features = torch.flatten(base_features, 1)  # (B, in_features_base)
            #base_features = base_features.mean(dim=1)  # (B, in_features_base)
            #print(base_features.shape)
            #base_features = base_features.mean(dim=2)  # (B, in_features_base)
            #print(base_features.shape)
            #base_features = base_features.mean(dim=2)  # (B, in_features_base)
            #print(base_features.shape)

        # 만약 이미 2D라면 그대로 사용
        feat1 = self.base_model_head(base_features)  # (B, feature_num)

        # Sub model (Swin Transformer): 보통 3D 텐서 반환, shape: (B, num_tokens, embed_dim)
        sub_features = self.sub_model.forward_features(x)
        if sub_features.dim() == 4:
            #print(sub_features.shape)
            #sub_features = F.adaptive_avg_pool2d(sub_features.transpose(1, 3), 1)
            #print(sub_features.shape)
            #sub_features = torch.flatten(sub_features, 1)
            #print(sub_features.shape)
            # 토큰 차원(dim=1)에 대해 평균을 내어 (B, embed_dim)로 변환
            sub_features = sub_features.mean(dim=1)
            #print(sub_features.shape)
            sub_features = sub_features.mean(dim=1)
            #print(sub_features.shape)
        feat2 = self.sub_model_head(sub_features)  # (B, feature_num)

        # 두 모델의 feature 결합 후 최종 분류
        combined_features = torch.cat([feat1, feat2], dim=1)
        output = self.classifier(combined_features)
        return output

# Residual Block 정의
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = None
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Skip connection (Add operation)
        return F.relu(out)

# 전체 모델 정의
class NIADermaNet(nn.Module):
    def __init__(self, num_classes=33):
        super(NIADermaNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet과 비슷한 블록 구조
        self.layer1 = self._make_layer(64, 64, num_blocks=2)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, downsample=True)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, downsample=True)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, downsample=True)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # GlobalAveragePool
        self.flatten = nn.Flatten()  # Flatten
        self.fc = nn.Linear(512, num_classes)  # Gemm (FC Layer)

    def _make_layer(self, in_channels, out_channels, num_blocks, downsample=False):
        layers = [ResidualBlock(in_channels, out_channels, downsample)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


class FPN_Swin(nn.Module):
    def __init__(self, num_classes=23):
        super(FPN_Swin, self).__init__()

        # 1️ Swin Transformer as Feature Extractor (Backbone)
        self.swin = timm.create_model("swin_base_patch4_window7_224", pretrained=True, features_only=True)

        # Swin Transformer의 Feature Map 채널 크기
        self.swin_channels = [128, 256, 512, 1024]  # Swin Transformer의 각 Stage 출력 채널

        # 2️ Feature Pyramid Network (FPN)
        self.fpn = ops.FeaturePyramidNetwork(
            in_channels_list=self.swin_channels,  # Swin Transformer에서 추출된 Feature Map을 입력
            out_channels=256
        )

        # 3️ Classification Layer
        self.final_fc = nn.Linear(256, num_classes)  # 최종 Feature Map을 분류기와 연결

    def forward(self, x):
        #  1. Swin Transformer Feature Extraction
        swin_features = self.swin(x)  # Swin Transformer의 Feature Map 가져오기

        #  2. FPN 입력 생성
        fpn_inputs = collections.OrderedDict()
        for i, feature in enumerate(swin_features):
            fpn_inputs[str(i)] = feature.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)

        #  3. FPN 적용
        fpn_output = self.fpn(fpn_inputs)  # FPN을 통해 Multi-Scale Feature Map 생성

        #  4. 최종 Feature Map 사용 (가장 작은 해상도 Feature)
        last_feature = fpn_output["3"]  # Swin Transformer의 마지막 Feature Map 사용

        #  5. Global Average Pooling (GAP)
        gap_feature = torch.mean(last_feature, dim=[2, 3])  # GAP 적용하여 1D 벡터 생성

        #  6. Classification
        output = self.final_fc(gap_feature)  # FC Layer를 통과하여 최종 클래스 예측

        return output
    

class MultiScaleFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256, dropout=0.4):
        super(MultiScaleFPN, self).__init__()

        # FPN 0 (Local Features)
        self.fpn0 = ops.FeaturePyramidNetwork(
            in_channels_list=in_channels_list, out_channels=out_channels
        )
        self.fpn0_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout)
                ) for _ in range(len(in_channels_list))
        ])

        # FPN 1 (Mid Level Features)
        self.fpn1 = ops.FeaturePyramidNetwork(
            in_channels_list=in_channels_list, out_channels=out_channels
        )
        self.fpn1_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=2, dilation=2),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout)
            )
            for _ in range(len(in_channels_list))
        ])

        # FPN 2 (Global Features, stride=2 + stride=4 혼합)
        self.fpn2 = ops.FeaturePyramidNetwork(
            in_channels_list=in_channels_list, out_channels=out_channels
        )
        self.fpn2_convs1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=4, dilation=4),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout)
            )
            for _ in range(len(in_channels_list))
        ])
        self.fpn2_convs2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=4, padding=6, dilation=6),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout)
            )
            for _ in range(len(in_channels_list))
        ])

    def forward(self, features):
        # FPN 0 (Local Features)
        fpn0_features = self.fpn0(features)
        fpn0_outs = {k: self.fpn0_convs[int(k)](v) for k, v in fpn0_features.items()}

        # FPN 1 (Mid-Level Features)
        fpn1_features = self.fpn1(features)
        fpn1_outs = {k: self.fpn1_convs[int(k)](v) for k, v in fpn1_features.items()}

        # FPN 2 (Global Features with stride=2 and stride=4)
        fpn2_features = self.fpn2(features)
        fpn2_outs1 = {k: self.fpn2_convs1[int(k)](v) for k, v in fpn2_features.items()}
        fpn2_outs2 = {k: self.fpn2_convs2[int(k)](v) for k, v in fpn2_features.items()}

        # stride=2와 stride=4를 합치기 위해 Interpolation 수행
        fpn2_final_outs = {
            k: (fpn2_outs1[k] + F.interpolate(fpn2_outs2[k], size=fpn2_outs1[k].shape[2:], mode="bilinear", align_corners=False)) / 2
            for k in fpn2_outs1.keys()
        }

        return fpn0_outs, fpn1_outs, fpn2_final_outs


class ConvNeXt_FPN(nn.Module):
    def __init__(self, num_classes=23, out_channels=256, dropout=0.4):
        super(ConvNeXt_FPN, self).__init__()

        self.backbone = timm.create_model("convnext_base", pretrained=True, features_only=True)
        self.in_channels_list = self.backbone.feature_info.channels()  # [128, 256, 512, 1024]

        # Multi-Scale FPN 적용
        self.multi_scale_fpn = MultiScaleFPN(in_channels_list=self.in_channels_list, out_channels=out_channels)

        self.final = nn.Sequential(
            nn.Linear(out_channels * len(self.in_channels_list) * 3, 2048),
            nn.ReLU(inplace=True),
            nn.LayerNorm(2048),
            nn.Dropout(dropout),
            # nn.Linear(2048, 1024),
            # nn.ReLU(inplace=True),
            # nn.LayerNorm(1024),
            # nn.Dropout(dropout),
            # nn.Linear(1024, 256),
            # nn.ReLU(inplace=True),
            # nn.LayerNorm(256),
            # nn.Dropout(dropout),
            # nn.Linear(256, num_classes)
            nn.Linear(2048, num_classes)
        )
        
        # Classification Layer
        self.fc1 = nn.Linear(out_channels * len(self.in_channels_list) * 3, 1024)  # out_channel * ([128, 256, 512, 1024])개수 * 3(fpn 개수)
        self.norm = nn.LayerNorm(len(self.in_channels_list))

        self.final_fc = nn.Linear(len(self.in_channels_list), num_classes)

    def forward(self, x):
        # ConvNeXt Feature 추출
        features = self.backbone(x)

        # Feature Map을 딕셔너리 형태로 변환
        feature_dict = {str(i): features[i] for i in range(len(features))}

        # Multi-Scale FPN 적용
        fpn0_outs, fpn1_outs, fpn2_outs = self.multi_scale_fpn(feature_dict)

        # 3개의 FPN Feature를 Global Average Pooling (GAP) 후 Concatenation
        fpn0_gap = torch.cat([torch.mean(fpn0_outs[k], dim=[2, 3]) for k in fpn0_outs.keys()], dim=1)
        fpn1_gap = torch.cat([torch.mean(fpn1_outs[k], dim=[2, 3]) for k in fpn1_outs.keys()], dim=1)
        fpn2_gap = torch.cat([torch.mean(fpn2_outs[k], dim=[2, 3]) for k in fpn2_outs.keys()], dim=1)

        # Feature Concatenation
        merged_feature = torch.cat([fpn0_gap, fpn1_gap, fpn2_gap], dim=1)

        # Classification
        output = self.final(merged_feature)

        return output
    

class FPN_Swin2(nn.Module):
    def __init__(self, num_classes=23):
        super(FPN_Swin2, self).__init__()
        self.swin = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, features_only=True)

        # Swin Transformer에서 선택할 Feature Map 채널 크기 (0, 2번 인덱스 -> batch가 64일 경우 터짐)
        # self.selected_layers = [0, 2]  # 사용할 feature index
        # self.swin_channels = [128, 512]  # Swin Transformer의 해당 Stage 출력 채널
        self.selected_layers = [1]
        self.swin_channels = [192]

        self.fpn = ops.FeaturePyramidNetwork(
            in_channels_list=self.swin_channels,  
            out_channels=256
        )

        self.final_fc = nn.Linear(256, num_classes) 

    def forward(self, x):
        swin_features = self.swin(x)  # Swin Transformer의 Feature Map 가져오기

        selected_features = {str(i): swin_features[idx].permute(0, 3, 1, 2) 
                             for i, idx in enumerate(self.selected_layers)}

        fpn_output = self.fpn(selected_features)  # FPN을 통해 Multi-Scale Feature Map 생성
        last_feature = fpn_output[str(len(self.selected_layers) - 1)]  # 선택된 마지막 feature 사용
        gap_feature = torch.mean(last_feature, dim=[2, 3])  # GAP 적용하여 1D 벡터 생성

        output = self.final_fc(gap_feature)  # FC Layer를 통과하여 최종 클래스 예측

        return output