import torch
import torch.nn as nn
from models.enet import enet
from models.convnext import convnext
from models.vit import vit

class FusionModel(nn.Module):
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
        super(FusionModel, self).__init__()
        
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