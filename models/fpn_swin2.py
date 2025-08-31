import torchvision.ops as ops
import torch.nn as nn
import timm
import torch
import collections

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