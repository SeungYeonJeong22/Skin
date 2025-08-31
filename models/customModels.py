import torchvision.ops as ops
import torch.nn as nn
import timm
import torch
import collections

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