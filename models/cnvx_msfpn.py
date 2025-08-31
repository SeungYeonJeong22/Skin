import torch
import torch.nn as nn
import torchvision.ops as ops
import timm
import torch.nn.functional as F

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