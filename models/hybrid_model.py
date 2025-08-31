import torch
import torch.nn as nn
from timm import create_model

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models.feature_extraction import create_feature_extractor


# ───────────────────────────────────────────────
# 1. FusionBlock: CNN & Transformer → Concat + Attention → Shared Fusion Feature
# ───────────────────────────────────────────────
class FusionBlock(nn.Module):
    def __init__(self, cnn_dim, tr_dim, fused_dim):
        super().__init__()
        self.proj_cnn = nn.Linear(cnn_dim, fused_dim)
        self.proj_tr = nn.Linear(tr_dim, fused_dim)
        self.attn = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(fused_dim)

    def forward(self, feat_cnn, feat_tr):
        B, c_cnn, h_cnn, w_cnn = feat_cnn.shape

        # Swin output (feat_tr)이 NHWC일 경우 자동으로 NCHW로 변환
        if feat_tr.shape[1] != c_cnn:
            feat_tr = feat_tr.permute(0, 3, 1, 2)  # [B, H, W, C] → [B, C, H, W]

        B, c_tr, h_tr, w_tr = feat_tr.shape

        # Flatten + Transpose → [B, C, H, W] → [B, N, C]
        feat_cnn_flat = feat_cnn.flatten(2).transpose(1, 2)
        feat_tr_flat = feat_tr.flatten(2).transpose(1, 2)

        proj_cnn = self.proj_cnn(feat_cnn_flat)
        proj_tr = self.proj_tr(feat_tr_flat)

        fused_cat = torch.cat([proj_cnn, proj_tr], dim=1)  # [B, N1+N2, fused_dim]
        attn_out, _ = self.attn(fused_cat, fused_cat, fused_cat)
        fused_final = self.norm(attn_out)

        return fused_final, h_cnn, w_cnn

# ───────────────────────────────────────────────
# 2. AttentionBack: Fusion Feature → Attention → Backbone Feature 보정
# ───────────────────────────────────────────────
class AttentionBack(nn.Module):
    def __init__(self, backbone_dim, fused_dim):
        super().__init__()
        self.query_proj = nn.Linear(backbone_dim, fused_dim)
        self.attn = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=4, batch_first=True)
        self.output_proj = nn.Linear(fused_dim, backbone_dim)
        self.norm = nn.LayerNorm(backbone_dim)

    def forward(self, backbone_feat, fused_feat, H, W):
        B = backbone_feat.shape[0]

        # Swin일 경우 NHWC로 나오는 경우 있음 → permute 필요
        if backbone_feat.shape[1] < backbone_feat.shape[-1]:  # ex: [B, 28, 28, 192]
            backbone_feat = backbone_feat.permute(0, 3, 1, 2)

        B, C, _, _ = backbone_feat.shape
        feat = backbone_feat.flatten(2).transpose(1, 2)  # [B, N, C]
        query = self.query_proj(feat)

        attn_out, _ = self.attn(query, fused_feat, fused_feat)
        attn_out = self.output_proj(attn_out)
        out = self.norm(attn_out + feat)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out


# ───────────────────────────────────────────────
# 3. 전체 하이브리드 모델: ConvNeXt + Swin + Fusion + 양방향 Attention
# ───────────────────────────────────────────────
class HybridFusionModel(nn.Module):
    def __init__(self, num_classes=5, fused_dim=256):
        super().__init__()

        # ConvNeXt-Tiny: feature extractor로 변환
        base_cnvx = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.convnext = create_feature_extractor(
            base_cnvx,
            return_nodes={
                "features.1": "stage1",
                "features.3": "stage2",
                "features.5": "stage3",
                "features.7": "stage4",
            }
        )

        # Swin-Tiny (timm 버전): features_only=True로 하면 자동으로 stage list 반환
        self.swin = create_model("swin_tiny_patch4_window7_224", pretrained=True, features_only=True)

        self.cnn_dims = [96, 192, 384, 768]
        self.tr_dims = [96, 192, 384, 768]

        self.fusion_blocks = nn.ModuleList([
            FusionBlock(c, t, fused_dim) for c, t in zip(self.cnn_dims, self.tr_dims)
        ])
        self.attn_back_cnn = nn.ModuleList([
            AttentionBack(c, fused_dim) for c in self.cnn_dims
        ])
        self.attn_back_tr = nn.ModuleList([
            AttentionBack(t, fused_dim) for t in self.tr_dims
        ])

        self.out_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # ConvNeXt: feature dict → 리스트로 변환
        feat_dict = self.convnext(x)
        cnn_feats = [feat_dict[f"stage{i+1}"] for i in range(4)]

        # Swin: 이미 list 형태로 반환됨
        tr_feats = self.swin(x)

        fused_out = None
        for i in range(4):
            if i == 0:
                # stage1은 fusion 없이 그대로 사용
                continue

            fused, H, W = self.fusion_blocks[i](cnn_feats[i], tr_feats[i])
            cnn_feats[i] = self.attn_back_cnn[i](cnn_feats[i], fused, H, W)
            tr_feats[i] = self.attn_back_tr[i](tr_feats[i], fused, H, W)
            fused_out = fused  # 마지막 fusion 결과 저장

        # stage1 제외 후에도 최소 1개 이상 fusion 통과한 경우 사용
        if fused_out is None:
            raise RuntimeError("Fusion output is None. Check your stage range.")

        pooled = fused_out.mean(dim=1)  # global average pooling
        return self.out_head(pooled)