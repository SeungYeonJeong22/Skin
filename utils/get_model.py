import torch
import numpy as np
import random
import os
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
from torch.cuda.amp import GradScaler, autocast

from warnings import filterwarnings
filterwarnings('ignore', category=UserWarning, module='torch')
filterwarnings('ignore', category=FutureWarning, module='torch')

from models.models import *
from models.customModels import *
from models.fpn_swin2 import *
from models.cnvx_msfpn import *
from models.niadermaSwin import *
from models.twoStage import *
from models.hybrid_model import *


def get_model(model_name, num_classes, minority_class_fn=None, minority_class_ids=None):
    # 모델 선택 & Grad-CAM에 쓰기 위한 마지막 레이어 선택
    # EfficientNet
    if model_name.__contains__('efficient'):
        model = enet(model_name=model_name, num_classes=num_classes)
        layer_name = None

    # ViT
    elif model_name.lower() == 'vit':
        model, feature_extractor = vit(num_classes=num_classes)
        layer_name = None

    # Swin Transformer tiny
    elif model_name.lower() == 'swin_transformer':
        model = swin_transformer(num_classes=num_classes)  
        layer_name = None

    # Swin Transformer tiny
    elif model_name.lower() == 'swin_transformer_tiny':
        model = swin_transformer_tiny(num_classes=num_classes)  
        layer_name =None

    # ConvNext
    elif model_name.lower() == 'convnext':
        model = convnext(num_classes=num_classes)
        layer_name = None

    # ConvNext Tiny
    elif model_name.lower() == 'convnext_tiny':
        model = convnext_tiny(num_classes=num_classes)  
        layer_name = None

    # MobileNet
    elif model_name.lower() == 'mobile':
        model = mobilenet(num_classes=num_classes)  
        layer_name = None

    # Sejin
    elif model_name.lower() == 'sejin':
        base_model_name = 'convnext_tiny'
        sub_model_name = 'swin_tiny_patch4_window7_224'
        model = Sejin(base_model_name=base_model_name, sub_model_name=sub_model_name, num_classes=num_classes)
        last_stage = None
        layer_name = None

    # Seungyeon        
    elif model_name.lower() == 'seungyeon':
        model = Seungyeon(num_classes=num_classes)
        layer_name = None

    # niaderma_33
    elif model_name.lower() == 'niaderma_33':
        model = NIADermaNet(num_classes=num_classes)
        layer_name = None

    # fpn strans efficient
    elif model_name.lower() == 'fpn_swin':
        model = FPN_Swin(num_classes=num_classes)
        layer_name = None

    # fpn strans efficient
    elif model_name.lower() == 'fpn_swin2':
        model = FPN_Swin2(num_classes=num_classes)
        layer_name = None

    elif model_name.lower() == 'cnvx_msfpn':
        model = ConvNeXt_FPN(num_classes=num_classes)
        layer_name = None

    elif model_name.lower() == 'niaderma_swin':
        model = NIADermaSwinHybrid(num_classes=num_classes)
        layer_name = None

    elif model_name.lower() == '2stage':
        model = TwoStageModel(num_classes=num_classes)
        layer_name = None

    elif model_name.lower() == 'hybrid':
        model = HybridFusionModel(num_classes=num_classes)
        layer_name = None
    else:
        raise ValueError("Unsupported model type (Model Name: efficient, vit, swin_transformer_tiny, convnext, convnext_tiny, mobile, sejin, seungyeon)")        

    return model, layer_name