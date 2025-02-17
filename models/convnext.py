import timm
import torch.nn as nn

def convnext(num_classes=1000):
    model = timm.create_model('convnext_base', pretrained=True)
    in_features = model.head.fc.in_features
    model.head.fc = nn.Linear(in_features, num_classes)
    return model