import torchvision.ops as ops
import torch.nn as nn
import torch.nn.functional as F
import timm
import torch
import numpy as np


class ARLC_Blocks(nn.Module):
    def __init__(self, channels):
        super(ARLC_Blocks, self).__init__()

        self.dwsc = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.layer_norm = nn.LayerNorm(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gelu = nn.GELU()

        self.alpha = nn.Parameter(torch.ones(1))
        self.softmax = torch.softmax()


    def forward(self, x):
        # x = batch, c, w, h
        l1 = x

        l2 = self.dwsc(x.permute(0, 2, 3, 1))
        l2 = self.layer_norm(l2)
        l2 = self.conv1(l2.permute(0, 3, 1, 2))
        l2 = self.gelu(l2)
        l2 = self.conv2(l2)

        l3 = self.layer_norm(l2.permute(0, 2, 3, 1))
        l3 = F.softmax(l3, dim=-1)
        l3 = l2.permute(0, 3, 1, 2)
        l3 = self.alpha * x * l3
        
        out = l1 + l2 + l3

        return out
        






class EAFM_Net(nn.Module):
    def __init__(self):
        super(EAFM_Net, self).__init__()

    

