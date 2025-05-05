"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class SupConResNet(nn.Module):
    """ResNet50 + projection head for SupCon"""
    def __init__(self, pretrained=True, head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # Remove final FC layer
        dim_in = base.fc.in_features

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(f"Unsupported head type: {head}")

    def forward(self, x):
        feat = self.encoder(x)         # (B, 2048, 1, 1)
        feat = torch.flatten(feat, 1)     # (B, 2048)
        feat = F.normalize(self.head(feat), dim=1)
        return feat
    
class SupConEfficientNet(nn.Module):
    def __init__(self, pretrained=True, head='mlp', feat_dim=128):
        super(SupConEfficientNet, self).__init__()
        base = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT if pretrained else None)

        # EfficientNet has 'features' and 'avgpool' instead of plain children
        self.encoder = nn.Sequential(
            base.features,
            base.avgpool  # AdaptiveAvgPool2d
        )
        dim_in = base.classifier[1].in_features  # EfficientNet-B3's classifier is Sequential([Dropout, Linear])

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(f"Unsupported head type: {head}")

    def forward(self, x):
        feat = self.encoder(x)        # (B, dim_in, 1, 1)
        feat = torch.flatten(feat, 1) # (B, dim_in)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

class LinearClassifier(nn.Module):
    def __init__(self, model, num_classes=2):
        super(LinearClassifier, self).__init__()
        if model == "resnet50":
            feat_dim = 2048
        elif model == "efficientnetb3":
            feat_dim = 1536
        else:
            raise NotImplementedError(f"Unsupported model: {model}")
        
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
    
