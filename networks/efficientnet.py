import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

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
    def __init__(self, num_classes=2):
        super(LinearClassifier, self).__init__()
        feat_dim = 1536
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)