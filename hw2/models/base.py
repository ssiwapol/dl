# Starter code

import torch.nn as nn

class ClassificationNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
            )
        
        self.cls_layer = nn.Linear(512, num_classes)
    
    def forward(self, x, return_feats=False):
        feats = self.backbone(x)
        out = self.cls_layer(feats)
        if return_feats:
            return feats
        else:
            return out
