import torch
import torch.nn as nn
import torchvision.models as models

class FishSegmentationClassificationModel(nn.Module):
    def __init__(self, num_classes=23):
        super(FishSegmentationClassificationModel, self).__init__()
        
        # Load pretrained ResNet18
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Remove the final fully connected layer
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        
        # Segmentation head with upsampling
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 8×8 → 16×16
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 16×16 → 32×32
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),  # 32×32 → 256×256
            
            nn.Conv2d(64, 1, kernel_size=1)  # Final output (1 channel for mask)
        )
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.classification_head = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.encoder(x)  # Shared Encoder
        
        # Segmentation output
        seg_output = self.segmentation_head(features)
        
        # Classification output
        cls_features = self.avgpool(features)
        cls_features = cls_features.view(cls_features.size(0), -1)
        cls_output = self.classification_head(cls_features)
        
        return seg_output, cls_output
