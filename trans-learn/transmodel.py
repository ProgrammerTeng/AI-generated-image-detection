import torch
import torch.nn as nn
from torchvision.models import resnet50


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False):
        super().__init__()
        # 加载预训练ResNet50
        self.backbone = resnet50(pretrained=pretrained)

        # 冻结卷积层（可选）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 替换最后的全连接层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # 权重初始化
        self._init_weights(self.backbone.fc)

    def forward(self, x):
        return self.backbone(x)

    def _init_weights(self, module):
        for layer in module:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)


def create_model(device='cuda:0', freeze_backbone=False):
    model = ResNetClassifier(freeze_backbone=freeze_backbone).to(device)
    print(f"Loaded ResNet50 with {'frozen' if freeze_backbone else 'unfrozen'} backbone")
    return model