import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV2Student(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2Student, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.last_channel, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)
