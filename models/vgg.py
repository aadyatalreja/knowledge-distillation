import torch
import torch.nn as nn

cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=100):  # Fixed: _init → __init__
        super(VGG, self).__init__()  # Fixed: _init_() → __init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    input_channel = 3
    for layer in cfg:
        if layer == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(input_channel, layer, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(layer))
            layers.append(nn.ReLU(inplace=True))
            input_channel = layer
    return nn.Sequential(*layers)

class VGG16Teacher(VGG):
    def __init__(self, pretrained_path=None, num_classes=10):  # Fixed: _init → __init__
        super(VGG16Teacher, self).__init__(make_layers(cfg['D'], batch_norm=True), num_classes)  # Fixed: _init_() → __init__()
        if pretrained_path:
            self.load_weights(pretrained_path)

    def load_weights(self, path):
        try:
            self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            print(f"Loaded pretrained weights from {path}")
        except Exception as e:
            print(f"Failed to load weights: {e}")

def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))