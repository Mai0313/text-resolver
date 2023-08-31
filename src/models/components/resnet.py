import torch.nn as nn
from torchvision import models


class CaptchaResNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 num_classes: int = 36,
                 num_chars: int = 5,
                 hidden_size: int = 256
                 ):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_chars * num_classes)
        )
        self.num_chars = num_chars
        self.num_classes = num_classes

    def forward(self, x):
        x = self.features(x)

        # Global Average Pooling (GAP) layer
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        x = x.view(-1, self.num_chars, self.num_classes)
        return x
