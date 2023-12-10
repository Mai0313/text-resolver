import torch.nn as nn
from lightning import LightningModule


class CaptchaUNet(LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        num_classes: int = 36,
        num_chars: int = 5,
        hidden_size: int = 256,
    ):
        super().__init__()

        # Encoder (Downsampling path)
        self.enc1 = self.conv_block(in_channels, hidden_size // 8, kernel_size, stride, padding)
        self.enc2 = self.conv_block(
            hidden_size // 8, hidden_size // 4, kernel_size, stride, padding
        )

        # Decoder (Upsampling path)
        self.dec1 = self.conv_block(
            hidden_size // 4, hidden_size // 8, kernel_size, stride, padding
        )

        # Classifier (The input dimension is set later dynamically)
        self.classifier = nn.Sequential(
            nn.Linear(0, hidden_size),  # Input dimension will be set dynamically
            nn.ReLU(),
            nn.Linear(hidden_size, num_chars * num_classes),
        )

        self.num_chars = num_chars
        self.num_classes = num_classes

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)

        # Decoder
        x = self.dec1(x2)

        # Dynamically calculate flat_size based on the output from the decoder
        self.flat_size = x.shape[1] * x.shape[2] * x.shape[3]

        # Update the input dimension for the first Linear layer in classifier
        self.classifier[0] = nn.Linear(self.flat_size, self.classifier[0].out_features).to(
            x.device
        )

        # Flatten and Classifier
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        x = x.view(-1, self.num_chars, self.num_classes)
        return x
