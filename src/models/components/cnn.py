from torch import nn


class CaptchaNet(nn.Module):
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

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size // 8, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(hidden_size // 8, hidden_size // 4, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(hidden_size // 4, hidden_size // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(hidden_size // 2, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 3 * 1, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_chars * num_classes),
        )

        self.num_chars = num_chars
        self.num_classes = num_classes

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(-1, self.num_chars, self.num_classes)
        return x


if __name__ == "__main__":
    _ = CaptchaNet()
