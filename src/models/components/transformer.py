import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.layer_norm1(x)
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.layer_norm2(x)
        return x


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
            nn.Conv2d(
                in_channels,
                hidden_size // 8,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(
                hidden_size // 8,
                hidden_size // 4,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(
                hidden_size // 4,
                hidden_size // 2,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(
                hidden_size // 2,
                hidden_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        d_model = hidden_size * 3 * 1  # 根據你的特徵提取器來調整
        nhead = 4
        dim_feedforward = 512

        self.transformer_block = TransformerBlock(d_model, nhead, dim_feedforward)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_size * 2),
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
        x = x.unsqueeze(1)  # 添加序列維度
        x = self.transformer_block(x)
        x = x.squeeze(1)  # 去除序列維度
        x = self.classifier(x)
        x = x.view(-1, self.num_chars, self.num_classes)
        return x
