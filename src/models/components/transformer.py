import math

import torch
from torch import nn
from pytorch_lightning import LightningModule
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, emb_size=768, img_size=96):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange("b e h w -> b (h w) e"),
        )

    def forward(self, x):
        x = self.projection(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerCaptchaSolver(LightningModule):
    def __init__(
        self,
        in_channels=1,
        patch_size=16,
        emb_size=768,
        img_size=96,
        num_classes=36,
        num_chars=5,
        num_layers=6,
        num_heads=12,
        hidden_size=2048,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.pos_encoder = PositionalEncoding(emb_size)

        encoder_layers = nn.TransformerEncoderLayer(
            emb_size, num_heads, hidden_size, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.fc1 = nn.Linear(emb_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes * num_chars)
        self.num_chars = num_chars
        self.num_classes = num_classes

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.fc1(x[:, -1, :])
        x = torch.relu(x)
        x = self.fc2(x)
        return x.view(-1, self.num_chars, self.num_classes)
