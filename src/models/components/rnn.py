import torch
import torch.nn as nn


class CaptchaRNN(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 36,
                 num_chars: int = 5,
                 hidden_size: int = 256,
                 rnn_type: str = 'LSTM',
                 num_layers: int = 1):
        super().__init__()

        # CNN feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_input_size = 7 * 2 * 256  # After three MaxPool2d with kernel size 2, the shape is [7, 2, 256]

        # RNN layer
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.rnn_input_size, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(self.rnn_input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer (classifier)
        self.classifier = nn.Linear(hidden_size, num_chars * num_classes)

        self.num_chars = num_chars
        self.num_classes = num_classes

    def forward(self, x):
        # Feature extraction
        x = self.feature_extractor(x)

        # Prepare data for RNN
        x = x.view(x.size(0), -1, self.rnn_input_size)

        # RNN layer
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        if isinstance(self.rnn, nn.LSTM):
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)

        # Classifier
        out = self.classifier(out[:, -1, :])
        out = out.view(-1, self.num_chars, self.num_classes)

        return out
