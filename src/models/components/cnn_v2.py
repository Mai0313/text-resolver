from torch import nn


class CaptchaNet(nn.Module):
    def __init__(self, nc=1, leakyRelu=False):
        super().__init__()

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [16, 32, 64, 64, 128, 128, 128]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module(f'conv{i}', nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module(f'relu{i}', nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module(f'relu{i}', nn.ReLU(True))

        convRelu(0)
        cnn.add_module(f'pooling{0}', nn.MaxPool2d(2, 2))
        convRelu(1)
        cnn.add_module(f'pooling{1}', nn.MaxPool2d(2, 2))
        convRelu(2, True)
        convRelu(3)
        cnn.add_module(f'pooling{2}', nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(4, True)
        convRelu(5)
        cnn.add_module(f'pooling{3}', nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(6, True)

        self.cnn = cnn

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 * 1, 512), # Update input size
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 5 * 36) # 5 characters, each character has 36 possibilities
        )

    def forward(self, input):
        x = self.cnn(input)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(-1, 5, 36)
        return x
