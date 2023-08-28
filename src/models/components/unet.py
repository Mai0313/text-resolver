import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class CaptchaUNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 kernel_size: int = 2,
                 stride: int = 1,
                 num_classes: int=36,
                 num_chars: int=5
                 ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.num_chars = num_chars
        self.num_classes = num_classes

        self.enc1 = UNetBlock(in_channels, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)

        self.pool = nn.MaxPool2d(self.stride)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=self.kernel_size, stride=self.stride)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=self.kernel_size, stride=self.stride)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=self.kernel_size, stride=self.stride)

        self.dec3 = UNetBlock(512, 256)
        self.dec2 = UNetBlock(256, 128)
        self.dec1 = UNetBlock(128, 64)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_conv = nn.Conv2d(64, num_classes * num_chars, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        upconv3_out = self.upconv3(self.pool(enc4))
        pooled_enc3 = self.pool(enc3)
        if upconv3_out.shape[2:] != pooled_enc3.shape[2:]:
            diffY = pooled_enc3.size()[2] - upconv3_out.size()[2]
            diffX = pooled_enc3.size()[3] - upconv3_out.size()[3]
            upconv3_out = F.pad(upconv3_out, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        dec3 = self.dec3(torch.cat([self.upconv3(self.pool(enc4)), self.pool(enc3)], 1))
        dec2 = self.dec2(torch.cat([self.upconv2(self.pool(dec3)), self.pool(enc2)], 1))
        dec1 = self.dec1(torch.cat([self.upconv1(self.pool(dec2)), self.pool(enc1)], 1))

        avg_pooled = self.global_avg_pool(dec1)
        out = self.final_conv(avg_pooled)
        out = out.view(-1, self.num_chars, self.num_classes)
        return out


if __name__ == "__main__":
    _ = CaptchaUNet()
