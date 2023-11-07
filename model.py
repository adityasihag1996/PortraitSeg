import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Define the D-Block as mentioned in the architecture
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        x = F.relu(self.depthwise(x))
        x = self.pointwise(x)
        x += residual
        return F.relu(x)

# Define the Encoder using MobileNetV2
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        mobile_net = models.mobilenet_v2(pretrained=True).features
        self.stage1 = mobile_net[:2]   # 112x112
        self.stage2 = mobile_net[2:4]  # 56x56
        self.stage3 = mobile_net[4:7]  # 28x28
        self.stage4 = mobile_net[7:14] # 14x14
        self.stage5 = mobile_net[14:]  # 7x7

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        return x1, x2, x3, x4, x5

# Define the Decoder with the DBlock and up-sampling layers
class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.up5 = nn.ConvTranspose2d(1280, 96, 2, stride=2)
        self.dblock5 = DBlock(96 + 32, 96)
        self.up4 = nn.ConvTranspose2d(96, 32, 2, stride=2)
        self.dblock4 = DBlock(32 + 24, 32)
        self.up3 = nn.ConvTranspose2d(32, 24, 2, stride=2)
        self.dblock3 = DBlock(24 + 16, 24)
        self.up2 = nn.ConvTranspose2d(24, 16, 2, stride=2)
        self.dblock2 = DBlock(16 + 8, 16)
        self.up1 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.dblock1 = DBlock(8, 8)
        self.final_conv = nn.Conv2d(8, num_classes, kernel_size=1)

    def forward(self, x1, x2, x3, x4, x5):
        u5 = self.up5(x5)
        x4 = torch.cat([u5, x4], dim=1)
        x4 = self.dblock5(x4)
        
        u4 = self.up4(x4)
        x3 = torch.cat([u4, x3], dim=1)
        x3 = self.dblock4(x3)
        
        u3 = self.up3(x3)
        x2 = torch.cat([u3, x2], dim=1)
        x2 = self.dblock3(x2)
        
        u2 = self.up2(x2)
        x1 = torch.cat([u2, x1], dim=1)
        x1 = self.dblock2(x1)
        
        u1 = self.up1(x1)
        out = self.dblock1(u1)
        out = self.final_conv(out)
        return out

# Define the overall PortraitNet model
class PortraitNet(nn.Module):
    def __init__(self, num_classes):
        super(PortraitNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        out = self.decoder(x1, x2, x3, x4, x5)
        return out
