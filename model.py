"""
PyTorch implementation of VGG16-UNet for Prostate Segmentation.
"""

import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle padding mismatches on odd resolutions
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        if diffY > 0 or diffX > 0:
            import torch.nn.functional as F
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class VGG16UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load full pretrained VGG16 features
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        
        # Encoder mappings based on original Keras architecture
        # s1: block1_conv2 (64 channels)
        self.enc1 = nn.Sequential(*vgg[:4])
        self.pool1 = vgg[4]
        
        # s2: block2_conv2 (128 channels)
        self.enc2 = nn.Sequential(*vgg[5:9])
        self.pool2 = vgg[9]
        
        # s3: block3_conv3 (256 channels)
        self.enc3 = nn.Sequential(*vgg[10:16])
        self.pool3 = vgg[16]
        
        # s4: block4_conv3 (512 channels)
        self.enc4 = nn.Sequential(*vgg[17:23])
        self.pool4 = vgg[23]

        # Bridge: block5_conv3 (512 channels)
        self.bridge = nn.Sequential(*vgg[24:30])
        
        # Decoder (upsampling & concat)
        self.dec1 = DecoderBlock(512, 512, 512)
        self.dec2 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec4 = DecoderBlock(128, 64, 64)
        
        # Output: 1 channel, sigmoid activation
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)         # (B, 64,  H,   W)
        p1 = self.pool1(s1)       # (B, 64,  H/2, W/2)
        
        s2 = self.enc2(p1)        # (B, 128, H/2, W/2)
        p2 = self.pool2(s2)       # (B, 128, H/4, W/4)
        
        s3 = self.enc3(p2)        # (B, 256, H/4, W/4)
        p3 = self.pool3(s3)       # (B, 256, H/8, W/8)
        
        s4 = self.enc4(p3)        # (B, 512, H/8, W/8)
        p4 = self.pool4(s4)       # (B, 512, H/16, W/16)
        
        # Bridge
        b = self.bridge(p4)       # (B, 512, H/16, W/16)
        
        # Decoder
        d1 = self.dec1(b, s4)     # (B, 512, H/8, W/8)
        d2 = self.dec2(d1, s3)    # (B, 256, H/4, W/4)
        d3 = self.dec3(d2, s2)    # (B, 128, H/2, W/2)
        d4 = self.dec4(d3, s1)    # (B, 64,  H,   W)
        
        # Output
        out = self.out_conv(d4)
        out = self.sigmoid(out)   # (B, 1,   H,   W)
        return out


def build_vgg16_unet(input_shape=None):
    """
    Returns the VGG16-UNet PyTorch model.
    input_shape is maintained for signature compatibility but unused in PyTorch init.
    """
    return VGG16UNet()


if __name__ == "__main__":
    model = build_vgg16_unet()
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")