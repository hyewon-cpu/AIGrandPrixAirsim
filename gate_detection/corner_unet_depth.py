from __future__ import annotations

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CornerUNetDepth(nn.Module):
    """5-level U-Net for depth-map inputs (single-channel)."""

    def __init__(self, in_channels: int = 1, out_channels: int = 4):
        super().__init__()
        enc_channels = [12, 18, 24, 32, 32]
        enc_kernels = [3, 3, 3, 5, 7]

        self.enc1 = ConvLayer(in_channels, enc_channels[0], enc_kernels[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvLayer(enc_channels[0], enc_channels[1], enc_kernels[1])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvLayer(enc_channels[1], enc_channels[2], enc_kernels[2])
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvLayer(enc_channels[2], enc_channels[3], enc_kernels[3])
        self.pool4 = nn.MaxPool2d(2)
        self.enc5 = ConvLayer(enc_channels[3], enc_channels[4], enc_kernels[4])

        self.up4 = nn.ConvTranspose2d(enc_channels[4], enc_channels[3], kernel_size=2, stride=2)
        self.dec4 = ConvLayer(enc_channels[3] * 2, enc_channels[3], enc_kernels[3])
        self.up3 = nn.ConvTranspose2d(enc_channels[3], enc_channels[2], kernel_size=2, stride=2)
        self.dec3 = ConvLayer(enc_channels[2] * 2, enc_channels[2], enc_kernels[2])
        self.up2 = nn.ConvTranspose2d(enc_channels[2], enc_channels[1], kernel_size=2, stride=2)
        self.dec2 = ConvLayer(enc_channels[1] * 2, enc_channels[1], enc_kernels[1])
        self.up1 = nn.ConvTranspose2d(enc_channels[1], enc_channels[0], kernel_size=2, stride=2)
        self.dec1 = ConvLayer(enc_channels[0] * 2, enc_channels[0], enc_kernels[0])

        # Final additional 12-filter layer on top of the U-Net output.
        self.refine = ConvLayer(enc_channels[0], 12, kernel_size=3)
        self.head = nn.Conv2d(12, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        e5 = self.enc5(self.pool4(e4))

        d4 = self.up4(e5)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d1 = self.refine(d1)
        return self.head(d1)

