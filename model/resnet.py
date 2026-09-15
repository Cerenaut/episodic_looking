from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ResNetConfig:
    in_channels: int = 3
    num_classes: int = 100

    # Number of residual blocks in each stage.
    # [2, 2, 2, 2] is approximately a CIFAR ResNet-18.
    blocks_per_stage: tuple[int, int, int, int] = (2, 2, 2, 2)

    # Channel width of the four stages.
    channels: tuple[int, int, int, int] = (64, 128, 256, 512)

    kernel_size: int = 3
    use_bias: bool = False

    # BatchNorm is conventional for this architecture.
    use_batch_norm: bool = True


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        config: ResNetConfig,
        activation: bool = True,
    ):
        super().__init__()

        padding = kernel_size // 2

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=config.use_bias,
            )
        ]

        if config.use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if activation:
            layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """
    Standard two-convolution residual block.

    If stride == 1 and the channel count is unchanged:

        output = F(x) + x

    If the shape changes:

        output = F(x) + projection(x)

    where projection is a learned 1x1 convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        config: ResNetConfig,
    ):
        super().__init__()

        self.conv1 = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=config.kernel_size,
            stride=stride,
            config=config,
            activation=True,
        )

        self.conv2 = ConvLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=config.kernel_size,
            stride=1,
            config=config,
            activation=False,
        )

        # Identity shortcut when possible.
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()

        # Projection shortcut when spatial size or channel count changes.
        else:
            self.shortcut = ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                config=config,
                activation=False,
            )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x:torch.Tensor, bias:torch.Tensor|None):
        residual = self.shortcut(x)

        y = self.conv1(x)
        y = self.conv2(y)

        y = y + residual
        y = self.activation(y)

        if bias is not None:
            y = y * bias
        return y


class ResidualStage(nn.Module):
    """
    A sequence of residual blocks.

    The first block optionally performs downsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        config: ResNetConfig,
    ):
        super().__init__()

        blocks = [
            ResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                config=config,
            )
        ]

        for _ in range(num_blocks - 1):
            blocks.append(
                ResidualBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    config=config,
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x:torch.Tensor, bias:torch.Tensor|None):
        # Apply bias to channels dim of each block
        for block in self.blocks:
            x = block(x=x, bias=bias)  # Can't pass args to Sequential
        return x
