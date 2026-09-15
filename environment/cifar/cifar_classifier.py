import torch
import torch.nn.functional as F
from torch import nn

from model.resnet import ConvLayer, ResidualStage, ResNetConfig


class CifarResNet(nn.Module):
    """
    CIFAR ResNet encoder.

    Input:
        [B, 3, 32, 32]

    Output:
        [B, final_channels]

    For the default configuration:

        [B, 3, 32, 32]
          -> [B, 64, 32, 32]
          -> [B, 128, 16, 16]
          -> [B, 256,  8,  8]
          -> [B, 512,  4,  4]
          -> [B, 512]
    """

    def __init__(self, config: ResNetConfig, bias_stage:int):
        super().__init__()

        self.config = config
        self.bias_stage = bias_stage

        c1, c2, c3, c4 = config.channels
        n1, n2, n3, n4 = config.blocks_per_stage

        # CIFAR stem: deliberately no initial downsampling.
        self.stem = ConvLayer(
            in_channels=config.in_channels,
            out_channels=c1,
            kernel_size=3,
            stride=1,
            config=config,
            activation=True,
        )

        # 4 Stages each 2x ResidualBlocks 
        self.stage1 = ResidualStage(
            in_channels=c1,
            out_channels=c1,
            num_blocks=n1,
            stride=1,
            config=config,
        )

        self.stage2 = ResidualStage(
            in_channels=c1,
            out_channels=c2,
            num_blocks=n2,
            stride=2,
            config=config,
        )

        self.stage3 = ResidualStage(
            in_channels=c2,
            out_channels=c3,
            num_blocks=n3,
            stride=2,
            config=config,
        )

        self.stage4 = ResidualStage(
            in_channels=c3,
            out_channels=c4,
            num_blocks=n4,
            stride=2,
            config=config,
        )

        # 4x4 -> 1x1
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.output_dim = c4

    def forward(self, x:torch.Tensor, bias:torch.Tensor|None):
        x = self.stem(x)

        # Option to change where the gating is applied programmatically
        g1 = self.get_gating(stage=1, bias=bias)
        g2 = self.get_gating(stage=2, bias=bias)
        g3 = self.get_gating(stage=3, bias=bias)
        g4 = self.get_gating(stage=4, bias=bias)

        e = None
        x = self.stage1(x=x, bias=g1)
        if g1 is not None:
            e = x.mean(dim=(2, 3,))
        x = self.stage2(x=x, bias=g2)
        if g2 is not None:
            e = x.mean(dim=(2, 3,))
        x = self.stage3(x=x, bias=g3)
        if g3 is not None:
            e = x.mean(dim=(2, 3,))
        x = self.stage4(x=x, bias=g4)  # B, C=512, H=4, W=4 = [B,C,H,W]
        if g4 is not None:
            e = x.mean(dim=(2, 3,))

        x = self.global_pool(x) # 4x4 -> 1x1

        # [B, 512, 1, 1] -> [B, 512]
        x = torch.flatten(x, start_dim=1)
        return x, e

    def get_gating(self, stage:int, bias:torch.Tensor|None):
        if bias is None or stage != self.bias_stage:
            return None  # no bias applied

        # Top-down Multiplicative Gating Integration Hook (Post-Activation, Pre-Pooling)
        # Bias comes directly as a continuous output from the PPO policy actor.
        # Using 2 * sigmoid constrains it safely between [0, 2] 
        # to support both inhibition and excitation.
        gain = 1.0
        gate = 2.0 * torch.sigmoid(gain * bias)  # Baseline

        # Some other gates were tested to explore resistance to bias at different stages.
        # Random gate (97% -> 7% train acc)
        #gate = torch.randn(bias.shape[0], bias.shape[1])

        # Knockout half gate (97% -> 68% train acc)
        #gate = torch.ones_like(bias)  # Neutral
        #gate[:, :64] = 0.0
        #gate[:, 64:] = 1.9

        # Reshape to gate format
        g = gate.unsqueeze(-1).unsqueeze(-1) 
        return g


class CifarClassifierHead(nn.Module):
    """
    Simple classifier head operating on the ResNet representation.

    Input:
        [B, feature_dim]

    Output:
        [B, num_classes]
    """

    def __init__(
        self,
        feature_dim: int,
        config: ResNetConfig,
    ):
        super().__init__()

        self.linear = nn.Linear(
            feature_dim,
            config.num_classes,
        )

    def forward(self, x):
        return self.linear(x)


class CifarClassifier(nn.Module):
    """
    Complete encoder + classifier.

    The "encoding" is taken to be the final output of the ResNet.
    The classifier output is the logits from the classifier head.

    Kept compositional so the encoder and classifier can later be
    frozen/replaced independently.
    """

    def __init__(self, config: ResNetConfig, bias_stage:int):
        super().__init__()

        self.encoder = CifarResNet(config, bias_stage=bias_stage)

        self.classifier = CifarClassifierHead(
            feature_dim=self.encoder.output_dim,
            config=config,
        )

    def get_bias_size(self) -> int:
        return self.encoder.config.channels[self.encoder.bias_stage -1]

    def get_encoded_size(self) -> int:
        return self.encoder.output_dim

    def features(self, x:torch.Tensor, bias:torch.Tensor|None) -> torch.Tensor:
        features, encoding = self.encoder(x=x, bias=bias)
        return features, encoding

    def classify(self, features:torch.Tensor) -> torch.Tensor:    
        logits = self.classifier(features)
        return logits

    def forward(self, x:torch.Tensor, bias:torch.Tensor|None):
        features, encoding = self.encoder(x=x, bias=bias)
        logits = self.classifier(features)
        return logits, encoding

    def loss(self, x, targets):
        logits = self(x)
        return F.cross_entropy(logits, targets)