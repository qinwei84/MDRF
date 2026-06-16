"""Official MDRF operators used in the paper experiments.

This file keeps the implementation structure of the original experimental
`mdrf_conv.py` and only translates/adds comments for public release. The core
operation is a shared-kernel multi-dilation convolution with an input-dependent
branch gate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MDRFConv(nn.Module):
    """
    Multi-scale Dynamic Receptive Field convolution for 2D feature maps.

    Key design:
    - A single base convolution kernel is shared by all dilation branches.
    - A lightweight input-level gate predicts branch weights from global context.
    - The branch responses are fused adaptively for each input sample.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, scales=[1, 2, 3], **kwargs):
        """
        Initialize the 2D MDRF convolution.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Shared convolution kernel size.
            stride: Shared stride for all dilation branches.
            scales: Dilation rates used by the parallel branches.
            **kwargs: Kept for compatibility with experiment code.
        """
        super().__init__()
        self.scales = scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # Shared base convolution parameters. The same weight and bias are used
        # by every dilation branch, so the receptive-field scale changes without
        # instantiating independent kernels for each branch.
        self.base_conv_weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.base_conv_bias = nn.Parameter(torch.Tensor(out_channels))

        # Lightweight branch-weight generator. Global pooling summarizes the
        # input feature map, and the 1x1 convolution predicts one normalized
        # weight for each dilation branch.
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, len(scales), kernel_size=1),
            nn.Softmax(dim=1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.base_conv_weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.base_conv_bias, 0)

    def forward(self, x):
        # Generate dynamic branch weights.
        # Shape: [B, N, 1, 1], where N is the number of dilation scales.
        weights = self.weight_gen(x)

        # Parallel multi-scale convolution with shared weights. For odd kernels
        # and stride=1, the dynamic padding keeps the output H/W unchanged.
        multi_scale_feats = []
        for d in self.scales:
            padding = d * (self.kernel_size - 1) // 2

            feat = F.conv2d(
                x,
                weight=self.base_conv_weight,
                bias=self.base_conv_bias,
                stride=self.stride,
                padding=padding,
                dilation=d
            )
            multi_scale_feats.append(feat)

        # Adaptive weighted fusion. Each branch weight has shape [B, 1, 1, 1]
        # and is broadcast over the corresponding feature response.
        fused_feat = sum(w * feat for w, feat in zip(weights.split(1, 1), multi_scale_feats))

        return fused_feat


class MDRFConv3D(nn.Module):
    """
    3D MDRF convolution for stereo cost-volume aggregation.

    This is the volumetric counterpart of `MDRFConv`. It shares one 3D kernel
    across dilation branches and uses a 3D global gate to fuse multi-scale
    cost-volume responses.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, scales=[1, 2, 3], **kwargs):
        super().__init__()
        self.scales = scales
        self.kernel_size = kernel_size
        self.stride = stride

        # Shared 3D convolution parameters for all dilation branches.
        self.base_conv_weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.base_conv_bias = nn.Parameter(torch.Tensor(out_channels))

        # 3D branch-weight generator. The gate predicts one normalized branch
        # weight per dilation scale from the whole cost volume.
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, len(scales), kernel_size=1),
            nn.Softmax(dim=1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.base_conv_weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.base_conv_bias, 0)

    def forward(self, x):
        # Input shape: [B, C, D, H, W].
        # Gate shape: [B, N, 1, 1, 1], where N is the number of dilation scales.
        weights = self.weight_gen(x)

        multi_scale_feats = []
        for d in self.scales:
            padding = d * (self.kernel_size - 1) // 2
            feat = F.conv3d(
                x,
                weight=self.base_conv_weight,
                bias=self.base_conv_bias,
                stride=self.stride,
                padding=padding,
                dilation=d
            )
            multi_scale_feats.append(feat)

        fused_feat = sum(w * feat for w, feat in zip(weights.split(1, 1), multi_scale_feats))
        return fused_feat
