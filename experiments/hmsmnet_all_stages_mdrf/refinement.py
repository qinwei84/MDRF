import torch
import torch.nn as nn
import torch.nn.functional as F
from mdrf import MDRFConv

# L2正则项系数（PyTorch中一般通过weight_decay在优化器里实现）
L2 = 1.0e-5

# --- 我们将保留原始的 conv_bn_act, 因为它在深层可能更稳定 ---
def conv_bn_act(filters, kernel_size, strides, padding, dilation_rate, in_channels):
    """
    Conv2d + BatchNorm2d + LeakyReLU
    """
    pad = padding if isinstance(padding, int) else (kernel_size // 2 if padding == 'same' else 0)
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,
            stride=strides, padding=pad, dilation=dilation_rate, bias=False
        ),
        nn.BatchNorm2d(filters),
        nn.LeakyReLU(negative_slope=0.3, inplace=True)
    )

# --- 保留原始的 conv2d, 用于最后的输出层 ---
def conv2d(filters, kernel_size, strides, padding, dilation_rate, in_channels):
    """
    2D卷积层，带padding和dilation
    """
    pad = padding if isinstance(padding, int) else (kernel_size // 2 if padding == 'same' else 0)
    return nn.Conv2d(
        in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,
        stride=strides, padding=pad, dilation=dilation_rate, bias=True
    )

class Refinement(nn.Module):
    """
    视差细化模块 - 升级版
    在入口处使用 MDRFConv2D 进行智能特征提取和修正。
    """
    def __init__(self, filters, img_channels=1):
        super(Refinement, self).__init__()
        
        # --- 1. 入口层: 使用MDRFConv2D进行智能特征提取 ---
        # 输入通道为: 1(视差)+C(图像)+1(gx)+1(gy)
        input_channels = 1 + img_channels + 1 + 1
        self.conv1 = nn.Sequential(
            MDRFConv(input_channels, filters, kernel_size=3, scales=[1, 2, 3]),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(negative_slope=0.3, inplace=True)
        )
        # --- 2. 第二层: 再次使用MDRFConv2D进行深层精炼 ---
        self.conv2 = nn.Sequential(
            MDRFConv(filters, filters, kernel_size=3, scales=[1, 2]),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(negative_slope=0.3, inplace=True)
        )
        
        # --- 3. 后续层: 回归到稳定、高效的标准卷积 ---
        self.conv3 = conv_bn_act(filters, 3, 1, 'same', 2, in_channels=filters)
        self.conv4 = conv_bn_act(filters, 3, 1, 'same', 3, in_channels=filters)
        self.conv5 = conv_bn_act(filters, 3, 1, 'same', 1, in_channels=filters)
        
        # --- 4. 输出层: 保持不变 ---
        self.conv6 = conv2d(1, 3, 1, 'same', 1, in_channels=filters)

    def forward(self, inputs):
        assert len(inputs) == 4
        disp, img, gx, gy = inputs[0], inputs[1], inputs[2], inputs[3]

        if disp.shape[0] == 0:
            return disp.new_zeros([0, 1, img.shape[2], img.shape[3]])

        scale_factor = img.shape[2] / disp.shape[2]
        disp_up = F.interpolate(disp, size=(img.shape[2], img.shape[3]), mode='bilinear', align_corners=False)
        disp_up = disp_up * scale_factor

        concat = torch.cat([disp_up, img, gx, gy], dim=1)

        delta = self.conv1(concat)
        delta = self.conv2(delta)
        delta = self.conv3(delta)
        delta = self.conv4(delta)
        delta = self.conv5(delta)
        delta = self.conv6(delta)
        
        if delta.shape[2:] != disp_up.shape[2:]:
            delta = F.interpolate(delta, size=disp_up.shape[2:], mode='bilinear', align_corners=False)

        disp_final = disp_up + delta
        return disp_final