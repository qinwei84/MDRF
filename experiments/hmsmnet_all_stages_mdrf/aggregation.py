import torch  # 导入PyTorch主库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式API
from mdrf import MDRFConv3D

L2 = 1.0e-5  # L2正则项系数
alpha = 0.2  # LeakyReLU负斜率

def conv3d(filters, kernel_size, strides, padding, in_channels):
    """
    3D卷积，输入输出shape: [B, C, D, H, W] -> [B, filters, D_out, H_out, W_out]
    """
    pad = padding if isinstance(padding, int) else (kernel_size // 2 if padding == 'same' else 0)  # 计算padding
    return nn.Conv3d(
        in_channels=in_channels,  # 输入通道数
        out_channels=filters,     # 输出通道数
        kernel_size=kernel_size,  # 卷积核大小
        stride=strides,           # 步长
        padding=pad,              # padding
        bias=True                 # 是否用偏置
    )

def conv3d_bn(filters, kernel_size, strides, padding, activation, in_channels):
    """
    3D卷积 + BN + 激活（可选）
    """
    pad = padding if isinstance(padding, int) else (kernel_size // 2 if padding == 'same' else 0)  # 计算padding
    layers = [
        nn.Conv3d(
            in_channels=in_channels,  # 输入通道
            out_channels=filters,     # 输出通道
            kernel_size=kernel_size,  # 卷积核大小
            stride=strides,           # 步长
            padding=pad,              # padding
            bias=False                # 不带偏置
        ),
        nn.BatchNorm3d(filters)        # 3D批归一化
    ]
    if activation:  # 如果需要激活函数
        layers.append(nn.LeakyReLU(negative_slope=alpha, inplace=True))  # LeakyReLU激活
    return nn.Sequential(*layers)  # 顺序组合返回

def trans_conv3d_bn(filters, kernel_size, strides, padding, activation, in_channels):
    """
    3D反卷积(上采样) + BN + 激活（可选）
    """
    if isinstance(padding, str) and padding == 'same':
        if kernel_size >= strides:
            pad = (kernel_size - strides) // 2
        else:
            # 当 kernel_size < strides 且 padding='same' 时，情况较复杂，
            # 可能需要特定的 output_padding。这里暂时设为0，依赖后续F.interpolate。
            pad = 0
        # 注意: 上述计算假设 output_padding=0。在PyTorch中，为了精确匹配TF的输出尺寸，
        # 有时还需要配合调整 output_padding 参数。
        # Hourglass 中的 F.interpolate 会作为最终的维度对齐保障。
    elif isinstance(padding, int):
        pad = padding
    else: # padding == 'valid' or 0
        pad = 0
    
    layers = [
        nn.ConvTranspose3d(
            in_channels=in_channels,  # 输入通道
            out_channels=filters,     # 输出通道
            kernel_size=kernel_size,  # 卷积核大小
            stride=strides,           # 步长（放大倍数）
            padding=pad,              # padding
            bias=False                # 不带偏置
        ),
        nn.BatchNorm3d(filters)        # 3D批归一化
    ]
    if activation:  # 如果需要激活函数
        layers.append(nn.LeakyReLU(negative_slope=alpha, inplace=True))  # LeakyReLU激活
    return nn.Sequential(*layers)  # 顺序组合返回

class Hourglass(nn.Module):
    """
    Hourglass 3D模块，层次特征聚合 (与您现有版本基本一致)
    输入: [B, C_in, D, H, W]
    输出: [B, C_out, D, H, W] (C_out 通常是 filters)
    """
    def __init__(self, filters, in_channels): # filters 是指 hourglass 主要输出的通道数
        super(Hourglass, self).__init__()
        # 内部卷积的通道数变化与原版TF逻辑一致
        # --- 新的实现 ---
        self.conv1 = nn.Sequential(
            MDRFConv3D(in_channels, filters, scales=[1, 2]),
            nn.BatchNorm3d(filters),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            MDRFConv3D(filters, filters, scales=[1, 2]),
            nn.BatchNorm3d(filters),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        # 下采样层，输出通道变为2*filters
        self.conv3 = conv3d_bn(2 * filters, 3, 2, 'same', True, in_channels=filters) # e.g., 16->32
        self.conv4 = conv3d_bn(2 * filters, 3, 1, 'same', True, in_channels=2*filters) # e.g., 32->32
        
        # 再次下采样
        self.conv5 = conv3d_bn(2 * filters, 3, 2, 'same', True, in_channels=2*filters) # e.g., 32->32
        self.conv6 = conv3d_bn(2 * filters, 3, 1, 'same', True, in_channels=2*filters) # e.g., 32->32
        
        # 上采样层
        # conv7 输出 2*filters 通道，与 x2 (conv4的输出) 维度和通道数都一致
        self.conv7 = trans_conv3d_bn(2 * filters, 4, 2, 'same', True, in_channels=2*filters) # e.g., 32->32
        # conv8 输出 filters 通道，与 x1 (conv2的输出) 维度和通道数都一致
        self.conv8 = trans_conv3d_bn(filters, 4, 2, 'same', True, in_channels=2*filters)     # e.g., 32->16

    def forward(self, x):
        x1_skip = self.conv1(x)   # 输出 channels = filters
        x1_skip = self.conv2(x1_skip) # 输出 channels = filters

        x2_down = self.conv3(x1_skip) # 输出 channels = 2*filters
        x2_skip = self.conv4(x2_down) # 输出 channels = 2*filters

        x3_down = self.conv5(x2_skip) # 输出 channels = 2*filters
        x3_inner = self.conv6(x3_down) # 输出 channels = 2*filters
        
        x4_up = self.conv7(x3_inner) # 输出 channels = 2*filters
        # 确保上采样后的x4_up与x2_skip的空间维度一致
        if x4_up.shape[-3:] != x2_skip.shape[-3:]:
            x4_up = F.interpolate(x4_up, size=x2_skip.shape[-3:], mode='trilinear', align_corners=False)
        x4_fuse = x4_up + x2_skip # 跳跃连接
        
        x5_up = self.conv8(x4_fuse) # 输出 channels = filters
        # 确保上采样后的x5_up与x1_skip的空间维度一致
        if x5_up.shape[-3:] != x1_skip.shape[-3:]:
            x5_up = F.interpolate(x5_up, size=x1_skip.shape[-3:], mode='trilinear', align_corners=False)
        x5_fuse = x5_up + x1_skip # 跳跃连接
        
        return x5_fuse

class FeatureFusion(nn.Module):
    """
    体积特征融合模块 - 升级版
    增加了MDRFConv3D作为空间精炼器，在通道注意力之前优化融合后的代价体。
    """
    def __init__(self, channels):
        super(FeatureFusion, self).__init__()
        
        # --- 新增：MDRFConv3D空间精炼器 ---
        # 它将在代价体相加后，进行一次智能的空间信息整合。
        self.spatial_refiner = nn.Sequential(
            MDRFConv3D(channels, channels, kernel_size=3, scales=[1, 2, 3]),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(negative_slope=alpha, inplace=True)
        )
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(channels, channels, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_list):
        coarser_features, finer_features = x_list[0], x_list[1]
        # ... (上采样和尺寸检查的代码保持不变) ...
        x1_upsampled = self.upsample(coarser_features)
        if x1_upsampled.shape[-3:] != finer_features.shape[-3:]:
            x1_upsampled = F.interpolate(x1_upsampled, size=finer_features.shape[-3:], mode='trilinear', align_corners=False)

        # 1. 初步合并（相加）
        x2_sum = x1_upsampled + finer_features
        
        # 2. --- 新增步骤：通过空间精炼器 ---
        refined_sum = self.spatial_refiner(x2_sum)

        # 3. 计算通道注意力权重 (在精炼后的特征上进行)
        b, c, _, _, _ = refined_sum.shape 
        att = self.avg_pool(refined_sum).view(b, c) 
        att = self.relu(self.fc1(att))
        att = self.sigmoid(self.fc2(att))
        
        att_weights_for_coarse = att
        att_weights_for_finer = 1.0 - att_weights_for_coarse
        
        # 4. 加权融合 (使用原始的上采样特征和精细特征)
        att_coarse_expanded = att_weights_for_coarse.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        att_finer_expanded = att_weights_for_finer.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fused_coarser = x1_upsampled * att_coarse_expanded
        fused_finer = finer_features * att_finer_expanded
        output = fused_coarser + fused_finer
        
        return output