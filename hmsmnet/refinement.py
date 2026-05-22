import torch  # 导入PyTorch主库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式API

# L2正则项系数（PyTorch中一般通过weight_decay在优化器里实现）
L2 = 1.0e-5

def conv2d(filters, kernel_size, strides, padding, dilation_rate, in_channels):
    """
    2D卷积层，带padding和dilation
    输入shape: [B, C, H, W], 输出shape: [B, filters, H_out, W_out]
    """
    pad = padding if isinstance(padding, int) else (kernel_size // 2 if padding == 'same' else 0)  # 计算padding
    return nn.Conv2d(
        in_channels=in_channels,      # 输入通道数
        out_channels=filters,         # 输出通道数
        kernel_size=kernel_size,      # 卷积核大小
        stride=strides,               # 步长
        padding=pad,                  # 填充
        dilation=dilation_rate,       # 膨胀系数
        bias=True                     # 是否使用偏置
    )

def conv_bn_act(filters, kernel_size, strides, padding, dilation_rate, in_channels):
    """
    Conv2d + BatchNorm2d + LeakyReLU
    """
    pad = padding if isinstance(padding, int) else (kernel_size // 2 if padding == 'same' else 0)  # 计算padding
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,      # 输入通道数
            out_channels=filters,         # 输出通道数
            kernel_size=kernel_size,      # 卷积核大小
            stride=strides,               # 步长
            padding=pad,                  # 填充
            dilation=dilation_rate,       # 膨胀系数
            bias=False                    # 不使用偏置
        ),
        nn.BatchNorm2d(filters),          # 批归一化
        nn.LeakyReLU(negative_slope=0.3, inplace=True) # <--- 修改点：明确设置 alpha
    )

class Refinement(nn.Module):
    """
    视差细化模块。
    输入:
      inputs[0]: disparity, [B, 1, H1, W1]
      inputs[1]: rgb/灰度图, [B, C, H2, W2]
      inputs[2]: gx, [B, 1, H2, W2]
      inputs[3]: gy, [B, 1, H2, W2]
    输出:
      disp_final: [B, 1, H2, W2]
    """
    def __init__(self, filters, img_channels=1):
        super(Refinement, self).__init__()  # 父类初始化
        # 注意输入通道数，delta输入为: [B, 1+C+1+1, H2, W2]
        self.conv1 = conv_bn_act(filters, 3, 1, 'same', 1, in_channels=1+img_channels+1+1)  # 第一层卷积
        self.conv2 = conv_bn_act(filters, 3, 1, 'same', 1, in_channels=filters)             # 第二层卷积
        self.conv3 = conv_bn_act(filters, 3, 1, 'same', 2, in_channels=filters)             # 第三层卷积，膨胀2
        self.conv4 = conv_bn_act(filters, 3, 1, 'same', 3, in_channels=filters)             # 第四层卷积，膨胀3
        self.conv5 = conv_bn_act(filters, 3, 1, 'same', 1, in_channels=filters)             # 第五层卷积
        self.conv6 = conv2d(1, 3, 1, 'same', 1, in_channels=filters)                        # 输出delta_disp

    def forward(self, inputs):
        # inputs: [disparity, rgb/img, gx, gy]
        assert len(inputs) == 4  # 检查输入长度
        disp = inputs[0]  # [B, 1, H1, W1] 视差图
        img = inputs[1]   # [B, C, H2, W2] 输入图像
        gx = inputs[2]    # [B, 1, H2, W2] x方向梯度
        gy = inputs[3]    # [B, 1, H2, W2] y方向梯度
        # <<<<这里加空batch判断！>>>>
        if disp.shape[0] == 0:
            # 返回一个空Tensor，避免后续cat/conv报错
            return disp.new_zeros([0, 1, img.shape[2], img.shape[3]])

        # 将disp上采样到img分辨率，scale_factor = H2/H1
        scale_factor = img.shape[2] / disp.shape[2]  # 尺度因子
        disp_up = F.interpolate(disp, size=(img.shape[2], img.shape[3]), mode='bilinear', align_corners=False)  # 上采样
        disp_up = disp_up * scale_factor  # 保持数值尺度一致

        # 拼接特征 [B, 1+C+1+1, H2, W2]
        concat = torch.cat([disp_up, img, gx, gy], dim=1)  # 拼接disp、图像、梯度

        # 经过多层卷积
        delta = self.conv1(concat)  # 第一层
        delta = self.conv2(delta)   # 第二层
        delta = self.conv3(delta)   # 第三层
        delta = self.conv4(delta)   # 第四层
        delta = self.conv5(delta)   # 第五层
        delta = self.conv6(delta)   # 输出delta_disp [B, 1, H2, W2]
        # === 关键修正：自动对齐尺寸 ===
        if delta.shape[2:] != disp_up.shape[2:]:
            delta = F.interpolate(delta, size=disp_up.shape[2:], mode='bilinear', align_corners=False)

        disp_final = disp_up + delta # 残差细化，得到最终视差
        return disp_final  # 返回细化后视差图        