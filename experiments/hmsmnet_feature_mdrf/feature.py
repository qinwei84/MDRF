import torch
import torch.nn as nn
import torch.nn.functional as F
from mdrf import MDRFConv

L2 = 1.0e-5

# ... (原始的 _calculate_same_padding, conv2d, avg_pool 保持不变) ...
def _calculate_same_padding(kernel_size, dilation_rate):
    return (kernel_size - 1) // 2 * dilation_rate

def conv2d(filters, kernel_size, strides, padding, dilation_rate, in_channels):
    pad = 0
    if padding == 'same':
        pad = _calculate_same_padding(kernel_size, dilation_rate)
    elif isinstance(padding, int):
        pad = padding
    conv_layer = nn.Conv2d( 
        in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, 
        stride=strides, padding=pad, dilation=dilation_rate, bias=True
    )
    nn.init.kaiming_normal_(conv_layer.weight, mode='fan_out', nonlinearity='relu')
    if conv_layer.bias is not None:
        nn.init.constant_(conv_layer.bias, 0)
    return conv_layer

def conv2d_bn(filters, kernel_size, strides, padding, dilation_rate, activation, in_channels):
    pad = 0
    if padding == 'same':
        pad = _calculate_same_padding(kernel_size, dilation_rate)
    elif isinstance(padding, int):
        pad = padding
    conv_layer = nn.Conv2d( 
            in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, 
            stride=strides, padding=pad, dilation=dilation_rate, bias=False
    )
    nn.init.kaiming_normal_(conv_layer.weight, mode='fan_out', nonlinearity='relu')
    bn_layer = nn.BatchNorm2d(filters, eps=0.001, momentum=0.01)
    layers = [conv_layer, bn_layer]
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

def avg_pool(pool_size, output_filters, input_filters):
    conv1x1 = nn.Conv2d(in_channels=input_filters, out_channels=output_filters, kernel_size=1, stride=1, bias=True)
    nn.init.kaiming_normal_(conv1x1.weight, mode='fan_out', nonlinearity='relu')
    if conv1x1.bias is not None:
        nn.init.constant_(conv1x1.bias, 0)
    return nn.Sequential(nn.AvgPool2d(kernel_size=pool_size), conv1x1)


# --- 【核心修改】在这里，我们将定义一个全新的MDRFBasicBlock ---

class MDRFBasicBlock(nn.Module):
    """
    残差块的MDRF版本 (MDRF-enhanced Basic Block)。
    将固定的3x3卷积替换为我们的MDRFConv，赋予特征提取动态感知能力。
    """
    def __init__(self, filters, in_channels, scales=[1, 2]):
        super(MDRFBasicBlock, self).__init__()
        
        # 使用MDRFConv替换标准卷积，scales可以作为超参数调整
        self.conv1 = nn.Sequential(
            MDRFConv(in_channels, filters, kernel_size=3, scales=scales),
            nn.BatchNorm2d(filters, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            MDRFConv(filters, filters, kernel_size=3, scales=scales),
            nn.BatchNorm2d(filters, eps=0.001, momentum=0.01)
        )
        self.relu = nn.ReLU(inplace=True)
        
        # shortcut部分逻辑与原始BasicBlock完全保持一致，确保公平对比
        self.shortcut = nn.Sequential() 
        if in_channels != filters: 
            shortcut_conv = nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, bias=False)
            nn.init.kaiming_normal_(shortcut_conv.weight, mode='fan_out', nonlinearity='relu') 
            shortcut_bn = nn.BatchNorm2d(filters, eps=0.001, momentum=0.01)
            self.shortcut = nn.Sequential(shortcut_conv, shortcut_bn)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        out = self.relu(out)
        return out

def make_mdrf_blocks(block_filters, num_blocks, current_in_channels, scales=[1, 2]):
    """
    构建一个包含多个MDRFBasicBlock的序列。
    """
    blocks = []
    for i in range(num_blocks):
        blocks.append(MDRFBasicBlock(block_filters,
                                     in_channels=current_in_channels if i == 0 else block_filters,
                                     scales=scales))
        if i == 0:
            current_in_channels = block_filters
    return nn.Sequential(*blocks)

class FeatureExtraction(nn.Module):
    def __init__(self, filters, in_channels=1, debug_save_path=None, debug_save_freq=200):
        super(FeatureExtraction, self).__init__()
        
        initial_filters = filters
        backbone_filters = 2 * initial_filters

        # 第一个卷积层，步长为2，进行2倍下采样
        self.conv0_1 = conv2d_bn(initial_filters, 5, 2, 'same', 1, True, in_channels=in_channels)
        # 第二个卷积层，步长为2，再进行2倍下采样
        self.conv0_2 = conv2d_bn(backbone_filters, 5, 2, 'same', 1, True, in_channels=initial_filters)
        
        # --- 【核心修改】在这里，我们将原始的残差块替换为MDRF残差块 ---
        # 我们将所有的残差块都进行替换，以最大化MDRF模块的影响力
        self.conv1_0 = make_mdrf_blocks(backbone_filters, 4, current_in_channels=backbone_filters, scales=[1, 2])
        self.conv1_1 = make_mdrf_blocks(backbone_filters, 2, current_in_channels=backbone_filters, scales=[1, 2, 4])
        self.conv1_2 = make_mdrf_blocks(backbone_filters, 2, current_in_channels=backbone_filters, scales=[1, 4, 8])
        self.conv1_3 = make_mdrf_blocks(backbone_filters, 2, current_in_channels=backbone_filters, scales=[1, 2])
        
        # 多尺度分支保持不变
        self.branch0 = avg_pool(pool_size=1, output_filters=initial_filters, input_filters=backbone_filters)
        self.branch1 = avg_pool(pool_size=2, output_filters=initial_filters, input_filters=backbone_filters)
        self.branch2 = avg_pool(pool_size=4, output_filters=initial_filters, input_filters=backbone_filters)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.conv0_2(x)
        x = self.conv1_0(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return [x0, x1, x2]