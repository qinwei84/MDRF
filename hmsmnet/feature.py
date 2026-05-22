import torch # 导入PyTorch主库
import torch.nn as nn # 导入PyTorch神经网络模块
import torch.nn.functional as F # 导入PyTorch函数式API，如interpolate
import os # 导入操作系统模块，用于路径操作（如创建文件夹）

L2 = 1.0e-5 # 定义L2正则化系数的全局变量（虽然在此PyTorch版本中主要通过优化器的weight_decay实现）

def _calculate_same_padding(kernel_size, dilation_rate):
    """
    辅助函数，用于计算在给定kernel_size和dilation_rate的情况下，
    为了实现'same'卷积（即输出空间维度与输入空间维度相同）所需的单边padding值。
    TensorFlow 'SAME' padding for dilation: (kernel_size - 1) // 2 * dilation_rate
    参数:
        kernel_size (int): 卷积核的原始大小 (例如 3)。
        dilation_rate (int): 卷积的膨胀率 (例如 1, 2, 4)。
    返回:
        int: 计算得到的单边padding值。
    """
    # TensorFlow 中 'SAME' padding 对于膨胀卷积的计算方式可以理解为：
    # 首先计算没有膨胀时的标准 'SAME' padding，即 (kernel_size - 1) // 2。
    # 然后，这个基础padding量在每个方向上被膨胀因子 dilation_rate 放大。
    # 例如，kernel_size=3, dilation_rate=1 => pad = (3-1)//2 * 1 = 1
    # 例如，kernel_size=3, dilation_rate=2 => pad = (3-1)//2 * 2 = 2
    return (kernel_size - 1) // 2 * dilation_rate

def conv2d(filters, kernel_size, strides, padding, dilation_rate, in_channels):
    """
    定义一个标准的2D卷积层，并根据padding参数智能计算padding值。
    参数:
        filters (int): 输出通道数。
        kernel_size (int or tuple): 卷积核大小。
        strides (int or tuple): 步长。
        padding (str or int): 填充方式。如果为 'same'，则自动计算以保持空间维度不变；
                              如果为整数，则直接使用该值作为单边padding。
        dilation_rate (int or tuple): 膨胀率。
        in_channels (int): 输入通道数。
    返回:
        nn.Conv2d: PyTorch的2D卷积层实例。
    """
    pad = 0 # 初始化padding值为0
    if padding == 'same': # 如果padding参数是字符串 'same'
        # 调用辅助函数计算实现 'same' 效果所需的单边padding值
        pad = _calculate_same_padding(kernel_size, dilation_rate)
    elif isinstance(padding, int): # 如果padding参数是一个整数
        pad = padding # 直接使用该整数作为单边padding值
    # else: pad保持为0，表示不进行显式padding (除非卷积核为1x1)
    ## ADDED: Kaiming Normal Initialization for Conv2d
    conv_layer = nn.Conv2d( 
        in_channels=in_channels, # 设置输入通道数
        out_channels=filters, # 设置输出通道数
        kernel_size=kernel_size, # 设置卷积核大小
        stride=strides, # 设置步长
        padding=pad, # 应用计算得到的或指定的单边padding值
        dilation=dilation_rate, # 设置膨胀率
        bias=True # 使用偏置项 (与原Keras版本行为一致)
    )
    nn.init.kaiming_normal_(conv_layer.weight, mode='fan_out', nonlinearity='relu')
    if conv_layer.bias is not None:
        nn.init.constant_(conv_layer.bias, 0)
    return conv_layer

def conv2d_bn(filters, kernel_size, strides, padding, dilation_rate, activation, in_channels):
    """
    定义一个包含2D卷积、批归一化（BatchNormalization）和可选ReLU激活函数的模块。
    padding参数的处理方式与conv2d函数类似。
    参数:
        filters (int): 输出通道数。
        kernel_size (int or tuple): 卷积核大小。
        strides (int or tuple): 步长。
        padding (str or int): 填充方式。
        dilation_rate (int or tuple): 膨胀率。
        activation (bool): 是否在BN之后添加ReLU激活函数。
        in_channels (int): 输入通道数。
    返回:
        nn.Sequential: 包含Conv-BN-(ReLU)的序列模块。
    """
    pad = 0 # 初始化padding值为0
    if padding == 'same': # 如果padding参数是字符串 'same'
        # 调用辅助函数计算实现 'same' 效果所需的单边padding值
        pad = _calculate_same_padding(kernel_size, dilation_rate)
    elif isinstance(padding, int): # 如果padding参数是一个整数
        pad = padding # 直接使用该整数作为单边padding值
    # else: pad保持为0
    ## 新增: 对conv2d_bn中的Conv2d进行Kaiming Normal初始化
    conv_layer = nn.Conv2d( 
            in_channels=in_channels, # 输入通道数
            out_channels=filters, # 输出通道数
            kernel_size=kernel_size, # 卷积核大小
            stride=strides, # 步长
            padding=pad, # 应用计算得到的或指定的单边padding值
            dilation=dilation_rate, # 膨胀率
            bias=False # 不使用偏置项，因为后续有批归一化(BN)层，BN层会学习自己的偏置
    )
    nn.init.kaiming_normal_(conv_layer.weight, mode='fan_out', nonlinearity='relu') # He正态初始化权重
    ## 修改: BatchNorm2d的参数 (eps 和 momentum) 以匹配TensorFlow Keras的默认值
    # 原始: nn.BatchNorm2d(filters)
    # TF Keras 默认: eps=0.001, momentum=0.99
    # PyTorch 等效 momentum: 0.01 (基于 Keras momentum 是旧均值权重的普遍理解)
    bn_layer = nn.BatchNorm2d(filters, eps=0.001, momentum=0.01) 
    # BatchNorm自身的学习参数gamma(weight)和beta(bias)在PyTorch中默认已初始化为1和0，通常无需再次显式初始化。
    # 若需显式:
    # nn.init.ones_(bn_layer.weight)
    # nn.init.zeros_(bn_layer.bias)

    layers = [conv_layer, bn_layer] # 创建一个层列表，包含卷积层和批归一化层
    
    if activation: # 如果activation参数为True
        layers.append(nn.ReLU(inplace=True)) # 在层列表中添加ReLU激活函数，inplace=True表示原地操作以节省内存
    return nn.Sequential(*layers) # 将层列表中的所有层按顺序组合成一个nn.Sequential模块并返回

def avg_pool(pool_size, output_filters, input_filters):
    """
    定义一个平均池化层后接一个1x1卷积层的模块，用于多尺度特征提取中的分支。
    参数:
        pool_size (int or tuple): 平均池化的核大小。
        output_filters (int): 1x1卷积后的输出通道数。
        input_filters (int): 输入到1x1卷积的通道数（即池化层的输出通道数）。
    返回:
        nn.Sequential: 包含AvgPool-Conv的序列模块。
    """
    ## 新增: 对avg_pool中的1x1卷积进行Kaiming Normal初始化
    conv1x1 = nn.Conv2d(in_channels=input_filters, out_channels=output_filters, kernel_size=1, stride=1, bias=True) # TF版本此处有偏置
    nn.init.kaiming_normal_(conv1x1.weight, mode='fan_out', nonlinearity='relu') # He正态初始化权重
    if conv1x1.bias is not None:
        nn.init.constant_(conv1x1.bias, 0) # 偏置初始化为0
    
    return nn.Sequential( 
        nn.AvgPool2d(kernel_size=pool_size), 
        conv1x1 
    )

class BasicBlock(nn.Module): # 定义残差学习的基本块 (ResNet中的BasicBlock)
    """
    残差块：(Conv-BN-ReLU) -> (Conv-BN) -> Shortcut Add -> ReLU
    """
    def __init__(self, filters, dilation_rate, in_channels):
        """
        初始化BasicBlock。
        参数:
            filters (int): 块内部卷积层以及块的输出通道数。
            dilation_rate (int or tuple): 卷积层的膨胀率。
            in_channels (int): 输入到这个块的特征图的通道数。
        """
        super(BasicBlock, self).__init__() # 调用父类nn.Module的初始化方法
        # conv1 和 conv2 将通过修改后的 conv2d_bn 函数进行初始化和BN参数设置
        self.conv1 = conv2d_bn(filters, 3, 1, 'same', dilation_rate, True, in_channels=in_channels)
        self.conv2 = conv2d_bn(filters, 3, 1, 'same', dilation_rate, False, in_channels=filters)
        self.relu = nn.ReLU(inplace=True) 
        
        self.shortcut = nn.Sequential() 
        if in_channels != filters: 
            ## 新增: 如果shortcut路径存在，对其Conv2d进行Kaiming Normal初始化
            shortcut_conv = nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, bias=False) # shortcut中的卷积通常无偏置，因后接BN
            nn.init.kaiming_normal_(shortcut_conv.weight, mode='fan_out', nonlinearity='relu') 
            # 注意：如果shortcut的1x1卷积后没有立即的ReLU（而是直接与主路相加后再ReLU），
            # 'nonlinearity'参数可能用'linear'更合适，或者保持'relu'影响不大。此处保持'relu'。
            
            ## 修改: 如果shortcut路径存在，对其BatchNorm2d参数进行设置
            shortcut_bn = nn.BatchNorm2d(filters, eps=0.001, momentum=0.01)

            self.shortcut = nn.Sequential(shortcut_conv, shortcut_bn)

    def forward(self, x): # 定义前向传播逻辑
        identity = self.shortcut(x) # 通过shortcut路径处理输入x，确保维度匹配
        out = self.conv1(x) # 第一个卷积序列
        out = self.conv2(out) # 第二个卷积序列
        # Keras原版BasicBlock不改变H,W，因此不需要F.interpolate来对齐空间尺寸
        # 如果后续有修改导致尺寸变化，则需要取消注释下面的对齐代码
        # if out.shape[-2:] != identity.shape[-2:]: 
        #     out = F.interpolate(out, size=identity.shape[-2:], mode='bilinear', align_corners=False)
        out = out + identity # 残差连接：主路径输出与shortcut路径输出相加
        out = self.relu(out) # 应用ReLU激活函数
        return out # 返回块的输出

def make_blocks(block_filters, dilation_rate, num_blocks, current_in_channels):
    """
    构建一个包含多个BasicBlock的序列。
    参数:
        block_filters (int): 每个BasicBlock内部以及最终输出的通道数。
        dilation_rate (int or tuple): BasicBlock中卷积层的膨胀率。
        num_blocks (int): 要堆叠的BasicBlock的数量。
        current_in_channels (int): 输入到这个序列的第一个BasicBlock的通道数。
    返回:
        nn.Sequential: 包含多个BasicBlock的序列模块。
    """
    blocks = [] # 初始化一个空列表来存放BasicBlock实例
    # 循环创建num_blocks个BasicBlock
    for i in range(num_blocks):
        # 第一个block的输入通道是current_in_channels，后续block的输入通道是block_filters
        # 所有block内部都产生block_filters个输出通道
        blocks.append(BasicBlock(block_filters, dilation_rate,
                                 in_channels=current_in_channels if i == 0 else block_filters))
        if i == 0: # 第一个block处理完毕后
            current_in_channels = block_filters # 后续block的输入通道数更新为block_filters
    return nn.Sequential(*blocks) # 将列表中的所有block组合成一个nn.Sequential模块

class FeatureExtraction(nn.Module): # 定义特征提取网络
    """
    多层卷积和残差块实现的特征提取主干。
    输入: [B, C, H, W]
    输出: 三个不同分辨率的特征:
           - l0: [B, initial_filters, H/4, W/4]
           - l1: [B, initial_filters, H/8, W/8]
           - l2: [B, initial_filters, H/16, W/16]
    """
    def __init__(self, filters, in_channels=1, debug_save_path=None, debug_save_freq=200):
        """
        初始化FeatureExtraction模块。
        参数:
            filters (int): 最终输出特征图 (l0, l1, l2) 的通道数，对应Keras原版中的filters参数。
            in_channels (int): 输入图像的通道数 (例如灰度图为1)。
            debug_save_path (str, optional): 保存l0调试图像的路径。默认为None。
            debug_save_freq (int, optional): 保存l0调试图像的频率 (每多少次forward调用)。默认为200。
        """
        super(FeatureExtraction, self).__init__() # 调用父类nn.Module的初始化方法
        
        initial_filters = filters # 例如，如果HMSMNet中filters=16，则initial_filters=16。这是l0,l1,l2的通道数
        backbone_filters = 2 * initial_filters # 特征提取主干部分的通道数，设为initial_filters的两倍，例如 2*16 = 32

        # 第一个卷积层，步长为2，进行2倍下采样。输出通道数为initial_filters (16)
        self.conv0_1 = conv2d_bn(initial_filters, 5, 2, 'same', 1, True, in_channels=in_channels)
        # 第二个卷积层，步长为2，再进行2倍下采样 (总共4倍下采样)。
        # 输入通道为initial_filters (16)，输出通道为backbone_filters (32)
        self.conv0_2 = conv2d_bn(backbone_filters, 5, 2, 'same', 1, True, in_channels=initial_filters)
        
        # 一系列堆叠的BasicBlock，用于深度特征提取。这些块都工作在backbone_filters (32) 通道上。
        # conv1_0: 4个BasicBlock，膨胀率为1
        self.conv1_0 = make_blocks(backbone_filters, 1, 4, current_in_channels=backbone_filters)
        # conv1_1: 2个BasicBlock，膨胀率为2
        self.conv1_1 = make_blocks(backbone_filters, 2, 2, current_in_channels=backbone_filters)
        # conv1_2: 2个BasicBlock，膨胀率为4
        self.conv1_2 = make_blocks(backbone_filters, 4, 2, current_in_channels=backbone_filters)
        # conv1_3: 2个BasicBlock，膨胀率为1
        self.conv1_3 = make_blocks(backbone_filters, 1, 2, current_in_channels=backbone_filters)
        # 经过conv1_3后，特征图的通道数仍然是backbone_filters (32)

        # 多尺度分支，通过平均池化和1x1卷积得到不同尺度的输出特征。
        # 输出通道数都调整回initial_filters (16)。
        # branch0: 对应1/4原始分辨率的特征 (l0)
        self.branch0 = avg_pool(pool_size=1, output_filters=initial_filters, input_filters=backbone_filters)
        # branch1: 对应1/8原始分辨率的特征 (l1)
        self.branch1 = avg_pool(pool_size=2, output_filters=initial_filters, input_filters=backbone_filters)
        # branch2: 对应1/16原始分辨率的特征 (l2)
        self.branch2 = avg_pool(pool_size=4, output_filters=initial_filters, input_filters=backbone_filters)


    def forward(self, x): # 定义前向传播逻辑
        x = self.conv0_1(x) # 通过第一个卷积层 (H/2, W/2, initial_filters)
        x = self.conv0_2(x) # 通过第二个卷积层 (H/4, W/4, backbone_filters)

        x = self.conv1_0(x) # 通过第一组残差块
        x = self.conv1_1(x) # 通过第二组残差块
        x = self.conv1_2(x) # 通过第三组残差块
        x = self.conv1_3(x) # 通过第四组残差块。此时x的通道数为backbone_filters (32)

        x0 = self.branch0(x) # 从主干特征生成l0 (H/4, W/4, initial_filters)
        x1 = self.branch1(x) # 从主干特征生成l1 (H/8, W/8, initial_filters)
        x2 = self.branch2(x) # 从主干特征生成l2 (H/16, W/16, initial_filters)
        
        return [x0, x1, x2] # 返回三个尺度的特征图列表