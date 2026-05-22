import torch
import torch.nn as nn

def initialize_model_weights_and_bn(m):
    """
    自定义模型初始化函数，用于 model.apply()。
    - 对卷积层和线性层使用 Kaiming Normal 初始化权重，偏置初始化为0。
    - 对 BatchNorm 层设置 eps=0.001, momentum=0.01，并将gamma(weight)初始化为1，beta(bias)初始化为0。
    """
    classname = m.__class__.__name__
    
    # --- 处理卷积层 (Conv1d, Conv2d, Conv3d) ---
    if classname.find('Conv') != -1:
        try:
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
            # print(f"Kaiming Normal initialized Conv layer: {classname}")
        except AttributeError: # 捕获那些可能没有 weight 或 bias 的特殊 'Conv' 模块 (尽管不太常见)
            # print(f"Skipping initialization for {classname} (no weight/bias or other issue)")
            pass

    # --- 处理线性层 (Linear) ---
    elif classname.find('Linear') != -1:
        try:
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
            # print(f"Kaiming Normal initialized Linear layer: {classname}")
        except AttributeError:
            # print(f"Skipping initialization for {classname} (no weight/bias or other issue)")
            pass
            
    # --- 处理 BatchNorm 层 (BatchNorm1d, BatchNorm2d, BatchNorm3d) ---
    elif classname.find('BatchNorm') != -1:
        try:
            # 设置 eps 和 momentum
            m.eps = 0.001
            # TensorFlow Keras BatchNormalization 的 momentum 参数 (默认为 0.99) 
            # 通常定义为: new_running_average = momentum * old_running_average + (1 - momentum) * current_batch_average
            # PyTorch BatchNorm 的 momentum 参数
            # 定义为: new_running_average = (1 - momentum) * old_running_average + momentum * current_batch_average
            # 因此，要匹配 Keras momentum = 0.99 (m_tf = 0.99)，
            # PyTorch momentum (m_pt) 应该设置为 1 - m_tf = 1 - 0.99 = 0.01.
            m.momentum = 0.01
            
            # 初始化 gamma (weight) 为 1, beta (bias) 为 0
            if m.weight is not None: # 检查是否存在 weight (gamma)
                 nn.init.constant_(m.weight.data, 1)
            if m.bias is not None: # 检查是否存在 bias (beta)
                 nn.init.constant_(m.bias.data, 0)
            # print(f"Configured BatchNorm layer: {classname} with eps={m.eps}, momentum={m.momentum}")
        except AttributeError:
            # print(f"Skipping configuration for {classname} (no eps/momentum/weight/bias or other issue)")
            pass
