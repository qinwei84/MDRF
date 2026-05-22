import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式API

class Estimation(nn.Module):
    """
    视差估计模块，输入 cost volume 通道数为64（拼接特征，filters=16）
    输入: [B, 64, D, H, W]
    输出: [B, H, W, 1]
    """
    def __init__(self, min_disp=-112, max_disp=16, in_channels=64):
        super(Estimation, self).__init__()  # 父类初始化
        self.min_disp = int(min_disp)  # 最小视差
        self.max_disp = int(max_disp)  # 最大视差 
        self.conv = nn.Conv3d(
            in_channels=in_channels,  # 手动指定为cost volume通道数
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        # x: [B, 64, D, H, W]
        original_dtype = x.dtype  # 保存原始数据类型
        x = x.to(torch.float32)  # 转换为float32，避免精度问题
        x = self.conv(x)          # 3D卷积降到1通道，[B, 1, D, H, W]
        # print(f"DEBUG: Estimation input to softmax (x) min: {x.min().item()}, max: {x.max().item()}, mean: {x.mean().item()}")
        # if torch.isnan(x).any() or torch.isinf(x).any():
        #     print(f"!!! NaN/Inf detected in input to softmax (x) in Estimation module !!!")
            # 如果在这里发现问题，说明是 self.conv 导致了NaN/Inf
        x = x.squeeze(1)          # 去掉通道维，变为 [B, D, H, W]
        x = x.permute(0, 2, 3, 1) # 调整为 [B, H, W, D]
        # 断言最后一维等于视差区间
        assert x.shape[-1] == self.max_disp - self.min_disp

        # 构造视差候选值，shape: [D]
        candidates = torch.linspace(
            float(self.min_disp), float(self.max_disp) - 1.0,
            self.max_disp - self.min_disp, device=x.device
        )
        # 对最后一维D做softmax归一化，得到概率分布
        probabilities = F.softmax(-1.0 * x, dim=-1)  # [B, H, W, D]
        # 概率加权求和得到视差预测，输出[B, H, W, 1]
        disparities = torch.sum(candidates * probabilities, dim=-1, keepdim=True)
        return disparities.to(original_dtype)  # 返回视差图 [B, H, W, 1]