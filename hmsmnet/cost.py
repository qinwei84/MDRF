import torch  # 导入PyTorch主库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式API

class CostConcatenation(nn.Module):
    """
    体积代价（Cost Volume）拼接实现，输入为左右特征图，输出为代价体积
    输入: list of [B, H, W, C]，左右特征图
    输出: [B, D, H, W, 2C]，D为视差范围
    """
    def __init__(self, min_disp=-112, max_disp=16):
        super(CostConcatenation, self).__init__()  # 父类初始化
        self.min_disp = int(min_disp)  # 最小视差
        self.max_disp = int(max_disp)  # 最大视差

    def forward(self, inputs):
        assert len(inputs) == 2  # 检查输入为2个
        left, right = inputs  # 拆包为左、右特征
        B, H, W, C = left.shape  # 获取shape
        cost_volume = []  # 存放每个视差下的拼接结果
        for i in range(self.min_disp, self.max_disp):  # 遍历视差
            if i < 0:
                # 负视差，右图向右移
                concat = torch.cat([left[:, :, :i, :], right[:, :, -i:, :]], dim=-1)  # 拼接特征
                pad_h = H - concat.shape[1]  # 高度需pad数
                pad_w = W - concat.shape[2]  # 宽度需pad数
                # F.pad参数为 (最后一维左, 最后一维右, w左, w右, h上, h下)
                concat = F.pad(concat, (0, 0, 0, pad_w, 0, pad_h), "constant", 0)  # pad到目标大小
            elif i > 0:
                # 正视差，右图向左移
                concat = torch.cat([left[:, :, i:, :], right[:, :, :-i, :]], dim=-1)
                pad_h = H - concat.shape[1] # Should be 0
                pad_w = W - concat.shape[2] # pad_w is i
                # To pad pad_w to the LEFT of the width dimension:
                # F.pad arguments for 4D tensor (B,H,W_eff,C_concat) is (pad_C_L, pad_C_R, pad_W_L, pad_W_R, pad_H_L, pad_H_R)
                concat = F.pad(concat, (0, 0, pad_w, 0, 0, pad_h), "constant", 0) # Pads pad_w to the LEFT
            else:
                # i == 0，不偏移
                concat = torch.cat([left, right], dim=-1)
            cost_volume.append(concat)  # 结果加入列表
        cost_volume = torch.stack(cost_volume, dim=1)  # 堆叠为5D tensor
        return cost_volume  # 返回体积代价

class CostDifference(nn.Module):
    """
    体积代价（Cost Volume）差分实现，输入为左右特征图，输出为代价体积
    输入: list of [B, H, W, C]，左右特征图
    输出: [B, D, H, W, C]，D为视差范围
    """
    def __init__(self, min_disp=-112, max_disp=16):
        super(CostDifference, self).__init__()  # 父类初始化
        self.min_disp = int(min_disp)  # 最小视差
        self.max_disp = int(max_disp)  # 最大视差

    def forward(self, inputs):
        assert len(inputs) == 2  # 检查输入为2个
        left, right = inputs  # 左右特征
        B, H, W, C = left.shape  # 获取shape
        cost_volume = []  # 存放每个视差下的差分结果
        for i in range(self.min_disp, self.max_disp):  # 遍历视差
            if i < 0:
                # 负视差
                diff = left[:, :, :i, :] - right[:, :, -i:, :]
                pad_h = H - diff.shape[1]
                pad_w = W - diff.shape[2]
                diff = F.pad(diff, (0, 0, 0, pad_w, 0, pad_h), "constant", 0)
            elif i > 0:
                # 正视差
                diff = left[:, :, i:, :] - right[:, :, :-i, :]
                pad_h = H - diff.shape[1]
                pad_w = W - diff.shape[2]
                diff = F.pad(diff, (0, 0, 0, pad_w, 0, pad_h), "constant", 0)
            else:
                # i == 0，不偏移
                diff = left - right
            cost_volume.append(diff)  # 结果加入列表
        cost_volume = torch.stack(cost_volume, dim=1)  # 堆叠为5D tensor
        return cost_volume  # 返回体积代价