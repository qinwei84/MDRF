import torch  # PyTorch主库
import torch.nn as nn  # 神经网络模块
from .feature import FeatureExtraction  # 导入特征提取网络
from .cost import CostConcatenation  # 导入代价体volume构建
from .aggregation import Hourglass, FeatureFusion  # 导入体积聚合模块
from .computation import Estimation  # 导入视差估计模块
from .refinement import Refinement  # 导入细化模块

class HMSMNet(nn.Module):  # 定义HMSMNet主网络类，继承nn.Module
    """
    HMSMNet整体组网，分为特征提取、代价体积构建、体积聚合、视差估计和细化
    """
    def __init__(self, height, width, channel, min_disp, max_disp):  # 初始化方法，输入图像尺寸和视差范围
        super(HMSMNet, self).__init__()  # 父类初始化
        self.height = height  # 输入图像高度
        self.width = width    # 输入图像宽度
        self.channel = channel  # 输入通道数
        self.min_disp = int(min_disp)  # 最小视差
        self.max_disp = int(max_disp)  # 最大视差
        
        # 特征提取模块，filters=16，输入通道为channel
        self.feature_extraction = FeatureExtraction(filters=16, in_channels=channel)
        
        # 代价体积构建模块，3个尺度，min/max_disp按比例缩放
        self.cost0 = CostConcatenation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        self.cost1 = CostConcatenation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        self.cost2 = CostConcatenation(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16)
        
        # 体积聚合模块，输入通道=2*filters=32
        self.hourglass0 = Hourglass(filters=16, in_channels=32)
        self.hourglass1 = Hourglass(filters=16, in_channels=32)
        self.hourglass2 = Hourglass(filters=16, in_channels=32)
        
        # 视差估计模块，输入通道=32，min/max_disp需与costX一致
        channels_for_fusion = 16 # 这个值应该与 hourglass0,1,2 的输出通道数一致 
        self.estimator2 = Estimation(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16, in_channels=16)
        self.fusion1 = FeatureFusion(channels=channels_for_fusion)  # 体积特征融合
        self.hourglass3 = Hourglass(filters=16, in_channels=channels_for_fusion)
        self.estimator1 = Estimation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8, in_channels=16)
        self.fusion2 = FeatureFusion(channels=channels_for_fusion)  # 体积特征融合
        self.hourglass4 = Hourglass(filters=16, in_channels=channels_for_fusion)
        self.estimator0 = Estimation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4, in_channels=16)

        # 细化模块，filters可适当加大
        self.refiner = Refinement(filters=32, img_channels=channel)
        
        # # --- 新增：用于调试保存 l0 的属性 ---
        # self.debug_l0_save_path = "debug_outputs/hmsmnet_l0_features"
        # self.debug_l0_save_freq = 100
        # self._forward_call_count_hmsm = 0 # 内部计数器，避免与 FeatureExtraction 中的计数器混淆
        # if self.debug_l0_save_path:
        #     os.makedirs(self.debug_l0_save_path, exist_ok=True)
        #     print(f"HMSMNet: Debug images for l0 will be saved to '{self.debug_l0_save_path}' every {self.debug_l0_save_freq} calls.")
        # # --- 结束新增 ---
        
    def forward(self, left_image, right_image, gx, gy):  # 前向传播
        """
        输入:
            left_image:  [B, C, H, W] 左图
            right_image: [B, C, H, W] 右图
            gx:          [B, 1, H, W] x方向梯度
            gy:          [B, 1, H, W] y方向梯度
        输出:
            disparity2, disparity1, disparity0, final_disp
        """
        
        # 特征提取（左、右图多尺度特征）
        l0, l1, l2 = self.feature_extraction(left_image)   # 左图三尺度特征
        r0, r1, r2 = self.feature_extraction(right_image)  # 右图三尺度特征 
        
        # 工具函数：NCHW转NHWC（代价体积构建需要）
        def to_nhwc(x): 
            return x.permute(0, 2, 3, 1) if x.shape[1] != x.shape[-1] else x
        
        # 工具函数：NHWC转NCHW（如果需要）
        def to_nchw(x): 
            return x.permute(0, 3, 1, 2) if x.shape[1] != x.shape[-1] else x
        
        # 代价体积构建，[B, H, W, C] -> [B, D, H, W, 2C]
        cost_volume0 = self.cost0([to_nhwc(l0), to_nhwc(r0)])  # 尺度0
        cost_volume1 = self.cost1([to_nhwc(l1), to_nhwc(r1)])  # 尺度1
        cost_volume2 = self.cost2([to_nhwc(l2), to_nhwc(r2)])  # 尺度2
        

        # 工具函数：NHWCD转NCDHW（3D卷积输入）
        def to_ncdhw(x): 
            return x.permute(0, 4, 1, 2, 3)  # [B, D, H, W, C] -> [B, C, D, H, W]
        
        # 体积聚合/3D卷积
        agg_cost0 = self.hourglass0(to_ncdhw(cost_volume0))  # 尺度0体积聚合
        agg_cost1 = self.hourglass1(to_ncdhw(cost_volume1))  # 尺度1体积聚合
        agg_cost2 = self.hourglass2(to_ncdhw(cost_volume2))  # 尺度2体积聚合
        # 视差估计2（最小分辨率），输出[ B, H2, W2, 1 ]
        disparity2 = self.estimator2(agg_cost2)  # 输入为[B, 32, D2, H2, W2]
        
        # 聚合&估计1（中等分辨率）
        fusion_cost1 = self.fusion1([agg_cost2, agg_cost1])   # 融合两个体积
        agg_fusion_cost1 = self.hourglass3(fusion_cost1)      # 聚合
        disparity1 = self.estimator1(agg_fusion_cost1)        # 估计视差

        # 聚合&估计0（最高分辨率）
        fusion_cost2 = self.fusion2([agg_fusion_cost1, agg_cost0])  # 融合
        agg_fusion_cost2 = self.hourglass4(fusion_cost2)            # 聚合
        # DEBUG
        # if torch.isnan(agg_fusion_cost2).any() or torch.isinf(agg_fusion_cost2).any():
        #     raise Exception("NaN/Inf in agg_fusion_cost2")
        disparity0 = self.estimator0(agg_fusion_cost2)              # 估计视差
        # DEBUG

        # 细化（补充细节，提升精度）
        disp0_nchw = disparity0.permute(0, 3, 1, 2)         # [B, H0, W0, 1] -> [B, 1, H0, W0]
        final_disp = self.refiner([disp0_nchw, left_image, gx, gy])  # 输入多模态细化
        #输出disparity2, disparity1, disparity0, final_disp的shape
        #print(f"Output shapes: disparity2={disparity2.shape}, disparity1={disparity1.shape}, "
        #      f"disparity0={disparity0.shape}, final_disp={final_disp.shape}")
        
        return disparity2, disparity1, disparity0, final_disp  # 返回所有尺度和最终视差图

    # 推理/预测函数可根据需要补充，需加载权重、读取图片、预处理等
    # def predict(self, left_dir, right_dir, output_dir, weights):
    #     pass  # 可按需要补充推理流程
