import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 只导入您的 3D 动态感受野模块
from mdrf import MDRFConv3D

# ================= 工具与基础模块 =================
def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                  padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes)
    )

def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
        nn.BatchNorm3d(out_planes)
    )

# ====== 新增：MDRF 3D 的基础包装块 (带 BN 归一化) ======
def mdrf_convbn_3d(in_planes, out_planes, kernel_size, stride, scales):
    return nn.Sequential(
        MDRFConv3D(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride, scales=scales),
        nn.BatchNorm3d(out_planes)  # 紧跟 BN 层，彻底解决方差偏移
    )

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation), nn.ReLU(inplace=True))
        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        out += identity
        return F.relu(out, inplace=True)

# ================= 2D 特征提取 (完全恢复原版 SPP，没有任何改动) =================
class feature_extraction(nn.Module):
    def __init__(self, in_channels=1):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(
            convbn(in_channels, 32, 3, 2, 1, 1), nn.ReLU(inplace=True),
            convbn(32, 32, 3, 1, 1, 1), nn.ReLU(inplace=True),
            convbn(32, 32, 3, 1, 1, 1), nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1) 
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)), convbn(128, 32, 1, 1, 0, 1), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)), convbn(128, 32, 1, 1, 0, 1), nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)), convbn(128, 32, 1, 1, 0, 1), nn.ReLU(inplace=True))
        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)), convbn(128, 32, 1, 1, 0, 1), nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(128, 32, kernel_size=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.interpolate(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=False)
        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.interpolate(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=False)
        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.interpolate(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=False)
        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.interpolate(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=False)

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)
        return output_feature

# ================= 3D CNN Hourglass (融入 MDRF3D 进行代价体聚合) =================
class hourglass_mdrf(nn.Module):
    def __init__(self, inplanes):
        super(hourglass_mdrf, self).__init__()
        # conv1: 空间与视差维度的降采样
        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1), nn.ReLU(inplace=True))
        
        # ====== [3D 即插即用位置 1] ======
        self.conv2 = mdrf_convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, scales=[1, 2, 3])
        
        # conv3: 再次降采样到 1/4 尺度
        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1), nn.ReLU(inplace=True))
        
        # ====== [3D 即插即用位置 2] ======
        self.conv4 = nn.Sequential(
            mdrf_convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, scales=[1, 2, 3]), 
            nn.ReLU(inplace=True)
        )
        
        # 上采样反卷积
        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False), nn.BatchNorm3d(inplanes*2))
        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False), nn.BatchNorm3d(inplanes))

    def forward(self, x, presqu, postsqu):
        out = self.conv1(x)
        pre = self.conv2(out)
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)
        out = self.conv3(pre)
        out = self.conv4(out)
        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)
        out = self.conv6(post)
        return out, pre, post

# ================= 视差回归 =================
class disparityregression(nn.Module):
    def __init__(self, min_disp, max_disp):
        super(disparityregression, self).__init__()
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.disp_range = max_disp - min_disp
        self.disp = torch.Tensor(np.reshape(np.array(range(min_disp, max_disp)), [1, self.disp_range, 1, 1]))

    def forward(self, x):
        self.disp = self.disp.to(x.device)
        out = torch.sum(x * self.disp, 1, keepdim=True)
        return out

# ================= 核心网络：仅 3D 聚合阶段 MDRF 增强版 =================
class PSMNet(nn.Module):
    def __init__(self, maxdisp, min_disp=0, in_channels=1):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.min_disp = min_disp
        self.disp_range = maxdisp - min_disp
        
        # 1. 2D 特征提取���全恢复 Baseline 原貌！
        self.feature_extraction = feature_extraction(in_channels=in_channels)

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d(32, 32, 3, 1, 1)) 

        # 2. 使用带有 3D MDRF 的 Hourglass
        self.dres2 = hourglass_mdrf(32)
        self.dres3 = hourglass_mdrf(32)
        self.dres4 = hourglass_mdrf(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, left, right):
        refimg_fea = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)

        D = self.disp_range // 4
        cost = torch.autograd.Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, D, refimg_fea.size()[2], refimg_fea.size()[3]).zero_()).to(left.device)

        for i in range(D):
            disp = i + (self.min_disp // 4)
            if disp < 0:
                cost[:, :refimg_fea.size()[1], i, :, :disp] = refimg_fea[:, :, :, :disp]
                cost[:, refimg_fea.size()[1]:, i, :, :disp] = targetimg_fea[:, :, :, -disp:]
            elif disp > 0:
                cost[:, :refimg_fea.size()[1], i, :, disp:] = refimg_fea[:, :, :, disp:]
                cost[:, refimg_fea.size()[1]:, i, :, disp:] = targetimg_fea[:, :, :, :-disp]
            else:
                cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea
        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        
        # 通过三个级联的 3D MDRF Hourglass 网络
        out1, pre1, post1 = self.dres2(cost0, None, None) 
        out1 = out1 + cost0
        out2, pre2, post2 = self.dres3(out1, pre1, post1) 
        out2 = out2 + cost0
        out3, pre3, post3 = self.dres4(out2, pre1, post2) 
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        cost1 = F.interpolate(cost1, [self.disp_range, left.size()[2], left.size()[3]], mode='trilinear', align_corners=False).squeeze(1)
        cost2 = F.interpolate(cost2, [self.disp_range, left.size()[2], left.size()[3]], mode='trilinear', align_corners=False).squeeze(1)
        cost3 = F.interpolate(cost3, [self.disp_range, left.size()[2], left.size()[3]], mode='trilinear', align_corners=False).squeeze(1)

        pred1 = disparityregression(self.min_disp, self.maxdisp)(F.softmax(cost1, dim=1))
        pred2 = disparityregression(self.min_disp, self.maxdisp)(F.softmax(cost2, dim=1))
        pred3 = disparityregression(self.min_disp, self.maxdisp)(F.softmax(cost3, dim=1))

        if self.training:
            return pred1, pred2, pred3
        else:
            return pred3