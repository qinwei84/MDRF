import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mdrf import MDRFConv

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

# ================= 特征提取 (将 SPP 替换为您的 MDRF) =================
class feature_extraction_mdrf_fusion(nn.Module):
    def __init__(self, in_channels=1):
        super(feature_extraction_mdrf_fusion, self).__init__()
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

        # 1. 保留 SPP 的“全局池化”分支，专治大面积无纹理！
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # 真正的全局视野
            convbn(128, 32, 1, 1, 0, 1), 
            nn.ReLU(inplace=True)
        )

        # 2. 您的 MDRF 模块！把膨胀率稍微开大一点，覆盖更合理的局部到中等范围
        self.mdrf_module = nn.Sequential(
            MDRFConv(in_channels=128, out_channels=64, kernel_size=3, scales=[1, 4, 8, 12]),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 3. 融合层：MDRF(64) + 全局(32) + raw(64) + skip(128) = 288 通道
        self.lastconv = nn.Sequential(
            convbn(288, 128, 3, 1, 1, 1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(128, 32, kernel_size=1, bias=False)
        )

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

        # 全局上下文分支
        out_global = self.global_branch(output_skip)
        out_global = F.interpolate(out_global, (output_skip.size(2), output_skip.size(3)), mode='bilinear', align_corners=False)

        # MDRF 动态多尺度分支
        out_mdrf = self.mdrf_module(output_skip)
        
        # 拼接：低层特征 + 高层特征 + MDRF精细多尺度 + 全局瞎猜兜底
        output_feature = torch.cat((output_raw, output_skip, out_mdrf, out_global), 1)
        output_feature = self.lastconv(output_feature)
        return output_feature
    
    def __init__(self, in_channels=1):
        super(feature_extraction_mdrf_fusion, self).__init__()
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

        # ====== [重头戏：用 MDRF 替换原版的 SPP] ======
        # 原版的 SPP 使用了 4 个不同尺度的大 Kernel AvgPool
        # 现在我们使用您的多尺度动态感受野 (MDRF)，让网络自己学着融合不同尺度的语义！
        # scales=[1, 3, 6, 9] 对应膨胀卷积提取不同大小的感受野
        self.mdrf_module = nn.Sequential(
            MDRFConv(in_channels=128, out_channels=128, kernel_size=3, scales=[1, 3, 6, 9]),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # 融合层：MDRF 的输出(128) + raw特征(64) + skip特征(128) = 320 通道
        self.lastconv = nn.Sequential(
            convbn(320, 128, 3, 1, 1, 1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(128, 32, kernel_size=1, bias=False)
        )

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

        # ====== [应用您的 MDRF] ======
        output_mdrf = self.mdrf_module(output_skip)
        
        # 将原始特征(低层)和跳积特征(高层)与 MDRF提取出的多尺度增强特征拼接
        output_feature = torch.cat((output_raw, output_skip, output_mdrf), 1)
        output_feature = self.lastconv(output_feature)
        return output_feature

# ================= 3D CNN Hourglass (保持不变，以控制变量) =================
class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()
        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1), nn.ReLU(inplace=True))
        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)
        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1), nn.ReLU(inplace=True))
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

# ================= 核心网络：PSMNet_MDRF =================
class PSMNet(nn.Module):
    def __init__(self, maxdisp, min_disp=0, in_channels=1):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.min_disp = min_disp
        self.disp_range = maxdisp - min_disp
        
        # 换成我们刚刚魔改的 MDRF 特征提取器
        self.feature_extraction = feature_extraction_mdrf_fusion(in_channels=in_channels)

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = hourglass(32)
        self.dres3 = hourglass(32)
        self.dres4 = hourglass(32)

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