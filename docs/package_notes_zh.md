# 本地整理说明

这个目录是 MDRF 论文接收前的 GitHub public preview 版本，只做本地整理，不包含任何 git 上传操作。HMSMNet 和 PSMNet 在这里是验证 MDRF 的宿主框架，不是仓库主角。

## 当前可以公开的内容

- `mdrf/`：MDRF/MDRF3D 的统一占位位置。
- `hmsmnet/`：HMSMNet 宿主框架。
- `psmnet/`：PSMNet 宿主框架。
- `experiments/`：不同阶段的 MDRF 实验骨架，包括 feature、refinement、3D hourglass、cost-volume fusion、all-stages，以及 PSMNet 的 SPP/3D 聚合迁移实验。
- `scripts/`：只做单对影像推理的示例脚本，需要用户自行提供权重。
- `weights/`：论文接收后放官方权重。
- `splits/`：论文接收后放 WHU-Stereo 和 US3D 的 split 文件。

## 当前刻意不放的内容

- MDRFConv / MDRFConv3D 的真实实现。
- 训练函数、训练入口、optimizer/scheduler 细节。
- 数据集 split 文件。
- checkpoint、TensorBoard 日志、预测 tiff、实验中间文件。

论文接收后，建议按以下位置补充公开资源：

```text
weights/hmsmnet/
weights/psmnet/
splits/whu-stereo/
splits/us3d/
```
- 本机绝对路径和私有实验配置。

## 后续正式开源时怎么补

正式接收后，把真实实现放到：

```text
mdrf/mdrf_conv.py
```

该文件需要提供 `MDRFConv` 和 `MDRFConv3D` 两个类。所有实验骨架都已经统一从 `mdrf` 包导入，因此只需要替换这一个文件，不需要逐个实验目录修改。
