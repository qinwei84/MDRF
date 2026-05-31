# MDRF 公开预览版使用说明

这个仓库是 **MDRF（Multi-scale Dynamic Receptive Field）模块** 的论文接收前公开预览包。这里的 HMSMNet 和 PSMNet 只是验证 MDRF 有效性和迁移性的宿主框架，不是本文真正要强调的开源主体。

## 当前包里有什么

- `mdrf/`：MDRF/MDRF3D 的统一放置位置，目前只有占位文件。
- `hmsmnet/`：用于验证 MDRF 的 HMSMNet baseline/宿主框架。
- `psmnet/`：用于验证 MDRF 可迁移性的 PSMNet baseline/宿主框架。
- `experiments/`：不同阶段的 MDRF 插入骨架，包括 feature、refinement、3D hourglass、cost-volume fusion、all-stages，以及 PSMNet 的 SPP/3D 聚合迁移实验。
- `weights/`：论文接收后用于放官方权重的预留目录。
- `splits/`：论文接收后用于放 WHU-Stereo 和 US3D split 的预留目录。
- `environment.yml`：本项目原始 conda 环境。

## 当前刻意不放什么

- 真实 `MDRFConv` / `MDRFConv3D` 实现。
- 数据集 split 文件。
- checkpoint。

不过目录口子已经留好，论文接收后建议放在：

```text
weights/hmsmnet/
weights/psmnet/
splits/whu-stereo/
splits/us3d/
```

## 安装环境

推荐直接使用 conda 环境：

```bash
conda env create -f environment.yml
conda activate hsssmnet
```

`requirements.txt` 只作为轻量查看代码时的 pip fallback，不建议用它替代完整 CUDA 环境。

## 正式版如何补 MDRF / MDRF3D

正式接收后，只需要替换这个文件：

```text
mdrf/mdrf_conv.py
```

真实文件需要导出两个类：

```python
class MDRFConv(nn.Module):
    ...

class MDRFConv3D(nn.Module):
    ...
```

所有实验骨架已经统一写成：

```python
from mdrf import MDRFConv
from mdrf import MDRFConv3D
```

所以后续只要维护 `mdrf/mdrf_conv.py` 这一个位置，不需要给每个实验目录分别复制 MDRF 文件。

## 论文接收后补充资源的位置

正式接收后，建议把权重和 split 放到固定目录：

```text
weights/hmsmnet/          # HMSMNet 宿主框架的 MDRF 权重
weights/psmnet/           # PSMNet 宿主框架的 MDRF 权重
splits/whu-stereo/        # WHU-Stereo 官方 split 对应的文件列表
splits/us3d/              # US3D，按 MaskCRNet/Rao2024 方案整理的 split 文件列表
```

split 文件建议使用 `train.txt`、`val.txt`、`test.txt`，每行一个样本标识。权重命名建议见 `weights/README.md`。

## 使用 baseline 推理

HMSMNet 宿主框架：

```bash
python scripts/infer_hmsmnet_pair.py \
  --left path/to/left.tif \
  --right path/to/right.tif \
  --checkpoint path/to/hmsmnet_baseline.pth \
  --output outputs/hmsmnet_disp.tif \
  --min-disp -128 \
  --max-disp 64
```

PSMNet 宿主框架：

```bash
python scripts/infer_psmnet_pair.py \
  --left path/to/left.tif \
  --right path/to/right.tif \
  --checkpoint path/to/psmnet_baseline.pth \
  --output outputs/psmnet_disp.tif \
  --min-disp -128 \
  --max-disp 64
```

当前预览包不附带权重。
