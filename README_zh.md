# MDRF 公开发布版使用说明

这个仓库是 **MDRF（Multi-scale Dynamic Receptive Field）模块** 的公开发布包。这里的 HMSMNet 和 PSMNet 只是验证 MDRF 有效性和迁移性的宿主框架，不是本文真正要强调的开源主体。

## 当前包里有什么

- `mdrf/`：正式 `MDRFConv` / `MDRFConv3D` 实现。
- `hmsmnet/`：用于验证 MDRF 的 HMSMNet baseline/宿主框架。
- `psmnet/`：用于验证 MDRF 可迁移性的 PSMNet baseline/宿主框架。
- `experiments/`：不同阶段的 MDRF 插入骨架，包括 feature、refinement、3D hourglass、cost-volume fusion、all-stages，以及 PSMNet 的 SPP/3D 聚合迁移实验。
- `weights/`：用于放官方权重或下载说明的预留目录。
- `splits/`：WHU-Stereo 和 US3D 的实验 split 文件。
- `environment.yml`：本项目原始 conda 环境。

split 文件已经放在：

```text
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

## MDRF / MDRF3D 实现

正式实现位于：

```text
mdrf/mdrf_conv.py
```

该文件导出两个类：

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

## split 与权重位置

split 已经按固定目录提供；权重可以后续放在对应目录：

```text
weights/hmsmnet/          # HMSMNet 宿主框架的 MDRF 权重
weights/psmnet/           # PSMNet 宿主框架的 MDRF 权重
splits/whu-stereo/        # WHU-Stereo 官方 split 对应的文件列表
splits/us3d/              # US3D，按 MaskCRNet/Rao2024 方案整理的 split 文件列表
```

WHU-Stereo 每行是 `left right disp` 三列相对路径；US3D 每行是 `left right disp label` 四列相对路径。权重命名建议见 `weights/README.md`。

## 使用 baseline 推理

HMSMNet 宿主框架：

```bash
python scripts/infer_hmsmnet_pair.py \
  --left path/to/left.tif \
  --right path/to/right.tif \
  --checkpoint weights/hmsmnet/hmsmnet_baseline_whu_stereo_epoch37.pth \
  --output outputs/hmsmnet_disp.tif \
  --min-disp -128 \
  --max-disp 64
```

PSMNet 宿主框架：

```bash
python scripts/infer_psmnet_pair.py \
  --left path/to/left.tif \
  --right path/to/right.tif \
  --checkpoint weights/psmnet/psmnet_baseline_whu_stereo.pth \
  --output outputs/psmnet_disp.tif \
  --min-disp -128 \
  --max-disp 64
```

当前源码包不强制附带权重。

## 开源协议

本项目采用 MIT License。具体文本见 [LICENSE](LICENSE)。
