# MDRF Public Preview

This repository is a preliminary public preview for the **Multi-scale Dynamic Receptive Field (MDRF)** module proposed in our stereo matching work. HMSMNet and PSMNet are included as host backbones for verification, transfer, and ablation experiments; they are not the main contribution of this repository.

## Status

This repository is a pre-acceptance public preview of the MDRF implementation framework. The current release provides the host backbones, stage-level MDRF insertion skeletons, stage-specific placement configurations, dilation-rate settings, environment setup, and configuration files needed to inspect the integration design before acceptance. Following the staged release plan, the official MDRFConv/MDRFConv3D operator source files, trained checkpoints, and dataset split files are temporarily withheld and will be added in the final public release after acceptance.

## Included

- `mdrf/`: placeholder location for the official MDRF operators.
- `hmsmnet/`: HMSMNet host backbone used to evaluate MDRF in different stereo stages.
- `psmnet/`: PSMNet host backbone used to test MDRF transferability.
- `experiments/`: staged MDRF insertion skeletons for feature extraction, refinement, 3D hourglass aggregation, cost-volume fusion, all-stage insertion, and PSMNet transfer experiments.
- `scripts/`: single-pair inference helpers for user-provided checkpoints.
- `weights/`: reserved release location for official checkpoints after paper acceptance.
- `splits/`: reserved release location for WHU-Stereo and US3D split files after paper acceptance.
- `environment.yml`: the original conda environment used in this project.
- `README_zh.md`: Chinese usage notes.

## Not Included in This Preview

- Official `MDRFConv` and `MDRFConv3D` source code.
- dataset split files.
- Pretrained weights.

Reserved folders are already provided for post-acceptance release assets:

```text
weights/hmsmnet/
weights/psmnet/
splits/whu-stereo/
splits/us3d/
```

## Installation

The recommended setup is the original conda environment used for this project:

```bash
conda env create -f environment.yml
conda activate hsssmnet
```

For lightweight CPU-side inspection only, `requirements.txt` is kept as a minimal pip fallback. A CUDA-enabled PyTorch environment is recommended for full-resolution stereo inference.

## Where to Add MDRF / MDRF3D

After the formal release, place the official operator implementation here:

```text
mdrf/mdrf_conv.py
```

The file should export both classes:

```python
class MDRFConv(nn.Module):
    ...

class MDRFConv3D(nn.Module):
    ...
```

All staged experiment files already import from the central package:

```python
from mdrf import MDRFConv
from mdrf import MDRFConv3D
```

Therefore, replacing `mdrf/mdrf_conv.py` is enough; the experiment folders do not need to be edited one by one.

## Post-Acceptance Assets

After paper acceptance, the official weights and split files should be uploaded to the reserved folders:

```text
weights/hmsmnet/          # HMSMNet-host MDRF checkpoints
weights/psmnet/           # PSMNet-host MDRF checkpoints
splits/whu-stereo/        # WHU-Stereo official split files used in this project
splits/us3d/              # US3D split files following the MaskCRNet/Rao2024 protocol
```

Suggested split filenames are `train.txt`, `val.txt`, and `test.txt`, with one sample identifier per line. Suggested weight filenames are documented in `weights/README.md`.

## Example Inference

HMSMNet host baseline:

```bash
python scripts/infer_hmsmnet_pair.py \
  --left path/to/left.tif \
  --right path/to/right.tif \
  --checkpoint path/to/hmsmnet_baseline.pth \
  --output outputs/hmsmnet_disp.tif \
  --min-disp -128 \
  --max-disp 64
```

PSMNet host baseline:

```bash
python scripts/infer_psmnet_pair.py \
  --left path/to/left.tif \
  --right path/to/right.tif \
  --checkpoint path/to/psmnet_baseline.pth \
  --output outputs/psmnet_disp.tif \
  --min-disp -128 \
  --max-disp 64
```

The scripts only define inference behavior and require externally supplied checkpoints. No weights are bundled in this preview.

## Experimental Skeletons

The staged variants under `experiments/` show where MDRF/MDRF3D is inserted for ablation and transfer experiments. These files document the integration points while keeping the core operator private until the formal release.

## Citation

Citation information will be added after the paper is accepted.

## 中文使用说明

中文说明见 [README_zh.md](README_zh.md)。简要来说：当前公开包以 MDRF 为主，HMSMNet/PSMNet 只是验证用宿主框架；正式开源时将 `mdrf/mdrf_conv.py` 替换为包含 `MDRFConv` 和 `MDRFConv3D` 的真实实现即可。
