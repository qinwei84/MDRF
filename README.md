# MDRF Public Release

This repository provides the released implementation of the **Multi-scale Dynamic Receptive Field (MDRF)** module proposed in our stereo matching work. HMSMNet and PSMNet are included as host backbones for verification, transfer, and ablation experiments; they are not the main contribution of this repository.

## Status

The paper has been accepted, so this package now includes the official `MDRFConv` and `MDRFConv3D` implementation and the dataset split files used for the reported WHU-Stereo and US3D experiments. Training entry points, optimizer schedules, TensorBoard logs, generated prediction rasters, and large checkpoints are still kept outside the source package.

## Included

- `mdrf/`: official 2D and 3D MDRF operators.
- `hmsmnet/`: HMSMNet host backbone used to evaluate MDRF in different stereo stages.
- `psmnet/`: PSMNet host backbone used to test MDRF transferability.
- `experiments/`: staged MDRF insertion skeletons for feature extraction, refinement, 3D hourglass aggregation, cost-volume fusion, all-stage insertion, and PSMNet transfer experiments.
- `scripts/`: single-pair inference helpers for user-provided checkpoints.
- `weights/`: reserved location for official checkpoints or download notes.
- `splits/`: WHU-Stereo and US3D split files used in this project.
- `environment.yml`: the original conda environment used in this project.
- `README_zh.md`: Chinese usage notes.

## Not Included in This Source Package

- Training scripts, training loops, and optimizer schedules.
- Intermediate checkpoints, logs, and prediction outputs.
- Local machine paths or private experiment configuration.

The split folders use portable dataset-relative paths:

```text
splits/whu-stereo/train.txt
splits/whu-stereo/val.txt
splits/whu-stereo/test.txt
splits/us3d/train.txt
splits/us3d/val.txt
splits/us3d/test.txt
```

## Installation

The recommended setup is the original conda environment used for this project:

```bash
conda env create -f environment.yml
conda activate hsssmnet
```

For lightweight CPU-side inspection only, `requirements.txt` is kept as a minimal pip fallback. A CUDA-enabled PyTorch environment is recommended for full-resolution stereo inference.

## MDRF / MDRF3D Operators

The official operator implementation is provided here:

```text
mdrf/mdrf_conv.py
```

The file exports both classes:

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

## Dataset Splits and Weights

The split files are included under:

```text
splits/whu-stereo/        # WHU-Stereo official split files used in this project
splits/us3d/              # US3D split files following the MaskCRNet/Rao2024 protocol
```

Each line stores relative paths from the corresponding dataset root. WHU-Stereo rows contain `left right disp`; US3D rows contain `left right disp label`. Suggested weight filenames are documented in `weights/README.md`.

## Example Inference

HMSMNet host baseline:

```bash
python scripts/infer_hmsmnet_pair.py \
  --left path/to/left.tif \
  --right path/to/right.tif \
  --checkpoint weights/hmsmnet/hmsmnet_baseline_whu_stereo_epoch37.pth \
  --output outputs/hmsmnet_disp.tif \
  --min-disp -128 \
  --max-disp 64
```

PSMNet host baseline:

```bash
python scripts/infer_psmnet_pair.py \
  --left path/to/left.tif \
  --right path/to/right.tif \
  --checkpoint weights/psmnet/psmnet_baseline_whu_stereo.pth \
  --output outputs/psmnet_disp.tif \
  --min-disp -128 \
  --max-disp 64
```

The scripts can use the released checkpoints under `weights/` or user-supplied checkpoints with compatible architectures.

## Experimental Skeletons

The staged variants under `experiments/` show where MDRF/MDRF3D is inserted for ablation and transfer experiments.

## Citation

Citation information will be updated with the final bibliographic record.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

## 中文使用说明

中文说明见 [README_zh.md](README_zh.md)。
