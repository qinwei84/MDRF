# MDRF Public Preview Release Scope

## Purpose

This preview package is centered on the MDRF module. HMSMNet and PSMNet are included only as host backbones for ablation and transferability verification. The official MDRF/MDRF3D operator implementation is reserved for the formal release.

## Current Public Contents

- Central MDRF placeholder package under `mdrf/`.
- HMSMNet and PSMNet host backbone definitions.
- Staged MDRF insertion skeletons under `experiments/`.
- Minimal inference scripts for user-provided checkpoints.
- Conda environment file and lightweight pip fallback requirements.
- Reserved folders for post-acceptance weights and dataset splits: `weights/` and `splits/`.

## Reserved for Formal Release

- Official `MDRFConv` and `MDRFConv3D` implementation in `mdrf/mdrf_conv.py`.
- Training entry points, optimizer details, and schedule code.
- Dataset split files and scripts that reproduce experimental splits.
- Official weights to be uploaded after acceptance under `weights/hmsmnet/` and `weights/psmnet/`.
- WHU-Stereo and US3D split files to be uploaded after acceptance under `splits/whu-stereo/` and `splits/us3d/`.
- Intermediate checkpoints, prediction rasters, TensorBoard logs, and local experiment outputs.

## Pre-Push Checklist

Before publishing this preview package, verify that:

- `mdrf/mdrf_conv.py` is still a placeholder before acceptance.
- No file contains local absolute paths.
- No checkpoint, raster prediction, log, or cache file is included before acceptance.
- No actual dataset split file is included before acceptance; only placeholder directories and README files are present.
- Import and syntax checks pass for all Python files.
