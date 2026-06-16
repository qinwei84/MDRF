# US3D Split

This directory contains the US3D split used in this project. The split follows the MaskCRNet/Rao2024-style protocol used for the paper experiments.

Files:

```text
train.txt
val.txt
test.txt
```

Each line stores four dataset-relative paths:

```text
left_image right_image disparity_map label_map
```

The dataset root is the US3D directory containing `JAX/` and `OMA/`. Counts are 4192 train pairs, 50 validation pairs, and 50 test pairs.
