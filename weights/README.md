# Model Weights

This directory is reserved for official model weights. No trained weights are included in the pre-acceptance preview.

After the paper is accepted, release weights can be placed under:

```text
weights/hmsmnet/
weights/psmnet/
```

Suggested filenames:

```text
weights/hmsmnet/mdrf_hmsmnet_whu_stereo.pth
weights/hmsmnet/mdrf_hmsmnet_us3d.pth
weights/psmnet/mdrf_psmnet_whu_stereo.pth
```

Please also add a short note for each released weight, including dataset, input resolution, disparity range, training protocol, and expected metrics.

If a checkpoint is larger than GitHub normal file limits, publish it through GitHub Releases or Git LFS and keep the download link in this directory.
