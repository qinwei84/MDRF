# MDRF Experimental Skeletons

This directory collects staged MDRF insertion points used for ablation and transfer experiments. HMSMNet and PSMNet are host frameworks; the MDRF operators are the target contribution.

The core implementation is not included in this public preview. All experiment files import the operators from the central package:

```python
from mdrf import MDRFConv
from mdrf import MDRFConv3D
```

After formal release, replace `mdrf/mdrf_conv.py` with the official implementation.

## Folders

- `hmsmnet_feature_mdrf/`: inserts 2D MDRF blocks into the HMSMNet feature extractor.
- `hmsmnet_refinement_mdrf/`: inserts 2D MDRF blocks into the refinement stage.
- `hmsmnet_hourglass_mdrf3d/`: inserts 3D MDRF blocks into the hourglass aggregation stage.
- `hmsmnet_fusion_mdrf3d/`: inserts 3D MDRF blocks into the cost-volume fusion stage.
- `hmsmnet_all_stages_mdrf/`: combines the feature, 3D aggregation/fusion, and refinement insertions.
- `psmnet_spp_mdrf/`: replaces the PSMNet SPP-style context branch with a 2D MDRF branch.
- `psmnet_3d_aggregation_mdrf/`: inserts 3D MDRF into PSMNet cost-volume aggregation.

Training scripts, data splits, checkpoints, and prediction outputs are intentionally excluded.
