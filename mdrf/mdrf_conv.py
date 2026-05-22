"""Placeholder for the MDRF operators.

This public preview keeps the core implementation private before paper acceptance.
For the formal release, replace this file with the official implementation that
provides both `MDRFConv` and `MDRFConv3D`.
"""

import torch.nn as nn


class _UnreleasedMDRF(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise ImportError(
            "The official MDRFConv/MDRFConv3D implementation is not included in this preview. "
            "Please replace mdrf/mdrf_conv.py with the formal release file."
        )


MDRFConv = _UnreleasedMDRF
MDRFConv3D = _UnreleasedMDRF
