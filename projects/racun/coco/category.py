from dataclasses import dataclass

from astro_datatools.core.datasets.coco.category import CocoCategoryBase


@dataclass
class LoTSS_B2S_SCS_CocoCategory(CocoCategoryBase):
    """Single-component source category for B2S Mask R-CNN annotations."""
    id: int = 1
    name: str = "SCS"


@dataclass
class LoTSS_B2S_MCS_CocoCategory(CocoCategoryBase):
    """Multi-component source category for B2S Mask R-CNN annotations."""
    id: int = 2
    name: str = "MCS"

