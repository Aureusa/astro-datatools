from dataclasses import dataclass

from astro_datatools.core.datasets.coco.category import CocoCategoryBase


@dataclass
class LoTSS_GRG_CocoCategory(CocoCategoryBase):
    id: int = 1
    name: str = "GRG"
    