from dataclasses import dataclass

from astro_datatools.core.datasets.coco.category import CocoCategoryBase


@dataclass
class LoTSS_B2S_CocoCategory(CocoCategoryBase):
    id: int = 1
    name: str = "radio_source"
    