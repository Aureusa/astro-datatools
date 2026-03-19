from dataclasses import dataclass

from astro_datatools.core.datasets.coco.category import CocoCategoryBase


@dataclass
class LoTSS_B2S_CocoCategory(CocoCategoryBase):
    id: int = 1
    name: str = "radio_source"


@dataclass
class LoTSS_B2S_SCS_CocoCategory(CocoCategoryBase):
    """Single-component source category for the B2S dataset."""
    id: int = 1
    name: str = "SCS"
    label: int = 1

    def to_dict(self) -> dict:
        """Convert the category to a dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "label": self.label, # This is the label that will be used in the annotations to indicate this category.
            # It should be unique across categories.
        }


@dataclass
class LoTSS_B2S_MCS_CocoCategory(CocoCategoryBase):
    """Multi-component source category for the B2S dataset."""
    id: int = 2
    name: str = "MCS"
    label: int = 2

    def to_dict(self) -> dict:
        """Convert the category to a dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "label": self.label, # This is the label that will be used in the annotations to indicate this category.
            # It should be unique across categories.
        }
    