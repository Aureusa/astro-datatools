from abc import ABC
from dataclasses import dataclass


@dataclass
class CocoAnnotationBase(ABC):
    id: int
    image_id: int
    category_id: int
    bbox: list
    area: float
    segmentation: list
    iscrowd: int = 0

    def __post_init__(self):
        if self.area < 0:
            raise ValueError("Area must be a non-negative float.")
        
        if not isinstance(self.bbox, list) or len(self.bbox) != 4:
            raise ValueError("Bounding box must be a list of four elements [x, y, width, height].")

    def to_dict(self) -> dict:
        """Convert the annotation to a dictionary representation."""
        return {
            "id": self.id,
            "image_id": self.image_id,
            "category_id": self.category_id,
            "bbox": self.bbox,
            "area": self.area,
            "segmentation": self.segmentation,
            "iscrowd": self.iscrowd
        }
