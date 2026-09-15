from abc import ABC
from dataclasses import dataclass


@dataclass
class CocoImageBase(ABC):
    id: int
    file_name: str
    width: int
    height: int

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive integers.")
        
        if not self.file_name:
            raise ValueError("File name must be a non-empty string.")

    def to_dict(self) -> dict:
        """Convert the image to a dictionary representation."""
        dict_representation = {
            "id": self.id,
            "file_name": self.file_name,
            "width": self.width,
            "height": self.height
        }
        if hasattr(self, 'metadata'):
            dict_representation['metadata'] = self.metadata
        return dict_representation
    
    def add_metadata(self, metadata: dict) -> dict:
        """Add metadata to the image dictionary representation."""
        self.metadata = metadata
        