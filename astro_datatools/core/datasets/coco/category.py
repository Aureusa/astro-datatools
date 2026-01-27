from abc import ABC
from dataclasses import dataclass
from typing import ClassVar, List, Dict

@dataclass
class CocoCategoryBase(ABC):
    id: int
    name: str

    def __post_init__(self):
        if not self.name:
            raise ValueError("Category name must be a non-empty string.")

    def to_dict(self) -> dict:
        """Convert the category to a dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
        }
    

@dataclass
class LoTSS_GRG_CocoCategory(CocoCategoryBase):
    id: int = 1
    name: str = "GRG"
