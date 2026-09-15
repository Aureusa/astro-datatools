from abc import ABC
from dataclasses import dataclass


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
