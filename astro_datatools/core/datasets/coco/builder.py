from abc import ABC, abstractmethod
from tqdm import tqdm
import json

from .sample import CocoSampleBase
from .category import CocoCategoryBase


class CocoDatasetBuilderBase(ABC):
    def build(self) -> dict:
        """
        Build the COCO dataset by generating samples, categories, and saving to a JSON file.

        :return: Dictionary representing the COCO dataset.
        :rtype: dict
        """
        # Define the COCO dataset structure
        coco = {
            "images": [],
            "annotations": [],
            "categories": [],
        }

        # Generate filepath to save the COCO dataset
        filepath = self._get_filepath()
        self._validate_filepath(filepath)

        # Add categories
        coco = self._add_categories(coco)
        self._validate_categories(coco)

        # Generate samples and register them
        samples = self._generate_samples()
        for samp in tqdm(samples, desc="Registering Samples"):
            result = samp.register_sample()
            coco["images"].append(result['image'])
            coco["annotations"].append(result['annotation'])

        # Save COCO dataset as JSON
        with open(filepath, 'w') as f:
            json.dump(coco, f, indent=4)

        return coco
    
    @abstractmethod
    def _get_filepath(self) -> str:
        pass
        
    @abstractmethod
    def _add_categories(self, coco: dict) -> dict:
        pass

    @abstractmethod
    def _generate_samples(self) -> list[CocoSampleBase]:
        pass

    def _validate_categories(self, coco: dict) -> None:
        if not coco.get("categories"):
            raise ValueError("No categories found in the COCO dataset.")
        
        # Categories should be dicts with 'id' and 'name' keys after to_dict() conversion
        for cat in coco["categories"]:
            if not isinstance(cat, dict):
                raise TypeError("All categories must be dictionaries.")
            if 'id' not in cat or 'name' not in cat:
                raise ValueError("Each category must have 'id' and 'name' keys.")
        
    def _validate_filepath(self, filepath: str) -> None:
        if not filepath.endswith('.json'):
            raise ValueError("The COCO dataset filepath must end with .json")
