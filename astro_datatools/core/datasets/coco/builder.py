from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
import json
import logging

from astro_datatools.logger import setup_logging

from .sample import CocoSampleBase
from .category import CocoCategoryBase


def _find_non_serializable(obj, path="root", out=None, max_items=50):
    if out is None:
        out = []
    if len(out) >= max_items:
        return out

    # Fast path: if json can encode it, skip recursion
    try:
        json.dumps(obj)
        return out
    except TypeError:
        pass

    if isinstance(obj, dict):
        for k, v in obj.items():
            _find_non_serializable(v, f"{path}[{repr(k)}]", out, max_items)
            if len(out) >= max_items:
                break
        return out

    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _find_non_serializable(v, f"{path}[{i}]", out, max_items)
            if len(out) >= max_items:
                break
        return out

    typename = type(obj).__name__
    detail = ""
    if np is not None and isinstance(obj, np.ndarray):
        detail = f" shape={obj.shape} dtype={obj.dtype}"
    out.append(f"{path}: {typename}{detail}")
    return out


class CocoDatasetBuilderBase(ABC):
    def __post_init__(self):
        # Ensure that logger is set up for the class
        # Check if there is a self.logger attribute, if not, set it up
        if not hasattr(self, 'logger'):
            self.logger = setup_logging(name=f"astro_datatools.{self.__class__.__name__}")

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

        # Generate samples and register them into the COCO dataset
        coco = self._populate_samples(coco)

        self.logger.info(f"COCO dataset built with {len(coco['images'])} images, "
                    f"{len(coco['annotations'])} annotations, "
                    f"and {len(coco['categories'])} categories.")

        try:
            # Save COCO dataset as JSON
            with open(filepath, 'w') as f:
                json.dump(coco, f, indent=4)
        except TypeError as e:
            bad = _find_non_serializable(coco)
            self.logger.error("\n[DEBUG] Non-serializable objects found in COCO dataset:")
            for line in bad:
                self.logger.error(f"  - {line}")
            raise TypeError(f"Failed to serialize COCO dataset to JSON: {e}")

        return coco

    def _register_sample(self, sample: CocoSampleBase, coco: dict) -> dict:
        """
        Register a sample into the COCO dataset structure.

        :param sample: The sample to register.
        :type sample: CocoSampleBase
        :param coco: The COCO dataset dictionary to update.
        :type coco: dict
        :return: Updated COCO dataset dictionary.
        :rtype: dict
        """
        result = sample.register_sample()
        if result is None:
            return coco
        coco["images"].append(result['image'])
        coco["annotations"].append(result['annotation'])
        return coco
    
    @abstractmethod
    def _get_filepath(self) -> str:
        pass
        
    @abstractmethod
    def _add_categories(self, coco: dict) -> dict:
        pass

    @abstractmethod
    def _populate_samples(self, coco: dict) -> dict:
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
