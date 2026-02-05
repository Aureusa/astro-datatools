from abc import ABC, abstractmethod
from tqdm import tqdm
import json
import logging

from .sample import CocoSampleBase
from .category import CocoCategoryBase


logger = logging.getLogger(__name__)


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
        # samples = self._generate_samples()
        # for samp in tqdm(samples, desc="Registering Samples"):
        #     result = samp.register_sample()
        #     coco["images"].append(result['image'])
        #     coco["annotations"].append(result['annotation'])
        coco = self._populate_samples(coco)

        logger.info(f"COCO dataset built with {len(coco['images'])} images, "
                    f"{len(coco['annotations'])} annotations, "
                    f"and {len(coco['categories'])} categories.")

        # Save COCO dataset as JSON
        with open(filepath, 'w') as f:
            json.dump(coco, f, indent=4)

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
