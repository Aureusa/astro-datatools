import numpy as np
import json
from copy import deepcopy
import os

from .probe import COCOProbe


class COCODatasetCleaner(COCOProbe):
    def __init__(self, annotations: dict, annotations_path: str, cleaned_annotations_path: str):
        super().__init__(annotations, annotations_path)

        self.annotations_path = annotations_path
        self.cleaned_annotations_path = cleaned_annotations_path

    def clean(self, save_cleaned_dataset: bool = True):
        removed_images_ids = []
        updated_images_ids = []
        updated_images = []
        for image in self.annotations['images']:
            mask = self._get_mask_from_annotations(image)
            grg_components = self._extract_gt_components(image)
            all_components = self._extract_all_components(image)
            non_grg_components = self._remove_grg_from_all_components(all_components, grg_components)

            cleaned_grg_components, cleaned_non_grg_components = self._add_non_grg_components_in_mask_to_grg_components(
                grg_components, non_grg_components, mask
            )

            if not cleaned_grg_components:
                # Empty GRG components list after clean up, means we remove the image from the dataset along with all its annotations.
                # We can achieve this by simply not adding it to the updated_images
                removed_images_ids.append(image['id'])
                continue

            updated_image = self._update_metadata_with_cleaned_components(
                image, cleaned_grg_components, cleaned_non_grg_components
            )
            updated_images.append(updated_image)

            if cleaned_grg_components != grg_components:
                updated_images_ids.append(image['id'])
        self.annotations['images'] = updated_images
        if save_cleaned_dataset:
            self._save_cleaned_dataset(self.annotations)
        return self.annotations, self.cleaned_annotations_path, updated_images_ids, removed_images_ids
    
    def _update_metadata_with_cleaned_components(self, image_dict: dict, grg_components: list, non_grg_components: list):
        """
        Update the annotations for the given image with the cleaned GRG components.
        This is a placeholder for the actual update logic.
        """
        image_dict = deepcopy(image_dict)
        image_dict['metadata']['grg_positions'] = grg_components
        image_dict['metadata']['all_component_positions'] = grg_components + non_grg_components
        return image_dict
    
    def _add_non_grg_components_in_mask_to_grg_components(
            self,
            grg_components: list,
            non_grg_components: list,
            mask: np.ndarray
        ):
        """
        Remove non-GRG components that are within the predicted mask from the dataset.
        This is a placeholder for the actual cleaning logic.
        """
        cleaned_non_grg_components = []
        for comp in non_grg_components:
            x, y = comp
            if mask[int(y), int(x)] == 0:
                cleaned_non_grg_components.append(comp)
            elif mask[int(y), int(x)] == 1:
                grg_components.append(comp)
            else:
                print("WTF")

        cleaned_grg_components = []
        for comp in grg_components:
            x, y = comp
            if mask[int(y), int(x)] == 0:
                cleaned_non_grg_components.append(comp)
            elif mask[int(y), int(x)] == 1:
                cleaned_grg_components.append(comp)
            else:
                print("WTF")
        return cleaned_grg_components, cleaned_non_grg_components
    
    def _save_cleaned_dataset(self, cleaned_annotations: dict):
        """
        Save the cleaned dataset to a new JSON file.
        """
        # Get the directory of the original annotations file
        with open(self.cleaned_annotations_path, 'w') as f:
            json.dump(cleaned_annotations, f)

        print(f"Cleaned dataset saved to {self.cleaned_annotations_path}")
