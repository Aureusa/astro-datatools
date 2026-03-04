from copy import deepcopy
from pycocotools.coco import COCO


class COCOProbe:
    def __init__(self, annotations, annotations_path: str):
        self.annotations = deepcopy(annotations)
        self.coco = COCO(annotations_path)

    def _get_mask_from_annotations(self, image_dict: dict):
        """
        Get the binary mask from the COCO annotations for the given image.
        """
        ann_ids = self.coco.getAnnIds(imgIds=[image_dict['id']], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        mask = self.coco.annToMask(anns[0])
        return mask
    
    def _get_bbox_from_annotations(self, image_dict: dict):
        """
        Get the bounding box from the COCO annotations for the given image.
        """
        ann_ids = self.coco.getAnnIds(imgIds=[image_dict['id']], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        bbox = anns[0]['bbox']
        return bbox
    
    def _extract_gt_components(self, image_metadata: dict):
        """
        These are the central coordinates of the radio components
        that belong together according to the manual association.
        It is a list of tuples (x, y) for each GRG in the image.
        The number of GRGs in the image is equal to the length of this list.
        """
        return image_metadata.get('grg_positions', [])

    def _extract_all_components(self, image_metadata: dict):
        """
        These are the central coordinates of all radio components in the image,
        including those that do not belong to the GRG according to the manual association.
        It is a list of tuples (x, y) for each radio component in the image,
        including those that do not belong to the GRG according to the manual association.
        The number of radio components in the image is equal to the length of this list.
        """
        all_components = image_metadata.get('all_component_positions', [])
        if not all_components:
            all_components = image_metadata.get('positions', [])
        if not all_components:
            raise ValueError("No component positions found in image metadata.")
        return all_components
    
    def _remove_grg_from_all_components(self, all_components: list, grg_components: list):
        """
        Remove the GRG components from the list of all components to get the list of non-GRG components.
        """
        return [comp for comp in all_components if comp not in grg_components]
    