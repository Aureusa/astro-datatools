from dataclasses import dataclass
import os
import numpy as np

from astro_datatools.core.datasets.coco.annotation import CocoAnnotationBase
from astro_datatools.core.datasets.coco.category import CocoCategoryBase
from astro_datatools.core.datasets.coco.image import CocoImageBase
from astro_datatools.core.datasets.coco.sample import CocoSampleBase
from astro_datatools.core.datasets.coco.utils import (
    mask_area, mask_to_polygon, save_coco_image, mask_to_rle
)


@dataclass
class LoTSS_B2S_CocoAnnotation:
    id: int
    image_id: int
    category_id: int
    gt_component_membership: list
    gt_proposal_validity: float

    def __post_init__(self):
        # Don't call super().__post_init__() since we have a different set of attributes and validation logic
        return

    def to_dict(self) -> dict:
        """Convert the annotation to a dictionary representation."""
        return {
            "id": self.id,
            "image_id": self.image_id,
            "category_id": self.category_id,
            "gt_component_membership": self.gt_component_membership,
            "gt_proposal_validity": self.gt_proposal_validity
        }

    
@dataclass
class Source_CocoCategory(CocoCategoryBase):
    id: int
    name: str

    def __post_init__(self):
        super().__post_init__()


@dataclass
class LoTSS_GRG_CocoImage(CocoImageBase):
    id: int
    file_name: str
    width: int
    height: int

    def __post_init__(self):
        super().__post_init__()


class LoTSS_B2S_Sample(CocoSampleBase):
    def __init__(
            self,
            id: int,
            image_id: int,
            category_id: int,
            ra: float,
            dec: float,
            rgb_image: np.ndarray, # RGB image as a numpy array (C, H, W)
            full_annotation: tuple, # Tuple containing the grg_seg, grg_bbox
            proposed_boxes: np.ndarray,
            proposal_scores: np.ndarray,
            candidates: dict[str, list[tuple[int, int]]],
            features,
            within_proposal_mask,
            rotated: bool = False,
            origin_id: int = None,
            rotation_angle: float = 0.0,
            stretch: str = "sqrt_stretch",
            reprojected: bool = False,
            old_redshift: float = None,
            new_redshift: float = None,
            directory: str = "",
            save_image: bool = True
        ):
        """
        Initialize a LoTSS sample. This class extends the CocoSampleBase. It includes
        attributes specific to the LoTSS dataset. Uses the segmentation map and bounding box
        to create corresponding COCO annotation, category, and image objects.
        """
        # IDs for the registration
        self.id = id
        self.image_id = image_id
        self.category_id = category_id

        # Image data
        self.rgb_image = rgb_image

        # Annotation handling
        self.full_annotation = full_annotation

        # Metadata specific to LoTSS
        self.ra = ra
        self.dec = dec
        self.candidates = candidates
        self.stretch = stretch
        self.rotated = rotated
        self.origin_id = origin_id
        self.rotation_angle = rotation_angle

        # Redshift and reprojection info
        self.reprojected = reprojected
        self.old_redshift = old_redshift
        self.new_redshift = new_redshift

        # Proposed boxes and scores
        self.proposed_boxes = proposed_boxes
        self.proposal_scores = proposal_scores
        self.features = features
        self.within_proposal_mask = within_proposal_mask


        # Action flags
        self.save_image = save_image
        self.image_directory = os.path.join(directory, "images")
        if not os.path.exists(self.image_directory):
            os.makedirs(self.image_directory)
        self.proposal_directory = os.path.join(directory, "proposals")
        if not os.path.exists(self.proposal_directory):
            os.makedirs(self.proposal_directory)

    def register_sample(self) -> dict:
        """
        Register the sample by creating and returning the COCO annotation, category,
        image, and metadata objects.

        :return: Dictionary containing the COCO image and annotation objects.
        The dictionary has the following structure:
        {
            'image': LoTSS_GRG_CocoImage,
            'annotation': LoTSS_GRG_CocoAnnotation
        }
        :rtype: dict
        """
        annotation = self._register_annotation()
        if annotation is None:
            return None
        image = self._register_image()
        image = self._register_metadata(image)
        self._save_proposals()

        return {
            'image': image.to_dict(),
            'annotation': annotation.to_dict()
        }

    def _generate_proposal_filename(self) -> str:
        """Generate proposal filename matching the image filename (with .npz extension)."""
        image_filename = self._generate_image_filename()
        # Replace .png with .npz
        return image_filename.replace('.png', '.npz')

    def _save_proposals(self):
        """Save proposals, proposal scores, and proposal-conditioned features."""
        if self.proposed_boxes is not None and len(self.proposed_boxes) > 0:
            proposal_filename = self._generate_proposal_filename()
            proposal_filepath = os.path.join(self.proposal_directory, proposal_filename)

            # Keep these compact to reduce IO overhead when loaded by a custom mapper.
            boxes = np.asarray(self.proposed_boxes, dtype=np.float32)
            scores = np.asarray(self.proposal_scores, dtype=np.float32)
            features = np.asarray(self.features, dtype=np.float32)
            within_proposal_mask = np.asarray(self.within_proposal_mask, dtype=np.bool_)
            
            np.savez(
                proposal_filepath,
                boxes=boxes,  # (N, 4) in [x1, y1, x2, y2] format
                scores=scores,  # (N,) objectness scores
                features=features,  # (N, C, F) features per proposal-component
                within_proposal_mask=within_proposal_mask,  # (N, C) mask
            )

    def _register_annotation(self) -> LoTSS_B2S_CocoAnnotation:
        gt_component_membership, gt_proposal_validity = self.full_annotation
        
        # Convert to list so they are JSON serializable in the annotation dict
        gt_component_membership = gt_component_membership.tolist() if isinstance(
            gt_component_membership, np.ndarray
        ) else gt_component_membership
        gt_proposal_validity = gt_proposal_validity.tolist() if isinstance(
            gt_proposal_validity, np.ndarray
        ) else gt_proposal_validity


        annotation = LoTSS_B2S_CocoAnnotation(
            id=self.id,
            image_id=self.image_id,
            category_id=self.category_id,
            gt_component_membership=gt_component_membership,
            gt_proposal_validity=gt_proposal_validity
        )
        return annotation
    
    def _register_image(self) -> LoTSS_GRG_CocoImage:
        file_name = self._generate_image_filename()
        full_filepath = os.path.join(self.image_directory, file_name)
        height, width = self.rgb_image.shape[1], self.rgb_image.shape[2]

        self._save_image(full_filepath)

        image = LoTSS_GRG_CocoImage(
            id=self.image_id,
            file_name=file_name,
            width=width,
            height=height
        )
        return image

    def _register_metadata(self, coco_image: LoTSS_GRG_CocoImage) -> LoTSS_GRG_CocoImage:
        metadata = {
            "RA": self.ra,
            "DEC": self.dec,
            "candidates": self.candidates,
            "rotated": self.rotated,
            "origin_id": self.origin_id,
            "rotation_angle": self.rotation_angle,
            "stretch": self.stretch,
            "grg_in_sample": True,
            "reprojected": self.reprojected,
            "old_redshift": self.old_redshift,
            "new_redshift": self.new_redshift
        }
        coco_image.add_metadata(metadata)
        return coco_image

    def _generate_image_filename(self) -> str:
        image_id = str(self.image_id).zfill(10)
        rotation_suffix = "_rotated" if self.rotated else ""
        rotation_angle_suffix = f"_angle{int(self.rotation_angle)}" if self.rotated else ""
        coordinates_suffix = f"_RA{self.ra}_DEC{self.dec}"
        filename = f"LoTSS_GRG_{image_id}{rotation_suffix}{rotation_angle_suffix}{coordinates_suffix}.png"
        return filename
    
    def _save_image(self, filepath: str):
        if self.save_image:
            save_coco_image(self.rgb_image, filepath)

    
class LoTSS_Search_Sample(CocoSampleBase):
    def __init__(
            self,
            id: int,
            image_id: int,
            grg_segmentation: np.ndarray,
            grg_bboxes: list,
            category_id: int,
            ra: float,
            dec: float,
            rgb_image: np.ndarray, # RGB image as a numpy array (C, H, W)
            proposed_boxes: np.ndarray,
            proposal_scores: np.ndarray,
            positions: dict[str, list[tuple[int, int]]], # {key: list of (x, y) positions}
            grg_positions: dict[str, list[tuple[int, int]]], # {key: list of (x, y) positions}
            seg_mode: str = "rle",
            stretch: str = "sqrt_stretch",
            iscrowd: int = 0,
            directory: str = "",
            save_image: bool = True
        ):
        # IDs for the registration
        self.id = id
        self.image_id = image_id
        self.category_id = category_id

        # Image data
        self.rgb_image = rgb_image

        # Annotation handling
        self.iscrowd = iscrowd

        # Metadata specific to LoTSS
        self.ra = ra
        self.dec = dec
        self.stretch = stretch

        # Proposed boxes and scores
        self.proposed_boxes = proposed_boxes
        self.proposal_scores = proposal_scores

        # Positions of components
        self.positions = positions
        self.grg_positions = grg_positions

        # Annotations info
        self.segmentation_mode = seg_mode
        self.grg_segmentation = grg_segmentation
        self.grg_bboxes = grg_bboxes

        # Action flags
        self.save_image = save_image
        self.image_directory = os.path.join(directory, "images")
        if not os.path.exists(self.image_directory):
            os.makedirs(self.image_directory)
        self.proposal_directory = os.path.join(directory, "proposals")
        if not os.path.exists(self.proposal_directory):
            os.makedirs(self.proposal_directory)

    def register_sample(self) -> dict:
        """
        Register the sample by creating and returning the COCO annotation, category,
        image, and metadata objects.

        :return: Dictionary containing the COCO image and annotation objects.
        The dictionary has the following structure:
        {
            'image': LoTSS_GRG_CocoImage,
            'annotation': LoTSS_GRG_CocoAnnotation
        }
        :rtype: dict
        """
        image = self._register_image()
        image = self._register_metadata(image)
        annotation = self._register_annotation()
        self._save_proposals()

        return {
            'image': image.to_dict(),
            'annotation': annotation.to_dict() if annotation is not None else None
        }

    def _register_annotation(self) -> LoTSS_B2S_CocoAnnotation:
        grg_seg, grg_bbox = self.grg_segmentation, self.grg_bboxes
        if grg_seg is None or grg_bbox is None:
            return None
        area = mask_area(grg_seg)
        if self.segmentation_mode == "rle":
            segmentation = mask_to_rle(grg_seg)
        elif self.segmentation_mode == "polygon":
            segmentation = mask_to_polygon(grg_seg)
        else:
            raise ValueError(f"Invalid segmentation mode: {self.segmentation_mode}. Must be 'rle' or 'polygon'.")
        bbox_xyxy = grg_bbox

        # If no segmentation found or bbox is invalid return None
        if area == 0 or not segmentation or bbox_xyxy is None:
            return None

        annotation = LoTSS_B2S_CocoAnnotation(
            id=self.id,
            image_id=self.image_id,
            category_id=self.category_id,
            bbox=bbox_xyxy,
            area=area,
            segmentation=segmentation,
            iscrowd=self.iscrowd
        )
        return annotation

    def _generate_proposal_filename(self) -> str:
        """Generate proposal filename matching the image filename (with .npz extension)."""
        image_filename = self._generate_image_filename()
        # Replace .png with .npz
        return image_filename.replace('.png', '.npz')

    def _save_proposals(self):
        """Save precomputed proposals in a format compatible with Detectron2."""
        if self.proposed_boxes is not None and len(self.proposed_boxes) > 0:
            proposal_filename = self._generate_proposal_filename()
            proposal_filepath = os.path.join(self.proposal_directory, proposal_filename)
            
            np.savez(
                proposal_filepath,
                boxes=self.proposed_boxes,  # (N, 4) in [x1, y1, x2, y2] format
                scores=self.proposal_scores  # (N,) objectness scores
            )
    
    def _register_image(self) -> LoTSS_GRG_CocoImage:
        file_name = self._generate_image_filename()
        full_filepath = os.path.join(self.image_directory, file_name)
        height, width = self.rgb_image.shape[1], self.rgb_image.shape[2]

        self._save_image(full_filepath)

        image = LoTSS_GRG_CocoImage(
            id=self.image_id,
            file_name=file_name,
            width=width,
            height=height
        )
        return image

    def _register_metadata(self, coco_image: LoTSS_GRG_CocoImage) -> LoTSS_GRG_CocoImage:
        metadata = {
            "RA": self.ra,
            "DEC": self.dec,
            "all_component_positions": self.positions,
            "stretch": self.stretch,
        }
        if self.grg_segmentation is not None and self.grg_bboxes is not None:
            metadata["grg_in_sample"] = True
        else:
            metadata["grg_in_sample"] = False
        if self.grg_positions is not None and len(self.grg_positions) > 0:
            metadata["grg_positions"] = self.grg_positions
        else:
            metadata["grg_positions"] = []
        coco_image.add_metadata(metadata)
        return coco_image

    def _generate_image_filename(self) -> str:
        image_id = str(self.image_id).zfill(10)
        coordinates_suffix = f"_RA{self.ra}_DEC{self.dec}"
        filename = f"LoTSS_Search_{image_id}{coordinates_suffix}.png"
        return filename
    
    def _save_image(self, filepath: str):
        if self.save_image:
            save_coco_image(self.rgb_image, filepath)


class LoTSS_Negative_GRG_Sample(LoTSS_Search_Sample):
    def __init__(
            self,
            id: int,
            image_id: int,
            category_id: int,
            ra: float,
            dec: float,
            rgb_image: np.ndarray, # RGB image as a numpy array (C, H, W)
            proposed_boxes: np.ndarray,
            proposal_scores: np.ndarray,
            positions: dict[str, list[tuple[int, int]]], # {key: list of (x, y) positions}
            stretch: str = "sqrt_stretch",
            iscrowd: int = 0,
            directory: str = "",
            save_image: bool = True
        ):
        super().__init__(
            id=id,
            image_id=image_id,
            grg_segmentation=None,
            grg_bboxes=None,
            category_id=category_id,
            ra=ra,
            dec=dec,
            rgb_image=rgb_image,
            proposed_boxes=proposed_boxes,
            proposal_scores=proposal_scores,
            positions=positions,
            grg_positions=[], # No GRG positions for negative samples
            seg_mode="rle", # Negative samples won't have GRG annotations, but we can still use RLE for consistency
            stretch=stretch,
            iscrowd=iscrowd,
            directory=directory,
            save_image=save_image
        )

    def _register_metadata(self, coco_image: LoTSS_GRG_CocoImage) -> LoTSS_GRG_CocoImage:
        metadata = {
            "RA": self.ra,
            "DEC": self.dec,
            "all_component_positions": self.positions,
            "grg_in_sample": False, # Explicitly indicate that this is a negative sample with no GRG
            "grg_positions": [], # No GRG positions for negative samples
            "stretch": self.stretch,
        }
        coco_image.add_metadata(metadata)
        return coco_image

    def _generate_image_filename(self) -> str:
        image_id = str(self.image_id).zfill(10)
        coordinates_suffix = f"_RA{self.ra}_DEC{self.dec}"
        filename = f"LoTSS_Negative_GRG_{image_id}{coordinates_suffix}.png"
        return filename
