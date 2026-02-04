from dataclasses import dataclass
import os
import numpy as np

from astro_datatools.core.datasets.coco.annotation import CocoAnnotationBase
from astro_datatools.core.datasets.coco.category import CocoCategoryBase
from astro_datatools.core.datasets.coco.image import CocoImageBase
from astro_datatools.core.datasets.coco.sample import CocoSampleBase
from astro_datatools.core.datasets.coco.utils import mask_area, mask_to_polygon, save_coco_image


@dataclass
class LoTSS_GRG_CocoAnnotation(CocoAnnotationBase):
    id: int
    image_id: int
    category_id: int
    bbox: list
    area: float
    segmentation: list
    iscrowd: int = 0

    def __post_init__(self):
        super().__post_init__()
    

@dataclass
class LoTSS_GRG_CocoCategory(CocoCategoryBase):
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


class LoTSS_Sample(CocoSampleBase):
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
            grg_positions: list[tuple[int, int]],
            all_component_positions: list[tuple[int, int]],
            rotated: bool = False,
            rotation_angle: float = 0.0,
            stretch: str = "sqrt_stretch",
            reprojected: bool = False,
            old_redshift: float = None,
            new_redshift: float = None,
            iscrowd: int = 0,
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
        self.iscrowd = iscrowd

        # Metadata specific to LoTSS
        self.ra = ra
        self.dec = dec
        self.stretch = stretch
        self.rotated = rotated
        self.rotation_angle = rotation_angle

        # Redshift and reprojection info
        self.reprojected = reprojected
        self.old_redshift = old_redshift
        self.new_redshift = new_redshift

        # Proposed boxes and scores
        self.proposed_boxes = proposed_boxes
        self.proposal_scores = proposal_scores

        # Positions of components
        self.grg_positions = grg_positions
        self.all_component_positions = all_component_positions

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
        annotation = self._register_annotation()  # Assuming data is in the second channel
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
        """Save precomputed proposals in a format compatible with Detectron2."""
        if self.proposed_boxes is not None and len(self.proposed_boxes) > 0:
            proposal_filename = self._generate_proposal_filename()
            proposal_filepath = os.path.join(self.proposal_directory, proposal_filename)
            
            np.savez_compressed(
                proposal_filepath,
                boxes=self.proposed_boxes,  # (N, 4) in [x1, y1, x2, y2] format
                scores=self.proposal_scores  # (N,) objectness scores
            )

    def _register_annotation(self) -> LoTSS_GRG_CocoAnnotation:
        grg_seg, grg_bbox = self.full_annotation
        area = mask_area(grg_seg)
        segmentation = mask_to_polygon(grg_seg)
        bbox_xyxy = grg_bbox # bbox_to_xywh(grg_bbox)

        annotation = LoTSS_GRG_CocoAnnotation(
            id=self.id,
            image_id=self.image_id,
            category_id=self.category_id,
            bbox=bbox_xyxy,
            area=area,
            segmentation=segmentation,
            iscrowd=self.iscrowd
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
            "grg_positions": self.grg_positions,
            "all_component_positions": self.all_component_positions,
            "rotated": self.rotated,
            "rotation_angle": self.rotation_angle,
            "stretch": self.stretch,
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
    