from dataclasses import dataclass
import os
import numpy as np

from astro_datatools.core.datasets.coco.annotation import CocoAnnotationBase
from astro_datatools.core.datasets.coco.image import CocoImageBase
from astro_datatools.core.datasets.coco.sample import CocoSampleBase
from astro_datatools.core.datasets.coco.utils import mask_area, mask_to_rle, save_coco_image


@dataclass
class LoTSS_B2S_CocoAnnotation(CocoAnnotationBase):
    id: int
    image_id: int
    category_id: int
    bbox: list
    area: float
    segmentation: dict
    iscrowd: int = 0

    def __post_init__(self):
        super().__post_init__()


@dataclass
class LoTSS_B2S_CocoImage(CocoImageBase):
    id: int
    file_name: str
    width: int
    height: int

    def __post_init__(self):
        super().__post_init__()


class LoTSS_B2S_MaskRCNN_Sample(CocoSampleBase):
    def __init__(
            self,
            id: int,
            image_id: int,
            ra: float,
            dec: float,
            rgb_image: np.ndarray,
            proposed_boxes: np.ndarray,
            proposal_scores: np.ndarray,
            instance_bboxes,
            instance_masks,
            instance_category_ids,
            candidates: dict[str, list[tuple[int, int]]],
            rotated: bool = False,
            origin_id: int = None,
            rotation_angle: float = 0.0,
            stretch: str = "sqrt_stretch",
            reprojected: bool = False,
            old_redshift: float = None,
            new_redshift: float = None,
            iscrowd: int = 0,
            directory: str = "",
            save_image: bool = True
        ):
        self.id = id
        self.image_id = image_id
        self.rgb_image = rgb_image
        self.iscrowd = iscrowd

        self.ra = ra
        self.dec = dec
        self.candidates = candidates
        self.stretch = stretch
        self.rotated = rotated
        self.origin_id = origin_id
        self.rotation_angle = rotation_angle

        self.reprojected = reprojected
        self.old_redshift = old_redshift
        self.new_redshift = new_redshift

        self.proposed_boxes = proposed_boxes
        self.proposal_scores = proposal_scores

        self.instance_bboxes = np.asarray(instance_bboxes, dtype=np.float32)
        self.instance_masks = list(instance_masks)
        self.instance_category_ids = np.asarray(instance_category_ids, dtype=np.int32)

        self.save_image = save_image
        self.image_directory = os.path.join(directory, "images")
        if not os.path.exists(self.image_directory):
            os.makedirs(self.image_directory)
        self.proposal_directory = os.path.join(directory, "proposals")
        if not os.path.exists(self.proposal_directory):
            os.makedirs(self.proposal_directory)

    def register_sample(self) -> dict:
        annotations = self._register_annotations()
        if not annotations:
            return None

        image = self._register_image()
        image = self._register_metadata(image)
        self._save_proposals()

        return {
            "image": image.to_dict(),
            "annotations": [ann.to_dict() for ann in annotations],
        }

    def _generate_proposal_filename(self) -> str:
        image_filename = self._generate_image_filename()
        return image_filename.replace(".png", ".npz")

    def _save_proposals(self):
        if self.proposed_boxes is not None and len(self.proposed_boxes) > 0:
            proposal_filename = self._generate_proposal_filename()
            proposal_filepath = os.path.join(self.proposal_directory, proposal_filename)

            boxes = np.asarray(self.proposed_boxes, dtype=np.float32)
            scores = np.asarray(self.proposal_scores, dtype=np.float32)

            np.savez(
                proposal_filepath,
                boxes=boxes,
                scores=scores,
            )

    def _register_annotations(self) -> list[LoTSS_B2S_CocoAnnotation]:
        annotations = []

        if self.instance_bboxes.shape[0] == 0:
            return annotations

        for bbox, mask, category_id in zip(
            self.instance_bboxes,
            self.instance_masks,
            self.instance_category_ids,
        ):
            mask_array = np.asarray(mask, dtype=np.uint8)
            area = mask_area(mask_array)
            if area <= 0:
                continue

            segmentation = mask_to_rle(mask_array)
            bbox_xyxy = [float(v) for v in np.asarray(bbox, dtype=np.float32).tolist()]

            annotations.append(
                LoTSS_B2S_CocoAnnotation(
                    id=-1,
                    image_id=self.image_id,
                    category_id=int(category_id),
                    bbox=bbox_xyxy,
                    area=area,
                    segmentation=segmentation,
                    iscrowd=self.iscrowd,
                )
            )

        return annotations

    def _register_image(self) -> LoTSS_B2S_CocoImage:
        file_name = self._generate_image_filename()
        full_filepath = os.path.join(self.image_directory, file_name)
        height, width = self.rgb_image.shape[1], self.rgb_image.shape[2]

        self._save_image(full_filepath)

        image = LoTSS_B2S_CocoImage(
            id=self.image_id,
            file_name=file_name,
            width=width,
            height=height,
        )
        return image

    def _register_metadata(self, coco_image: LoTSS_B2S_CocoImage) -> LoTSS_B2S_CocoImage:
        metadata = {
            "RA": self.ra,
            "DEC": self.dec,
            "candidates": self.candidates,
            "rotated": self.rotated,
            "origin_id": self.origin_id,
            "rotation_angle": self.rotation_angle,
            "stretch": self.stretch,
            "reprojected": self.reprojected,
            "old_redshift": self.old_redshift,
            "new_redshift": self.new_redshift,
        }
        coco_image.add_metadata(metadata)
        return coco_image

    def _generate_image_filename(self) -> str:
        image_id = str(self.image_id).zfill(10)
        rotation_suffix = "_rotated" if self.rotated else ""
        rotation_angle_suffix = f"_angle{int(self.rotation_angle)}" if self.rotated else ""
        coordinates_suffix = f"_RA{self.ra}_DEC{self.dec}"
        filename = f"LoTSS_B2S_MaskRCNN_{image_id}{rotation_suffix}{rotation_angle_suffix}{coordinates_suffix}.png"
        return filename

    def _save_image(self, filepath: str):
        if self.save_image:
            save_coco_image(self.rgb_image, filepath)
