import cv2
import numpy as np
from pycocotools import mask as mask_utils


def mask_to_rle(mask: np.ndarray) -> dict:
    """
    Convert binary mask to COCO RLE format.
    More efficient for complex segmentations.
    
    :param mask: 2D numpy array (H, W)
    :return: RLE-encoded mask
    """
    # Ensure Fortran-contiguous order
    mask_fortran = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask_fortran)
    # Decode bytes to string for JSON serialization
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def mask_to_polygon(mask: np.ndarray) -> list:
    """
    Convert a binary mask to COCO polygon format.
    
    :param mask: 2D numpy array representing the binary mask.
    :type mask: np.ndarray
    :return: List of polygons in COCO format.
    :rtype: list
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    segmentation = []
    for contour in contours:
        if contour.size < 6:
            continue
        segmentation.append(contour.flatten().tolist())

    return segmentation


def mask_area(mask: np.ndarray) -> float:
    """
    Calculate the area of a binary mask.
    
    :param mask: 2D numpy array representing the binary mask.
    :type mask: np.ndarray
    :return: Area of the mask.
    :rtype: float
    """
    return float(np.sum(mask > 0))


def bbox_to_xywh(bbox: dict) -> list:
    """
    Convert bounding box from a dictionary in the format:
    {
        "top": float,
        "bottom": float,
        "left": float,
        "right": float
    }
    to a list in the format [x, y, w, h] format.
    
    :param bbox: List or tuple of bounding box in [x_min, y_min, x_max, y_max] format.
    :type bbox: dict
    :return: List of bounding box in (x, y, w, h) format.
    :rtype: list
    """
    x1, y1, x2, y2 = bbox["left"], bbox["bottom"], bbox["right"], bbox["top"]
    return [x1, y1, x2 - x1, y2 - y1]

def save_coco_image(image: np.ndarray, file_name: str, normalize_per_image: bool = True):
    """
    Save a COCO image (C, H, W) numpy array to a PNG file.
    Uses 16-bit PNG to preserve dynamic range for float data.

    :param image: 3D numpy array representing the image in (C, H, W) format.
    :type image: np.ndarray
    :param file_name: Name of the file to save the image to.
    :type file_name: str
    :param normalize_per_image: If True, normalizes each image to its own min/max.
                                 If False, assumes values are already in [0,1] range.
    :type normalize_per_image: bool
    """
    # Convert from (C, H, W) to (H, W, C)
    image_hwc = np.transpose(image, (1, 2, 0))
    
    if image_hwc.dtype in [np.float32, np.float64]:
        if normalize_per_image:
            # Normalize per channel to preserve the distinct channel characteristics
            img_min = image_hwc.min(axis=(0, 1), keepdims=True)
            img_max = image_hwc.max(axis=(0, 1), keepdims=True)
            
            # Avoid division by zero
            range_mask = (img_max - img_min) > 1e-8
            image_normalized = np.where(
                range_mask,
                (image_hwc - img_min) / (img_max - img_min),
                0
            )
            # Scale to 16-bit range [0, 65535]
            image_uint16 = (image_normalized * 65535).astype(np.uint16)
        else:
            # Clip to [0, 1] and scale to [0, 65535]
            image_clipped = np.clip(image_hwc, 0, 1)
            image_uint16 = (image_clipped * 65535).astype(np.uint16)
        
        # Convert from RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_uint16, cv2.COLOR_RGB2BGR)
    else:
        # Already integer type
        image_bgr = cv2.cvtColor(image_hwc, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(file_name, image_bgr)