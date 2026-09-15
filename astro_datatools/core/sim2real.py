import numpy as np

from ..augment import ConvolveAugment, ShotNoiseAugment
from ..transforms.geometric import crop
from ..transforms.normalise import normalise
from ..transforms.embed import embed_in_background as eib


def sim2real(
        image: np.ndarray,
        psf_kernel: np.ndarray,
        background: np.ndarray,
        *,
        crop_size: int = 128,
        norm_bounds: tuple = (0, 1),
        shot_noise_intensity: int = 1000,
        embed_type: str = "default"
    ) -> np.ndarray:
    """
    Convert a simulated image to a real image using a series of augmentations and transformations.

    :param image: Input simulated image.
    :type image: np.ndarray
    :param psf_kernel: Point spread function kernel for convolution.
    :type psf_kernel: np.ndarray
    :param background: Background image to embed the simulated image into.
    :type background: np.ndarray
    :param crop_size: Size of the center crop to apply to the image.
    :type crop_size: int
    :param norm_bounds: Normalization bounds for the image.
    :type norm_bounds: tuple
    :param shot_noise_intensity: Expected number of detected photons or photoelectrons
    represented by a normalized pixel value of 1. Must be positive. For example, intensity=1000
    means that a pixel with value 1 corresponds to 1000 expected detected photons,
    while a pixel with value 0.2 corresponds to 200 expected photons.
    :type shot_noise_intensity: int
    :param embed_type: Type of embedding to use when embedding the image into the background.
                    Supported types:
                    - "default": Sum the image and background arrays.
                    - "fill": Embed the image into the background,
                    keeping non-zero pixels from the image. Effectively
                    replaces the background pixels with the image pixels
                    wherever the image is non-zero.
    :type embed_type: str
    :return: Transformed real image.
    :rtype: np.ndarray
    """
    bg_already_cropped = False
    if image.shape != background.shape:
        if background.ndim == image.ndim:
            if background.shape[-1] == crop_size and background.shape[-2] == crop_size:
                bg_already_cropped = True
            else:
                raise ValueError("Image and background have the same dimensions but the background does not match the crop size or the image shape.")
        else:
            raise ValueError("Image and background must have the same dimensions. image.ndim={}, background.ndim={}".format(image.ndim, background.ndim))
        
    if image.ndim == 4:
        # The image is assumed to have a batch dimension. (B,C,H,W)
        batch_size, channels, height, width = image.shape
    elif image.ndim == 3:
        # The image is assumed to have no batch dimension. (C,H,W)
        channels, height, width = image.shape
    else:
        raise ValueError("Unsupported image dimensions. Expected 3 (C,H,W) or 4 dimensions (B,C,H,W).")
    
    # Apply geometric center crop
    image = crop(
        image,
        size=crop_size
    )
    if not bg_already_cropped:
        background = crop(
            background,
            size=crop_size
        )

    # Apply normalization
    image = normalise(image, bounds=norm_bounds)

    # Apply convolution augmentation
    conv_augment = ConvolveAugment(psf_kernel=psf_kernel)
    image = conv_augment.augment(image)

    # Apply shot noise augmentation
    shot_noise_augment = ShotNoiseAugment(intensity=shot_noise_intensity)
    image = shot_noise_augment.augment(image)

    # Embed in background
    image = eib(image, background, embed_type=embed_type)
    
    return image
