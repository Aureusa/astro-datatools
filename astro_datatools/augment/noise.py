import numpy as np
from .base import BaseAugment


class ShotNoiseAugment(BaseAugment):
    """
    Add Poisson-distributed shot noise to simulated astronomical images.

    Shot noise is an important source of noise in astronomical observations.
    It arises from the discrete nature of photon detection: even when a source
    has a fixed expected brightness, the number of photons detected in each
    pixel fluctuates according to Poisson statistics.

    This augmentation assumes that input images are normalized to the range
    ``[0, 1]``. The ``intensity`` parameter defines the expected number of
    detected photons (or photoelectrons) corresponding to a pixel value of
    ``1``. Each normalized pixel value is therefore converted into an expected
    photon count, sampled from a Poisson distribution, and converted back to
    the normalized ``[0, 1]`` representation.

    For an image pixel with normalized value ``x``, the operation is:

        expected_counts = x * intensity
        observed_counts = Poisson(expected_counts)
        noisy_pixel = observed_counts / intensity

    Consequently, brighter pixels have larger absolute fluctuations, while
    the relative effect of shot noise becomes smaller for brighter sources,
    as expected for Poisson counting statistics.

    Parameters
    ----------
    intensity : float
        Expected number of detected photons or photoelectrons represented by
        a normalized pixel value of ``1``. Must be positive.

        For example, ``intensity=1000`` means that a pixel with value ``1``
        corresponds to 1000 expected detected photons, while a pixel with
        value ``0.2`` corresponds to 200 expected photons.

    Notes
    -----
    This augmentation models photon-counting (shot) noise only. Real
    astronomical observations can contain additional noise sources, such as
    sky/background shot noise, detector read noise, dark current, and
    calibration effects. These effects are not included here.

    The input image must contain values in the range ``[0, 1]``. The output
    is clipped to the same range.

    Examples
    --------
    For a normalized image and ``intensity=1000``:

        image = 0.5

    corresponds to an expected 500 detected photons. The observed count is
    sampled as:

        observed_count ~ Poisson(500)

    and then normalized back to approximately ``0.5``.
    """
    def __init__(self, intensity):
        self.intensity = intensity

    def augment(self, image):
        self._validate_image_bounds(image)

        expected_counts = image * self.intensity
        noisy_counts = np.random.poisson(expected_counts)

        noisy_image = noisy_counts / self.intensity

        return np.clip(noisy_image, 0, 1)

    def _validate_image_bounds(self, image):
        if np.any(image < 0) or np.any(image > 1):
            raise ValueError("Image values should be in the range [0, 1]")
        