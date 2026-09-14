import numpy as np
from scipy.signal import convolve2d

from astro_datatools.logger import setup_logging
from .base import BaseAugment


class ConvolveAugment(BaseAugment):
    """
    Apply an astronomical point spread function (PSF) to simulated images.

    This augmentation is designed specifically for astronomical imaging, where
    the PSF describes the blurring introduced by the telescope optics,
    atmosphere, detector, and other instrumental effects.

    The PSF is applied independently to each image channel. A PSF may be
    provided in one of two forms:

    - 2D PSF with shape ``(kernel_height, kernel_width)``:
      The same PSF is applied independently to every channel of the image.

    - 3D PSF with shape ``(channels, kernel_height, kernel_width)``:
      A separate PSF is applied to each channel. In this case, channel ``c``
      of the image is convolved only with channel ``c`` of the PSF:

          output[c] = conv2d(image[c], psf[c])

    No convolution is performed across channels. Thus, a 3D PSF in this
    implementation does not represent a conventional 3D convolution kernel;
    its first dimension represents independent channel-specific PSFs.

    Supported image shapes are:

    - ``(height, width)``:
      A single-channel 2D image. Only a 2D PSF is supported.

    - ``(channels, height, width)``:
      A multi-channel image. A 2D PSF is applied to all channels, or a
      channel-specific 3D PSF is applied independently to each channel.

    - ``(batch, channels, height, width)``:
      A batch of multi-channel images. The same PSF is applied independently
      to every image in the batch. For a 3D PSF, the same channel-specific
      PSF is used for the corresponding channel of every image in the batch.

    Convolution is performed using ``scipy.signal.convolve2d`` with
    ``mode="same"``, so the output has the same spatial dimensions as the
    input image.

    Parameters
    ----------
    psf_kernel : numpy.ndarray
        The astronomical point spread function. Must have one of the
        following shapes:

        - ``(H, W)`` for a PSF shared across all channels.
        - ``(C, H, W)`` for independent, channel-specific PSFs.

        For a 3D PSF, ``C`` must match the number of channels in the input
        image.

    Notes
    -----
    The PSF is assumed to represent independent spatial blurring for each
    channel. Cross-channel mixing is intentionally not performed.

    For example, given an image with shape ``(3, 100, 100)`` and a PSF with
    shape ``(3, 3, 3)``, the operation is equivalent to:

        output[0] = convolve2d(image[0], psf_kernel[0], mode="same")
        output[1] = convolve2d(image[1], psf_kernel[1], mode="same")
        output[2] = convolve2d(image[2], psf_kernel[2], mode="same")

    This design reflects the use of channel-dependent PSFs in astronomical
    imaging, where different observational bands may have different optical
    characteristics.
    """

    def __init__(self, psf_kernel):
        self.logger = setup_logging(name=f"astro_datatools.augment.{self.__class__.__name__}")
        self.one_time_warning_send = False
        self.psf_kernel = psf_kernel

    def augment(self, image):
        return self.convolve(image)

    def convolve(self, image):
        if image.ndim == 2:
            # Image of shape (height, width)
            return self._convolve_2d(image)
        
        if image.ndim == 3:
            # Image of shape (channels, height, width)
            return self._convolve_3d(image)

        if image.ndim == 4:
            # Image of shape (batch, channels, height, width)
            return self._convolve_batched(image)

    def _convolve_3d(self, image):
        if self.psf_kernel.ndim == 2:
            if not self.one_time_warning_send:
                self.logger.warning(
                    "You are using the same PSF for all channels. "
                    "It is common that different channels have different optical characteristics. "
                    "If using the same PSF for all channels is not intended, please provide a separate PSF "
                    "for each channel in shape (channels, height, width). "
                    "Ignore this warning if you are using a batch of image with shape (batch, height, width). "
                    "However, for batch processing it is faster to use an images with a shape "
                    "(batch, channels, height, width). "
                    "Continuing with the same PSF for all channels. "
                )
                self.one_time_warning_send = True

            # Apply 2D convolution to each channel separately
            convolved = np.empty_like(image)
            for c in range(image.shape[0]):
                convolved[c] = convolve2d(
                    image[c],
                    self.psf_kernel,
                    mode='same'
                )

            return convolved
        elif self.psf_kernel.ndim == 3:
            if self.psf_kernel.shape[0] != image.shape[0]:
                raise ValueError(
                    "Number of PSF channels must match number of image channels."
                )

            convolved = np.empty_like(image)
            for c in range(image.shape[0]):
                convolved[c] = convolve2d(
                    image[c],
                    self.psf_kernel[c],
                    mode="same",
                )

            return convolved
        else:
            raise ValueError("PSF kernel must be either 2D or 3D for 3D images.")

    def _convolve_batched(self, image):
        batch, channels, height, width = image.shape

        if self.psf_kernel.ndim == 2:
            flattened = image.reshape(batch * channels, height, width)

            return self._convolve_3d(flattened).reshape(
                batch, channels, height, width
            )

        elif self.psf_kernel.ndim == 3:
            if self.psf_kernel.shape[0] != channels:
                raise ValueError(
                    "Number of PSF channels must match number of image channels."
                )

            convolved = np.empty_like(image)

            for b in range(batch):
                convolved[b] = self._convolve_3d(image[b])

            return convolved

        else:
            raise ValueError(
                "PSF kernel must be either 2D or 3D for batched images."
            )

    def _convolve_2d(self, image):
        if self.psf_kernel.ndim != 2:
            raise ValueError("PSF kernel must be 2D for 2D images.")
        return convolve2d(image, self.psf_kernel, mode='same')
