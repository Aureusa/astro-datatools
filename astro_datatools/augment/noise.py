from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import convolve

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

        For images in physical units this is the effective gain: detected
        photoelectrons per image unit (e.g. per mosaic count).
    bounds : tuple of float or None, default ``(0, 1)``
        With ``(low, high)`` the input must lie within the range and the
        output is clipped to it (the original behaviour, for normalized
        images). With ``None`` the image may have any scale (e.g. calibrated
        flux): zero-mean Poisson noise with variance ``image / intensity`` is
        added, pixels ``<= 0`` get no noise (they have no expected photons),
        and nothing is clipped, so e.g. slightly negative PSF-ringing pixels
        keep their values.
    correlation_kernel : numpy.ndarray or None, default ``None``
        Optional 2D kernel that makes the noise spatially correlated, e.g. to
        match the pixel-to-pixel correlation that resampling introduces in
        stacked/mosaicked survey images (see :func:`estimate_correlation_kernel`).
        The zero-mean Poisson noise field is convolved with the kernel,
        normalised to unit sum of squares so the per-pixel noise variance is
        preserved (exactly for uniform noise, approximately where the variance
        changes quickly). Because the noise is spread over the kernel
        footprint, pixels next to a source can receive a little noise even if
        they have no expected photons themselves. ``None`` keeps independent
        (white) noise.

    Notes
    -----
    This augmentation models photon-counting (shot) noise only. Real
    astronomical observations can contain additional noise sources, such as
    sky/background shot noise, detector read noise, dark current, and
    calibration effects. These effects are not included here.

    With the default ``bounds`` the input image must contain values in the
    range ``[0, 1]`` and the output is clipped to the same range.

    Examples
    --------
    For a normalized image and ``intensity=1000``:

        image = 0.5

    corresponds to an expected 500 detected photons. The observed count is
    sampled as:

        observed_count ~ Poisson(500)

    and then normalized back to approximately ``0.5``.
    """
    def __init__(self, intensity, bounds=(0.0, 1.0), correlation_kernel=None):
        if intensity <= 0:
            raise ValueError("intensity must be positive")
        self.intensity = intensity
        self.bounds = None if bounds is None else (float(bounds[0]), float(bounds[1]))
        self.correlation_kernel = (
            None if correlation_kernel is None else normalise_correlation_kernel(correlation_kernel)
        )

    def augment(self, image):
        if self.bounds is not None:
            self._validate_image_bounds(image)

        if self.correlation_kernel is None:
            if self.bounds is None:
                expected_counts = np.clip(image, 0, None) * self.intensity
                noise = (np.random.poisson(expected_counts) - expected_counts) / self.intensity
                return image + noise

            expected_counts = image * self.intensity
            noisy_counts = np.random.poisson(expected_counts)

            noisy_image = noisy_counts / self.intensity

            return np.clip(noisy_image, *self.bounds)

        expected_counts = np.clip(image, 0, None) * self.intensity
        noise = (np.random.poisson(expected_counts) - expected_counts) / self.intensity
        noise = _convolve_spatial(noise, self.correlation_kernel)
        noisy_image = image + noise
        if self.bounds is None:
            return noisy_image
        return np.clip(noisy_image, *self.bounds)

    def _validate_image_bounds(self, image):
        low, high = self.bounds
        if np.any(image < low) or np.any(image > high):
            raise ValueError(f"Image values should be in the range [{low:g}, {high:g}]")


def normalise_correlation_kernel(kernel: np.ndarray) -> np.ndarray:
    """
    Scale a noise-correlation kernel to unit sum of squares.

    Convolving white noise of variance ``v`` with such a kernel yields
    correlated noise that still has per-pixel variance ``v``.

    :param kernel: 2D kernel.
    :return: Normalised float64 copy.
    """
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2:
        raise ValueError(f"Correlation kernel must be 2D, got shape {kernel.shape}")
    norm = np.sqrt((kernel**2).sum())
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Correlation kernel must have a non-zero, finite sum of squares.")
    return kernel / norm


def noise_autocorrelation(
    noise: np.ndarray,
    mask: Optional[np.ndarray] = None,
    max_lag: int = 5,
    clip_sigma: Optional[float] = 5.0,
) -> np.ndarray:
    """
    Measure the normalised spatial autocorrelation of a noise image.

    Pixels flagged ``False`` in ``mask`` (sources, bad data) and non-finite
    pixels are ignored, as are pixels deviating by more than ``clip_sigma``
    times the (robust) noise level. Several images of the same noise field
    (e.g. a stack of sky patches) can be passed as a ``(n, H, W)`` array; they
    are combined.

    :param noise: ``(H, W)`` or ``(n, H, W)`` noise image(s), e.g. source-free sky.
    :param mask: Boolean array of the same shape, ``True`` where pixels may be used.
    :param max_lag: Largest pixel lag measured along each axis.
    :param clip_sigma: Outlier rejection threshold, or ``None`` to disable.
    :return: ``(2 * max_lag + 1, 2 * max_lag + 1)`` array, value 1 at the centre
        (zero lag); entry ``[max_lag + dy, max_lag + dx]`` is the correlation
        between pixels separated by ``(dy, dx)``.
    """
    noise = np.asarray(noise, dtype=np.float64)
    if noise.ndim == 2:
        noise = noise[None]
    if noise.ndim != 3:
        raise ValueError(f"Expected (H, W) or (n, H, W) noise, got shape {noise.shape}")
    good = np.isfinite(noise)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        good &= mask if mask.ndim == 3 else mask[None]
    if clip_sigma is not None:
        values = noise[good]
        med = np.median(values)
        robust_sigma = 1.4826 * np.median(np.abs(values - med))
        good &= np.abs(noise - med) <= clip_sigma * robust_sigma

    L = int(max_lag)
    num = np.zeros((2 * L + 1, 2 * L + 1))
    cnt = np.zeros((2 * L + 1, 2 * L + 1))
    for img, g in zip(noise, good):
        x = np.where(g, img - img[g].mean(), 0.0)
        gf = g.astype(np.float64)
        h, w = x.shape
        for dy in range(-L, L + 1):
            for dx in range(-L, L + 1):
                a = np.s_[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)]
                b = np.s_[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
                num[L + dy, L + dx] += (x[a] * x[b]).sum()
                cnt[L + dy, L + dx] += (gf[a] * gf[b]).sum()
    if cnt[L, L] == 0:
        raise ValueError("No usable pixels to measure the autocorrelation.")
    acf = num / np.maximum(cnt, 1)
    return acf / acf[L, L]


def correlation_kernel_from_autocorrelation(acf: np.ndarray, size: int = 5, grid: int = 64) -> np.ndarray:
    """
    Build a kernel whose self-convolution reproduces a target autocorrelation.

    White noise convolved with kernel ``K`` has autocorrelation ``K * K``
    (``K`` correlated with itself). ``K`` is taken as the zero-phase
    "square root" of the target: the inverse Fourier transform of the square
    root of the target's power spectrum (negative spectral values, which
    measurement noise can produce, are set to zero). The result is cropped to
    ``size x size`` and normalised to unit sum of squares.

    :param acf: Odd-sized square autocorrelation, centre = zero lag (see
        :func:`noise_autocorrelation`); lags outside it are taken as zero.
    :param size: Odd side length of the returned kernel.
    :param grid: Size of the Fourier grid used internally.
    :return: ``(size, size)`` kernel with unit sum of squares.
    """
    acf = np.asarray(acf, dtype=np.float64)
    if acf.ndim != 2 or acf.shape[0] != acf.shape[1] or acf.shape[0] % 2 == 0:
        raise ValueError(f"acf must be a square array with odd side, got shape {acf.shape}")
    if size % 2 == 0 or size < 1:
        raise ValueError("size must be a positive odd integer")
    L = acf.shape[0] // 2
    grid = max(grid, 4 * (L + size))
    acf = 0.5 * (acf + acf[::-1, ::-1])  # enforce the symmetry of an autocorrelation

    target = np.zeros((grid, grid))
    c = grid // 2
    target[c - L:c + L + 1, c - L:c + L + 1] = acf
    power = np.clip(np.fft.fft2(np.fft.ifftshift(target)).real, 0, None)
    kernel_full = np.fft.fftshift(np.fft.ifft2(np.sqrt(power)).real)
    h = size // 2
    return normalise_correlation_kernel(kernel_full[c - h:c + h + 1, c - h:c + h + 1])


def estimate_correlation_kernel(
    noise: np.ndarray,
    mask: Optional[np.ndarray] = None,
    max_lag: int = 1,
    size: int = 5,
    floor_lags: Optional[Tuple[int, int]] = (3, 5),
    clip_sigma: Optional[float] = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate a noise-correlation kernel from source-free sky, for :class:`ShotNoiseAugment`.

    Resampling during stacking/mosaicking correlates the noise of
    neighbouring pixels. This measures that correlation on real sky and
    returns a kernel which, applied to white noise, reproduces it.

    Steps: measure the autocorrelation out to lag ``floor_lags[1]``; subtract
    the roughly constant "floor" seen at lags ``floor_lags[0]..floor_lags[1]``
    (caused by large-scale background residuals, not by pixel-to-pixel
    correlation) and renormalise; keep lags up to ``max_lag`` (1 = direct and
    diagonal neighbours); convert to a kernel with
    :func:`correlation_kernel_from_autocorrelation`.

    :param noise: ``(H, W)`` or ``(n, H, W)`` sky image(s) with sources masked via ``mask``.
    :param mask: Boolean array, ``True`` where pixels are source-free and valid.
    :param max_lag: Largest lag kept in the model.
    :param size: Odd side length of the kernel.
    :param floor_lags: ``(min, max)`` Chebyshev lags used to measure the floor, or ``None``.
    :param clip_sigma: Outlier rejection threshold passed to :func:`noise_autocorrelation`.
    :return: ``(kernel, acf)``: the ``(size, size)`` kernel and the measured
        (floor-corrected) autocorrelation out to ``max_lag`` that it models.
    """
    outer = max(max_lag, floor_lags[1] if floor_lags else max_lag)
    acf = noise_autocorrelation(noise, mask, max_lag=outer, clip_sigma=clip_sigma)
    L = outer
    if floor_lags:
        dy, dx = np.mgrid[-L:L + 1, -L:L + 1]
        cheb = np.maximum(np.abs(dy), np.abs(dx))
        ring = (cheb >= floor_lags[0]) & (cheb <= floor_lags[1])
        floor = float(np.median(acf[ring]))
        acf = (acf - floor) / (1 - floor)
    acf = acf[L - max_lag:L + max_lag + 1, L - max_lag:L + max_lag + 1]
    return correlation_kernel_from_autocorrelation(acf, size=size), acf


def kernel_autocorrelation(kernel: np.ndarray, max_lag: int = 2) -> np.ndarray:
    """
    Autocorrelation of white noise after convolution with ``kernel``.

    :param kernel: 2D kernel (normalised internally to unit sum of squares).
    :param max_lag: Largest lag returned along each axis.
    :return: ``(2 * max_lag + 1, 2 * max_lag + 1)`` array, 1 at the centre.
    """
    k = normalise_correlation_kernel(kernel)
    h, w = k.shape
    L = int(max_lag)
    out = np.zeros((2 * L + 1, 2 * L + 1))
    for dy in range(-L, L + 1):
        for dx in range(-L, L + 1):
            a = k[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)]
            b = k[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
            out[L + dy, L + dx] = (a * b).sum() if a.size else 0.0
    return out


def _convolve_spatial(noise: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve over the last two axes only (channels/batch are treated independently)."""
    noise = np.asarray(noise, dtype=np.float64)
    k = kernel.reshape((1,) * (noise.ndim - 2) + kernel.shape)
    return convolve(noise, k, mode="reflect")
