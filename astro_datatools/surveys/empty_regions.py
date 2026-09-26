"""Find source-free ("empty") sky regions in survey images, e.g. for realistic backgrounds."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import gaussian_filter


@dataclass
class EmptyRegions:
    """Result of :func:`find_empty_regions`.

    :param patches: ``(n, size, size)`` float32 cut-outs with the global sky median subtracted.
    :param source_fractions: ``(n,)`` fraction of pixels flagged as source in each patch.
    :param corners: ``(n, 2)`` integer ``(top, left)`` pixel corners in the input image.
    :param size: Patch side length in pixels.
    :param sky_median: Sigma-clipped median of the valid input pixels (subtracted from patches).
    :param sky_sigma: Sigma-clipped standard deviation of the valid input pixels.
    :param source_mask: Boolean source mask of the full input image.
    :param n_candidates: Number of fully valid windows that were considered.
    :param wcs: WCS of the input image, if known (set by survey helpers).
    :param meta: Extra metadata (survey, band, BUNIT, ...).
    """

    patches: np.ndarray
    source_fractions: np.ndarray
    corners: np.ndarray
    size: int
    sky_median: float
    sky_sigma: float
    source_mask: np.ndarray
    n_candidates: int
    wcs: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.patches)

    @property
    def centers_pix(self) -> np.ndarray:
        """``(n, 2)`` array of patch centres as ``(x, y)`` pixel coordinates (0-based)."""
        half = (self.size - 1) / 2.0
        return np.stack([self.corners[:, 1] + half, self.corners[:, 0] + half], axis=1)

    def patch_masks(self, dilate: int = 3) -> np.ndarray:
        """``(n, size, size)`` boolean masks, ``True`` on source-free pixels of each patch.

        :param dilate: Number of pixels by which detected sources are grown, to exclude their faint wings.
        """
        from scipy.ndimage import binary_dilation

        masks = []
        for top, left in self.corners:
            src = self.source_mask[top:top + self.size, left:left + self.size]
            if dilate > 0:
                src = binary_dilation(src, iterations=dilate)
            masks.append(~src)
        return np.stack(masks)

    def noise_correlation_kernel(self, max_lag: int = 1, size: int = 5, dilate: int = 3, **kwargs):
        """Estimate the sky's pixel-to-pixel noise correlation as a kernel for correlated shot noise.

        Uses the source-free pixels of all patches (see
        :func:`astro_datatools.augment.noise.estimate_correlation_kernel`); pass the kernel as
        ``ShotNoiseAugment(correlation_kernel=...)`` or ``Sim2Real(noise_correlation_kernel=...)``.

        :param max_lag: Largest lag modelled (1 = direct and diagonal neighbours).
        :param size: Odd side length of the kernel.
        :param dilate: Source-mask dilation in pixels.
        :param kwargs: Passed to :func:`~astro_datatools.augment.noise.estimate_correlation_kernel`.
        :return: ``(kernel, acf)``, see :func:`~astro_datatools.augment.noise.estimate_correlation_kernel`.
        """
        from ..augment.noise import estimate_correlation_kernel

        return estimate_correlation_kernel(
            self.patches, self.patch_masks(dilate), max_lag=max_lag, size=size, **kwargs
        )

    def centers_world(self, wcs: Any = None):
        """Patch centres as a :class:`~astropy.coordinates.SkyCoord` array.

        :param wcs: WCS to use (defaults to :attr:`wcs`).
        """
        wcs = wcs if wcs is not None else self.wcs
        if wcs is None:
            raise ValueError("No WCS available; pass one explicitly.")
        c = self.centers_pix
        return wcs.pixel_to_world(c[:, 0], c[:, 1])


def _window_sums(arr: np.ndarray, size: int, tops: np.ndarray, lefts: np.ndarray) -> np.ndarray:
    """Sum of ``arr`` over every ``size`` x ``size`` window at the grid ``tops`` x ``lefts``."""
    integral = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(arr, axis=0, dtype=np.float64), axis=1)
    t = tops[:, None]
    l = lefts[None, :]
    return (
        integral[t + size, l + size]
        - integral[t, l + size]
        - integral[t + size, l]
        + integral[t, l]
    )


def find_empty_regions(
    image: np.ndarray,
    size: int,
    n: int,
    nsigma: float = 5.0,
    smoothing_sigma: float = 1.0,
    stride: Optional[int] = None,
    max_source_fraction: Optional[float] = None,
) -> EmptyRegions:
    """Select the ``n`` least source-contaminated, non-overlapping square windows of an image.

    Algorithm:

    1. Valid pixels are finite and non-zero (zero = no coverage in most mosaics).
    2. Sky median and sigma are estimated with 3-sigma clipped statistics.
    3. The median-subtracted image (invalid pixels set to 0) is smoothed with a
       Gaussian of ``smoothing_sigma`` pixels, and its noise is re-estimated the
       same way. Pixels with ``smoothed > nsigma * sigma_smoothed`` form the
       source mask.
    4. ``size`` x ``size`` windows on a grid with step ``stride`` are scored by
       the fraction of source pixels; windows touching any invalid pixel are
       rejected. Ties are broken by the smoothed flux in the window.
    5. The ``n`` lowest-scoring windows are chosen greedily so that none overlap.

    .. note::
       At Euclid depth a 20-40 arcsec window is almost never truly empty (faint
       galaxies, stellar halos). This function therefore returns the *least*
       contaminated windows and reports their contamination in
       ``source_fractions``; use ``max_source_fraction`` to enforce a hard cut.

    :param image: 2D image array.
    :param size: Window side length in pixels.
    :param n: Number of windows to return.
    :param nsigma: Detection threshold in units of the smoothed-image sigma.
    :param smoothing_sigma: Gaussian smoothing sigma in pixels (0 disables smoothing).
    :param stride: Grid step in pixels (default ``max(1, size // 4)``).
    :param max_source_fraction: If given, windows with a larger source fraction are rejected.
    :return: Selected patches and metadata.
    :rtype: EmptyRegions
    :raises ValueError: If the inputs are invalid or fewer than ``n`` acceptable,
        non-overlapping windows exist.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}.")
    size = int(size)
    n = int(n)
    if size < 1 or n < 1:
        raise ValueError("size and n must be positive integers.")
    ny, nx = image.shape
    if size > ny or size > nx:
        raise ValueError(f"Window size {size} is larger than the image {image.shape}.")
    stride = max(1, size // 4) if stride is None else int(stride)
    if stride < 1:
        raise ValueError("stride must be >= 1.")

    valid = np.isfinite(image) & (image != 0)
    if valid.sum() < 10:
        raise ValueError("Image has too few valid (finite, non-zero) pixels.")

    sky_median, _, sky_sigma = sigma_clipped_stats(image[valid], sigma=3)
    residual = np.where(valid, image - sky_median, 0.0)
    smoothed = gaussian_filter(residual, smoothing_sigma) if smoothing_sigma > 0 else residual
    _, sm_median, sm_sigma = sigma_clipped_stats(smoothed[valid], sigma=3)
    source_mask = ((smoothed - sm_median) > nsigma * sm_sigma) & valid

    tops = np.arange(0, ny - size + 1, stride)
    lefts = np.arange(0, nx - size + 1, stride)
    n_invalid = _window_sums((~valid).astype(np.float64), size, tops, lefts)
    n_source = _window_sums(source_mask.astype(np.float64), size, tops, lefts)
    flux = _window_sums(smoothed, size, tops, lefts)

    score = n_source / float(size * size)
    ok = n_invalid < 0.5
    if max_source_fraction is not None:
        ok &= score <= max_source_fraction + 1e-12
    ti, li = np.nonzero(ok)
    n_candidates = len(ti)

    cand_score = score[ti, li]
    cand_flux = flux[ti, li]
    order = np.lexsort((cand_flux, cand_score))  # primary: score, secondary: flux

    selected: List[tuple] = []
    for idx in order:
        top, left = int(tops[ti[idx]]), int(lefts[li[idx]])
        if all(abs(top - t) >= size or abs(left - l) >= size for t, l, _ in selected):
            selected.append((top, left, float(cand_score[idx])))
            if len(selected) == n:
                break

    if len(selected) < n:
        cut = f" with source fraction <= {max_source_fraction}" if max_source_fraction is not None else ""
        raise ValueError(
            f"Requested {n} non-overlapping {size}x{size} windows but only {len(selected)} "
            f"could be found ({n_candidates} fully valid candidate windows{cut}, stride={stride}). "
            "Use a larger image, a smaller size or n, or relax max_source_fraction."
        )

    corners = np.array([(t, l) for t, l, _ in selected], dtype=int)
    fractions = np.array([s for _, _, s in selected], dtype=np.float64)
    patches = np.stack(
        [image[t:t + size, l:l + size] - sky_median for t, l in corners]
    ).astype(np.float32)

    return EmptyRegions(
        patches=patches,
        source_fractions=fractions,
        corners=corners,
        size=size,
        sky_median=float(sky_median),
        sky_sigma=float(sky_sigma),
        source_mask=source_mask,
        n_candidates=n_candidates,
    )
