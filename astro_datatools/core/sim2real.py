"""
Turn simulated galaxy images into realistic survey observations.

The pipeline implemented by :class:`Sim2Real` is::

    sim --rebin--> rebinned --PSF--> convolved --crop/pad--> cropped
        --shot noise--> noisy --+ background--> final

Physics in brief
----------------
A simulation pixel has a fixed *physical* size (e.g. 200 pc), whereas a
telescope pixel has a fixed *angular* size (e.g. 0.1 arcsec for Euclid VIS).
The physical length subtended by one telescope pixel at redshift ``z`` is::

    target_pixel_kpc(z) = pixel_scale * D_A(z) = pixel_scale * kpc_proper_per_arcsec(z)

where ``D_A`` is the angular diameter distance of the chosen cosmology. The
simulated image is therefore resampled by ``sim_pixel_kpc / target_pixel_kpc(z)``
so that one output pixel covers one telescope pixel. Because ``D_A`` is not
monotonic, a galaxy covers the fewest pixels around ``z ~ 1.6`` and grows
again (in pixels) at higher redshift. At ``z = 0``, ``D_A = 0`` and a telescope
pixel subtends zero length (the galaxy would be infinitely large), so
``z <= 0`` is rejected.

Not modelled
------------
Only the *geometry* of observing at redshift ``z`` is modelled. The following
are intentionally **not** applied and should be handled upstream if needed:

- cosmological surface-brightness dimming ``(1 + z)^-4``,
- k-corrections / bandpass shifting and evolution of the stellar populations,
- conversion of the simulated pixel values to calibrated physical units. If
  the input is already calibrated (e.g. flux in the units of the background
  mosaic), pass ``conserve="flux"``, ``clip=None``, ``background_sigma=None``
  and the effective gain as ``shot_noise_intensity``; otherwise the shot noise
  and the optional background rescaling are in arbitrary "display" units.
"""
from typing import Dict, Optional, Sequence, Tuple, Union

import astropy.units as u
import numpy as np
from astropy.cosmology import Cosmology, Planck18
from astropy.stats import sigma_clipped_stats

from ..augment import ConvolveAugment, ShotNoiseAugment
from ..augment.noise import normalise_correlation_kernel
from ..logger import setup_logging
from ..transforms.embed import embed_in_background
from ..transforms.geometric import crop_or_pad, rebin


#: Keys of the dict returned by :meth:`Sim2Real.run` with ``return_stages=True``, in order.
STAGE_NAMES = ("sim", "rebinned", "convolved", "cropped", "noisy", "background", "final")


class Sim2Real:
    """
    Convert simulated galaxy images into realistic survey-like observations.

    The steps, in order, are:

    1. **Rebin** the simulated image from its physical pixel size to the
       telescope pixel size at redshift ``z`` using exact area-weighted
       rebinning (:func:`astro_datatools.transforms.geometric.rebin`), not
       interpolation (which aliases when shrinking by large factors). The
       rebinned size per axis is
       ``max(1, round(n * sim_pixel_kpc / target_pixel_kpc(z)))``; because of
       this rounding the effective output pixel differs slightly from the
       target (by at most half a pixel over the full image extent).
    2. **Convolve** with the PSF (:class:`~astro_datatools.augment.ConvolveAugment`).
       The PSF must be sampled at the telescope pixel scale (e.g. for Euclid
       VIS, 1 PSF pixel = 1 VIS pixel = 0.1 arcsec). Convolution happens
       before cropping so light from outside the final field is spread in.
    3. **Crop or pad** to ``output_size`` without resampling: centre crop when
       the rebinned image is larger, symmetric zero padding when smaller.
    4. **Shot noise** (:class:`~astro_datatools.augment.ShotNoiseAugment`),
       skipped when ``shot_noise_intensity`` is ``None``. Images within
       ``[0, 1]`` get the normalized (clipped) behaviour; any other image is
       treated as physical units and gets unclipped Poisson noise with
       variance ``value / shot_noise_intensity``.
    5. **Background embedding**
       (:func:`~astro_datatools.transforms.embed.embed_in_background`) into a
       real sky patch, skipped when no background is given. The patch can be
       rescaled so its noise level equals ``background_sigma``.

    The final image is returned as ``float32`` and is not clipped (real sky
    noise is negative in places). See the module docstring for the effects
    that are *not* modelled (surface-brightness dimming, k-corrections, noise
    in calibrated units).

    Images are channel-first: ``(C, H, W)``, a batch ``(B, C, H, W)``, or a
    single-band ``(H, W)`` image.

    :param sim_pixel_size: Physical (proper, not comoving) size of one
        simulation pixel, as a length Quantity, e.g. ``0.2 * u.kpc``.
    :type sim_pixel_size: astropy.units.Quantity
    :param pixel_scale: Angular size of one telescope pixel, as an angle
        Quantity, e.g. ``0.1 * u.arcsec`` for Euclid VIS. The PSF passed to
        :meth:`run` must be sampled at this scale.
    :type pixel_scale: astropy.units.Quantity
    :param output_size: Side length in pixels of the (square) output image.
    :type output_size: int
    :param cosmology: Cosmology used for the angular diameter distance. Should
        be the cosmology of the simulation. Defaults to
        :data:`astropy.cosmology.Planck18`.
    :type cosmology: astropy.cosmology.Cosmology | None
    :param conserve: Rebinning mode, ``"surface_brightness"`` (default,
        preserves the mean pixel value; use for image-like pixel values) or
        ``"flux"`` (preserves the total sum; use for flux maps).
    :type conserve: str
    :param normalise_psf: If ``True`` (default), each PSF (per channel) is
        normalised to unit sum before convolution, so convolution preserves
        total light.
    :type normalise_psf: bool
    :param clip: Clipping applied right after convolution:

        - ``"auto"`` (default): clip to ``[0, 1]`` only if the rebinned image
          (before convolution) lies within ``[0, 1]``. With a unit-sum,
          non-negative PSF the convolution cannot leave that range, so this
          only removes small excursions caused by negative PSF pixels (common
          in PSFEx models), which :class:`ShotNoiseAugment` would reject.
          Otherwise (e.g. flux maps) nothing is clipped.
        - ``(low, high)``: always clip to this range.
        - ``None``: never clip.
    :type clip: str | tuple[float, float] | None
    :param shot_noise_intensity: Expected photon count represented by a pixel
        value of 1 (see :class:`ShotNoiseAugment`). For calibrated images this
        is the effective gain in detected photoelectrons per image unit (e.g.
        :meth:`astro_datatools.surveys.EuclidSurvey.estimate_gain`). ``None``
        (default) skips the shot noise step.
    :type shot_noise_intensity: float | None
    :param embed_type: Embedding mode for the background, ``"default"`` (sum)
        or ``"fill"`` (background replaced wherever the image is non-zero).
    :type embed_type: str
    :param background_sigma: If set, each background patch (per channel) is
        divided by its noise level and multiplied by this value, i.e. the sky
        noise is rescaled to ``background_sigma`` in the units of the
        simulated image (which are display units in ``[0, 1]``, while real
        sky is in calibrated units). A float or one value per channel.
        ``None`` (default) uses the background as given.
    :type background_sigma: float | Sequence[float] | None
    :param seed: Seed for the shot noise. :class:`ShotNoiseAugment` draws from
        NumPy's *global* RNG. If ``seed`` is ``None`` (default) the global RNG
        is used as is (so ``np.random.seed`` controls it). If a seed is given,
        a private random stream is used instead: the global RNG state is
        saved, swapped for the private stream while the noise is drawn, and
        restored afterwards, so the global RNG is never reseeded or advanced.
        Successive calls continue the private stream (reproducible sequence).
        This swap is not thread-safe.
    :type seed: int | None
    :param noise_correlation_kernel: Optional 2D kernel that spatially
        correlates the shot noise like the resampled noise of a mosaic, e.g.
        from :meth:`astro_datatools.surveys.EmptyRegions.noise_correlation_kernel`
        (see ``ShotNoiseAugment(correlation_kernel=...)``). ``None`` (default)
        gives independent per-pixel noise.
    :type noise_correlation_kernel: numpy.ndarray | None
    """

    def __init__(
        self,
        sim_pixel_size: u.Quantity,
        pixel_scale: u.Quantity,
        output_size: int,
        *,
        cosmology: Optional[Cosmology] = None,
        conserve: str = "surface_brightness",
        normalise_psf: bool = True,
        clip: Union[str, Tuple[float, float], None] = "auto",
        shot_noise_intensity: Optional[float] = None,
        embed_type: str = "default",
        background_sigma: Optional[Union[float, Sequence[float]]] = None,
        seed: Optional[int] = None,
        noise_correlation_kernel: Optional[np.ndarray] = None,
    ):
        self.logger = setup_logging(name=f"astro_datatools.core.{self.__class__.__name__}")

        self.sim_pixel_size = _as_quantity(sim_pixel_size, u.kpc, "sim_pixel_size", "length")
        self.pixel_scale = _as_quantity(pixel_scale, u.arcsec, "pixel_scale", "angle")
        if self.sim_pixel_size.value <= 0 or self.pixel_scale.value <= 0:
            raise ValueError("sim_pixel_size and pixel_scale must be positive.")

        if int(output_size) != output_size or output_size < 1:
            raise ValueError(f"output_size must be a positive integer, got {output_size!r}")
        self.output_size = int(output_size)

        self.cosmology = Planck18 if cosmology is None else cosmology

        if conserve not in ("surface_brightness", "flux"):
            raise ValueError(
                f"Unknown conserve mode {conserve!r}; expected 'surface_brightness' or 'flux'"
            )
        self.conserve = conserve
        self.normalise_psf = normalise_psf

        if isinstance(clip, (tuple, list)) and len(clip) == 2:
            clip = (float(clip[0]), float(clip[1]))
        elif not (clip is None or clip == "auto"):
            raise ValueError(f"clip must be 'auto', None or a (low, high) tuple, got {clip!r}")
        self.clip = clip

        if shot_noise_intensity is not None and shot_noise_intensity <= 0:
            raise ValueError("shot_noise_intensity must be positive or None.")
        self.shot_noise_intensity = shot_noise_intensity

        if embed_type not in ("default", "fill"):
            raise ValueError(f"Unsupported embed_type {embed_type!r}; expected 'default' or 'fill'")
        self.embed_type = embed_type
        self.background_sigma = background_sigma

        self.seed = seed
        self._rng_state = None if seed is None else np.random.RandomState(seed).get_state()
        self.noise_correlation_kernel = (
            None if noise_correlation_kernel is None else normalise_correlation_kernel(noise_correlation_kernel)
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def target_pixel_kpc(self, z: float) -> u.Quantity:
        """
        Physical (proper) length subtended by one telescope pixel at redshift ``z``.

        Computed as ``pixel_scale * cosmology.kpc_proper_per_arcmin(z)``, i.e.
        the pixel angle times the angular diameter distance.

        :param z: Redshift, must be > 0.
        :type z: float
        :return: Pixel size in kpc.
        :rtype: astropy.units.Quantity
        :raises ValueError: If ``z <= 0`` (the angular diameter distance is zero).
        """
        z = _check_redshift(z)
        per_arcsec = self.cosmology.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec)
        return (per_arcsec * self.pixel_scale).to(u.kpc)

    def rebin_factor(self, z: float) -> float:
        """
        Ratio of simulation pixel size to telescope pixel size at redshift ``z``.

        Values below one mean the image shrinks (in pixels) when observed.

        :param z: Redshift, must be > 0.
        :type z: float
        :return: ``sim_pixel_kpc / target_pixel_kpc(z)``.
        :rtype: float
        """
        return float((self.sim_pixel_size / self.target_pixel_kpc(z)).to_value(u.dimensionless_unscaled))

    def rebinned_shape(self, input_shape: Tuple[int, ...], z: float) -> Tuple[int, int]:
        """
        Spatial shape of the rebinned image (before crop/pad) at redshift ``z``.

        Each axis becomes ``max(1, round(n * rebin_factor(z)))``. If this is
        larger than ``output_size`` the image is cropped, otherwise padded.

        :param input_shape: Shape of the simulated image; only the last two
            entries (height, width) are used, so ``image.shape`` can be passed.
        :type input_shape: tuple[int, ...]
        :param z: Redshift, must be > 0.
        :type z: float
        :return: ``(height, width)`` of the rebinned image.
        :rtype: tuple[int, int]
        """
        factor = self.rebin_factor(z)
        ny, nx = tuple(input_shape)[-2:]
        return (max(1, int(round(ny * factor))), max(1, int(round(nx * factor))))

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def __call__(
        self,
        image: np.ndarray,
        z: float,
        psf: np.ndarray,
        background: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Shortcut for :meth:`run` that returns only the final image."""
        kwargs["return_stages"] = False
        return self.run(image, z, psf, background, **kwargs)

    def run(
        self,
        image: np.ndarray,
        z: float,
        psf: np.ndarray,
        background: Optional[np.ndarray] = None,
        *,
        background_noise: Optional[Union[float, np.ndarray]] = None,
        seed: Optional[int] = None,
        return_stages: bool = False,
    ) -> Union[np.ndarray, Dict[str, Optional[np.ndarray]]]:
        """
        Observe a simulated image at redshift ``z``.

        :param image: Simulated image, ``(C, H, W)``, ``(B, C, H, W)`` or
            ``(H, W)``, with pixels of size ``sim_pixel_size``.
        :type image: np.ndarray
        :param z: Redshift at which the galaxy is observed, must be > 0.
        :type z: float
        :param psf: PSF sampled at ``pixel_scale``: ``(h, w)`` shared by all
            channels, or ``(C, h, w)`` with one PSF per channel. Odd kernel
            sizes keep the image centred exactly.
        :type psf: np.ndarray
        :param background: Real sky patch of spatial size
            ``output_size x output_size``: ``(S, S)`` (used for every channel),
            ``(C, S, S)``, or ``(B, C, S, S)`` for batched input. ``None``
            skips the embedding step.
        :type background: np.ndarray | None
        :param background_noise: Noise level (sigma) of the background patch,
            used only when ``background_sigma`` is set. A float or an array
            broadcastable to the background's non-spatial shape (e.g. one value
            per channel). If ``None``, it is estimated per channel with
            :func:`astropy.stats.sigma_clipped_stats`.
        :type background_noise: float | np.ndarray | None
        :param seed: Seed for this call's shot noise, overriding the
            constructor ``seed`` (the global RNG is not modified, see the
            class docstring).
        :type seed: int | None
        :param return_stages: If ``True``, return a dict with every
            intermediate stage instead of only the final image.
        :type return_stages: bool
        :return: Final ``float32`` image of shape ``(..., output_size, output_size)``,
            or, with ``return_stages=True``, a dict with keys ``"sim"``,
            ``"rebinned"``, ``"convolved"``, ``"cropped"``, ``"noisy"``,
            ``"background"`` and ``"final"`` (in pipeline order, see
            :data:`STAGE_NAMES`). Skipped steps (``"noisy"``, ``"background"``)
            are ``None``. ``"convolved"`` is after clipping; ``"background"``
            is the (rescaled) patch as it was added.
        :rtype: np.ndarray | dict[str, np.ndarray | None]
        :raises ValueError: For ``z <= 0``, unsupported shapes, PSF/background
            mismatches.
        """
        image = np.asarray(image)
        if image.ndim not in (2, 3, 4):
            raise ValueError(
                f"Expected image of shape (H, W), (C, H, W) or (B, C, H, W), got {image.shape}"
            )
        squeeze = image.ndim == 2
        work = image[None] if squeeze else image
        channels = work.shape[-3]
        stages: Dict[str, Optional[np.ndarray]] = {name: None for name in STAGE_NAMES}
        stages["sim"] = image

        # 1. Rebin to the telescope pixel scale at redshift z.
        out_shape = self.rebinned_shape(work.shape, z)
        self.logger.debug(
            "z=%s: %s px -> %s px (target pixel %.4f kpc)",
            z, work.shape[-2:], out_shape, self.target_pixel_kpc(z).value,
        )
        rebinned = rebin(work.astype(np.float64, copy=False), out_shape, conserve=self.conserve)
        stages["rebinned"] = rebinned

        # 2. PSF convolution at the native PSF sampling.
        psf_stack = self._prepare_psf(psf, channels)
        convolved = ConvolveAugment(psf_kernel=psf_stack).augment(rebinned)
        clip_range = self._clip_range(rebinned)
        if clip_range is not None:
            convolved = np.clip(convolved, *clip_range)
        stages["convolved"] = convolved

        # 3. Crop or pad to the output size (no resampling).
        current = crop_or_pad(convolved, self.output_size)
        stages["cropped"] = current

        # 4. Shot noise.
        if self.shot_noise_intensity is not None:
            current = self._shot_noise(current, seed)
            stages["noisy"] = current

        # 5. Background embedding.
        if background is not None:
            bg = self._prepare_background(background, current.shape, channels, background_noise)
            stages["background"] = bg
            current = embed_in_background(current, bg, embed_type=self.embed_type)

        stages["final"] = current.astype(np.float32)

        if squeeze:
            for name in STAGE_NAMES[1:]:
                if stages[name] is not None:
                    stages[name] = stages[name][0]

        return stages if return_stages else stages["final"]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prepare_psf(self, psf: np.ndarray, channels: int) -> np.ndarray:
        """Return a (C, h, w) PSF stack, normalised to unit sum per channel if requested."""
        psf = np.asarray(psf, dtype=np.float64)
        if psf.ndim == 2:
            # One copy per channel: avoids ConvolveAugment's shared-PSF warning.
            psf = np.repeat(psf[None], channels, axis=0)
        elif psf.ndim == 3:
            if psf.shape[0] != channels:
                raise ValueError(f"PSF has {psf.shape[0]} channels but the image has {channels}.")
        else:
            raise ValueError(f"PSF must have shape (h, w) or (C, h, w), got {psf.shape}")

        if self.normalise_psf:
            sums = psf.sum(axis=(-2, -1), keepdims=True)
            if np.any(sums == 0):
                raise ValueError("PSF sums to zero and cannot be normalised.")
            psf = psf / sums
        return psf

    def _clip_range(self, rebinned: np.ndarray) -> Optional[Tuple[float, float]]:
        if self.clip is None:
            return None
        if self.clip == "auto":
            if rebinned.min() >= 0 and rebinned.max() <= 1:
                return (0.0, 1.0)
            return None
        return self.clip

    def _shot_noise(self, image: np.ndarray, seed: Optional[int]) -> np.ndarray:
        # Normalized [0, 1] images keep ShotNoiseAugment's clipped behaviour; anything else is
        # treated as physical units (e.g. calibrated flux) and gets unclipped, zero-mean noise.
        in_unit_range = image.min() >= 0 and image.max() <= 1
        augment = ShotNoiseAugment(
            intensity=self.shot_noise_intensity,
            bounds=(0.0, 1.0) if in_unit_range else None,
            correlation_kernel=self.noise_correlation_kernel,
        )

        if seed is not None:
            private_state = np.random.RandomState(seed).get_state()
        elif self._rng_state is not None:
            private_state = self._rng_state
        else:
            return augment.augment(image)

        # Draw from a private stream without disturbing the global RNG.
        saved = np.random.get_state()
        try:
            np.random.set_state(private_state)
            noisy = augment.augment(image)
            if seed is None:
                self._rng_state = np.random.get_state()
        finally:
            np.random.set_state(saved)
        return noisy

    def _prepare_background(
        self,
        background: np.ndarray,
        image_shape: Tuple[int, ...],
        channels: int,
        background_noise: Optional[Union[float, np.ndarray]],
    ) -> np.ndarray:
        """Validate, optionally rescale, and broadcast the background to the image shape."""
        bg = np.asarray(background, dtype=np.float64)
        size = self.output_size
        if bg.ndim < 2 or bg.shape[-2:] != (size, size):
            raise ValueError(
                f"Background spatial shape {bg.shape[-2:]} does not match output_size "
                f"({size}, {size}). Cut the background patch to the output size."
            )
        if bg.ndim == 2:
            bg = bg[None]
        if bg.shape[-3] not in (1, channels):
            raise ValueError(f"Background has {bg.shape[-3]} channels but the image has {channels}.")
        if bg.ndim > len(image_shape):
            raise ValueError(
                f"Background shape {bg.shape} is incompatible with image shape {image_shape}."
            )

        if self.background_sigma is not None:
            if background_noise is None:
                _, _, noise = sigma_clipped_stats(bg, axis=(-2, -1))
            else:
                noise = background_noise
            noise = np.asarray(noise, dtype=np.float64)
            if np.any(~np.isfinite(noise)) or np.any(noise <= 0):
                raise ValueError(f"Background noise must be positive and finite, got {noise}.")
            target = np.asarray(self.background_sigma, dtype=np.float64)
            bg = bg * (target[..., None, None] / noise[..., None, None])

        try:
            return np.broadcast_to(bg, image_shape).copy()
        except ValueError:
            raise ValueError(
                f"Background shape {bg.shape} cannot be broadcast to image shape {image_shape}."
            ) from None


def _check_redshift(z: float) -> float:
    z = float(z)
    if not np.isfinite(z) or z <= 0:
        raise ValueError(
            f"Redshift must be > 0, got z={z}. At z=0 the angular diameter distance is 0, "
            "so a telescope pixel subtends zero physical length and the galaxy would be "
            "infinitely large on the detector."
        )
    return z


def _as_quantity(value, unit: u.UnitBase, name: str, kind: str) -> u.Quantity:
    if not isinstance(value, u.Quantity):
        raise TypeError(
            f"{name} must be an astropy Quantity with {kind} units (e.g. 1.0 * u.{unit}), got {value!r}"
        )
    try:
        return value.to(unit)
    except u.UnitConversionError:
        raise u.UnitConversionError(f"{name} must have {kind} units, got {value.unit}") from None
