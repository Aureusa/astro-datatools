"""Base interfaces shared by all survey archive clients.

Every survey client (Euclid, and later e.g. LOFAR) subclasses :class:`BaseSurvey`
and returns the same container types, so downstream code can swap archives
without changes:

- :meth:`BaseSurvey.query_images` returns an :class:`astropy.table.Table` of
  image products covering a sky region.
- :meth:`BaseSurvey.get_cutout` returns a :class:`Cutout`.
"""
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.units import Quantity
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from astro_datatools.logger import setup_logging

CoordinateLike = Union[SkyCoord, Tuple[float, float], Sequence[float]]
AngleLike = Union[Quantity, float]


class SurveyError(RuntimeError):
    """Base error raised by survey clients."""


class SurveyAuthenticationError(SurveyError):
    """Raised when an archive operation requires the user to log in first."""


def to_skycoord(coordinate: CoordinateLike) -> SkyCoord:
    """Convert a coordinate-like object to a scalar ICRS :class:`SkyCoord`.

    :param coordinate: A ``SkyCoord`` or an ``(ra, dec)`` pair in degrees.
    :type coordinate: SkyCoord or tuple
    :return: ICRS sky coordinate.
    :rtype: SkyCoord
    """
    if isinstance(coordinate, SkyCoord):
        coord = coordinate
    else:
        ra, dec = coordinate
        coord = SkyCoord(ra, dec, unit="deg", frame="icrs")
    if not coord.isscalar:
        raise ValueError("Expected a single (scalar) sky coordinate.")
    return coord.icrs


def to_angle(value: AngleLike, default_unit: u.Unit = u.arcsec) -> Quantity:
    """Convert a float or Quantity to an angular Quantity.

    :param value: Angle as a Quantity, or a bare number interpreted in ``default_unit``.
    :param default_unit: Unit used for bare numbers (default: arcsec).
    :return: Angular quantity.
    :rtype: Quantity
    """
    q = value if isinstance(value, Quantity) else Quantity(value, default_unit)
    if q.unit.physical_type != "angle":
        raise ValueError(f"Expected an angular quantity, got unit '{q.unit}'.")
    return q


@dataclass
class Cutout:
    """Image cutout returned by every survey client.

    :param data: 2D image array.
    :param header: FITS header of the image HDU.
    :param wcs: Celestial WCS of the image.
    :param pixel_scale: Pixel scale as an angle per pixel (e.g. ``0.1 arcsec``).
    :param survey: Name of the survey the cutout came from.
    :param band: Band / filter / instrument identifier (survey-specific).
    :param path: Path of the FITS file on disk, if saved.
    :param meta: Extra survey-specific metadata (tile id, release, ...).
    """

    data: np.ndarray
    header: fits.Header
    wcs: WCS
    pixel_scale: Quantity
    survey: str
    band: Optional[str] = None
    path: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the image array."""
        return self.data.shape

    @property
    def unit(self) -> Optional[str]:
        """Value of the ``BUNIT`` header keyword, if present."""
        return self.header.get("BUNIT")

    @property
    def center(self) -> SkyCoord:
        """Sky coordinate of the image centre according to the WCS."""
        ny, nx = self.data.shape[-2:]
        return self.wcs.pixel_to_world((nx - 1) / 2.0, (ny - 1) / 2.0)

    @classmethod
    def from_fits(
        cls,
        path: str,
        survey: str,
        band: Optional[str] = None,
        hdu: Optional[int] = None,
        default_pixel_scale: Optional[Quantity] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "Cutout":
        """Load a cutout from a FITS file.

        The pixel scale is derived from the WCS; ``default_pixel_scale`` is only
        used when the header has no usable celestial WCS.

        :param path: Path to the FITS file.
        :param survey: Survey name to record.
        :param band: Band / instrument identifier to record.
        :param hdu: HDU index; by default the first HDU with 2D image data.
        :param default_pixel_scale: Fallback pixel scale.
        :param meta: Extra metadata to attach.
        :return: Loaded cutout.
        :rtype: Cutout
        """
        with fits.open(path, memmap=False) as hdul:
            if hdu is None:
                hdu = next(
                    (i for i, h in enumerate(hdul) if h.data is not None and np.ndim(h.data) >= 2),
                    None,
                )
                if hdu is None:
                    raise SurveyError(f"No 2D image data found in '{path}'.")
            data = np.asarray(hdul[hdu].data)
            header = hdul[hdu].header.copy()

        wcs = WCS(header).celestial
        pixel_scale = None
        if wcs.has_celestial:
            try:
                scales = proj_plane_pixel_scales(wcs)  # in CUNIT (deg for celestial)
                pixel_scale = (float(np.mean(scales)) * u.deg).to(u.arcsec)
            except Exception:  # pragma: no cover - malformed WCS
                pixel_scale = None
        if pixel_scale is None or not np.isfinite(pixel_scale.value) or pixel_scale.value <= 0:
            if default_pixel_scale is None:
                raise SurveyError(f"Could not determine pixel scale for '{path}'.")
            pixel_scale = default_pixel_scale.to(u.arcsec)

        return cls(
            data=data,
            header=header,
            wcs=wcs,
            pixel_scale=pixel_scale,
            survey=survey,
            band=band,
            path=str(path),
            meta=dict(meta or {}),
        )


class BaseSurvey(ABC):
    """Abstract base class for survey archive clients.

    Subclasses must set :attr:`name` and implement :meth:`query_images` and
    :meth:`get_cutout`. :meth:`login` / :meth:`logout` are no-ops by default,
    which is correct for public archives; :meth:`query_catalog` raises
    ``NotImplementedError`` unless overridden.

    Clients can be used as context managers, which calls :meth:`logout` on exit.
    """

    #: Registry name of the survey (e.g. ``"euclid"``).
    name: str = "base"
    #: Default band / instrument used when none is given.
    default_band: Optional[str] = None

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logged_in = False
        self.logger = setup_logging(name=f"astro_datatools.surveys.{self.name}")

    # -- authentication ---------------------------------------------------
    def login(self, **kwargs) -> None:
        """Authenticate with the archive. No-op for public archives.

        Implementations must never persist passwords.
        """
        self.logged_in = True

    def logout(self) -> None:
        """End the archive session. No-op for public archives."""
        self.logged_in = False

    def __enter__(self) -> "BaseSurvey":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.logged_in:
            self.logout()

    # -- queries ------------------------------------------------------------
    @abstractmethod
    def query_images(
        self,
        coordinate: CoordinateLike,
        radius: AngleLike,
        **filters,
    ) -> Table:
        """Find image products overlapping a circular sky region.

        :param coordinate: Centre of the search region.
        :param radius: Search radius (Quantity, or arcsec if a bare number).
        :param filters: Survey-specific filters (band, instrument, release...).
        :return: Table with one row per image product.
        :rtype: astropy.table.Table
        """

    @abstractmethod
    def get_cutout(
        self,
        coordinate: CoordinateLike,
        size: AngleLike,
        band: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> Cutout:
        """Retrieve an image cutout centred on ``coordinate``.

        :param coordinate: Centre of the cutout.
        :param size: Requested side length of the cutout (Quantity, or arcsec).
        :param band: Band / instrument identifier (survey-specific).
        :param output_path: Where to save the FITS file. If it already exists
            (and is non-empty) implementations should reuse it instead of
            downloading again.
        :return: The cutout.
        :rtype: Cutout
        """

    def query_catalog(self, coordinate: CoordinateLike, radius: AngleLike, **kwargs) -> Table:
        """Query a source catalogue around ``coordinate``.

        :raises NotImplementedError: Unless the survey implements it.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement query_catalog().")

    # -- convenience --------------------------------------------------------
    def default_cache_filename(self, coordinate: CoordinateLike, size: AngleLike, band: Optional[str]) -> str:
        """Deterministic file name for caching a cutout on disk.

        :param coordinate: Cutout centre.
        :param size: Cutout side length.
        :param band: Band identifier.
        :return: File name (no directory).
        :rtype: str
        """
        coord = to_skycoord(coordinate)
        size_as = to_angle(size).to_value(u.arcsec)
        band_str = (band or self.default_band or "img").replace("/", "-")
        return (
            f"{self.name}_{band_str}_ra{coord.ra.deg:.5f}_dec{coord.dec.deg:+.5f}"
            f"_size{size_as:.1f}arcsec.fits"
        )

    def find_empty_sky(
        self,
        coordinate: CoordinateLike,
        n: int = 16,
        size_pix: int = 256,
        cutout_size: AngleLike = 4 * u.arcmin,
        band: Optional[str] = None,
        cache_dir: Optional[str] = None,
        nsigma: float = 5.0,
        smoothing_sigma: float = 1.0,
        stride: Optional[int] = None,
        max_source_fraction: Optional[float] = None,
        **cutout_kwargs,
    ):
        """Download a cutout and extract the ``n`` least source-contaminated patches.

        Wraps :meth:`get_cutout` and
        :func:`astro_datatools.surveys.empty_regions.find_empty_regions`. With
        ``cache_dir`` set, the cutout is saved there under a deterministic name
        and reused on subsequent calls.

        :param coordinate: Centre of the cutout to search.
        :param n: Number of patches to return.
        :param size_pix: Side length of each patch in pixels.
        :param cutout_size: Side length of the downloaded cutout.
        :param band: Band / instrument (defaults to :attr:`default_band`).
        :param cache_dir: Directory for caching the cutout FITS file.
        :param nsigma: Detection threshold for the source mask.
        :param smoothing_sigma: Gaussian smoothing (pixels) before thresholding.
        :param stride: Window stride in pixels (default ``size_pix // 4``).
        :param max_source_fraction: Reject windows above this source fraction.
        :param cutout_kwargs: Passed to :meth:`get_cutout`.
        :return: Selected regions; ``meta`` holds survey, band, ``bunit``
            (header BUNIT, may be None), ``magzero`` (header MAGZERO, may be
            None), pixel scale, cutout path/shape and centre.
        :rtype: astro_datatools.surveys.empty_regions.EmptyRegions
        """
        from .empty_regions import find_empty_regions

        band = band or self.default_band
        output_path = None
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            output_path = os.path.join(
                cache_dir, self.default_cache_filename(coordinate, cutout_size, band)
            )
        cutout = self.get_cutout(
            coordinate, cutout_size, band=band, output_path=output_path, **cutout_kwargs
        )
        regions = find_empty_regions(
            cutout.data,
            size=size_pix,
            n=n,
            nsigma=nsigma,
            smoothing_sigma=smoothing_sigma,
            stride=stride,
            max_source_fraction=max_source_fraction,
        )
        regions.wcs = cutout.wcs
        regions.meta.update(
            {
                "survey": cutout.survey,
                "band": cutout.band,
                "bunit": cutout.unit,
                "magzero": cutout.header.get("MAGZERO"),
                "pixel_scale": cutout.pixel_scale,
                "cutout_path": cutout.path,
                "cutout_shape": cutout.shape,
                "coordinate": to_skycoord(coordinate),
            }
        )
        regions.meta.update(cutout.meta)
        return regions
