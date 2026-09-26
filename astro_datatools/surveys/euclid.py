"""Euclid archive client built on :mod:`astroquery.esa.euclid`.

``astroquery`` is an optional dependency (``pip install astro-datatools[surveys]``)
and is only imported when the client is first used.

Metadata queries (:meth:`EuclidSurvey.query_images`) work anonymously.
Cutouts of public (Q1) tiles have been observed to work anonymously too, but
for proprietary data (or at times for all data) the cutout service answers
HTTP 200 with an empty file when not logged in; :meth:`EuclidSurvey.get_cutout`
detects this and raises
:class:`~astro_datatools.surveys.base.SurveyAuthenticationError` asking you to
:meth:`~EuclidSurvey.login`.

MER mosaic cutouts have no ``BUNIT`` keyword; pixel values are in image units
with AB zero point given by ``MAGZERO`` (24.6 for VIS in Q1).
"""
import os
import re
import tempfile
import time
from typing import Any, Optional, Sequence

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from .base import (
    AngleLike,
    BaseSurvey,
    CoordinateLike,
    Cutout,
    SurveyAuthenticationError,
    SurveyError,
    to_angle,
    to_skycoord,
)
from .registry import register_survey

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_\-]+$")


def _require_astroquery():
    try:
        from astroquery.esa.euclid import Euclid, EuclidClass  # noqa: F401
    except ImportError as err:  # pragma: no cover - depends on environment
        raise ImportError(
            "EuclidSurvey requires astroquery. Install it with "
            "`pip install astroquery` or `pip install astro-datatools[surveys]`."
        ) from err
    return Euclid, EuclidClass


def normalize_band(band: Optional[str]) -> Optional[str]:
    """Normalise a Euclid band name to the archive's ``filter_name`` convention.

    ``"vis"`` -> ``"VIS"``; ``"H"``, ``"NIR-H"``, ``"nir_h"`` -> ``"NIR_H"``.
    Other names (e.g. external ``"HSC_g"``) are passed through unchanged apart
    from validation.

    :param band: Band name or ``None``.
    :return: Normalised band name or ``None``.
    """
    if band is None:
        return None
    band = str(band).strip()
    if not _SAFE_TOKEN.match(band):
        raise ValueError(f"Invalid band name '{band}'.")
    upper = band.upper().replace("-", "_")
    if upper == "VIS":
        return "VIS"
    if upper in ("Y", "J", "H"):
        return f"NIR_{upper}"
    if upper in ("NIR_Y", "NIR_J", "NIR_H"):
        return upper
    return band


def _validate_token(value: Optional[str], what: str) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if not _SAFE_TOKEN.match(value):
        raise ValueError(f"Invalid {what} '{value}'.")
    return value


@register_survey("euclid")
class EuclidSurvey(BaseSurvey):
    """Client for the ESA Euclid science archive.

    Example::

        from astro_datatools.surveys import get_survey
        euclid = get_survey("euclid")
        tiles = euclid.query_images(euclid.EDF_NORTH, 1 * u.arcmin, band="VIS")
        euclid.login(user="my_cosmos_user")      # if needed; prompts for the password
        cutout = euclid.get_cutout(euclid.EDF_NORTH, 1 * u.arcmin, band="VIS")

    :param client: Pre-configured astroquery ``EuclidClass`` instance (mainly for
        testing). Defaults to astroquery's shared ``Euclid`` object.
    :param environment: Archive environment (e.g. ``"PDR"``, ``"IDR"``); creates
        a dedicated ``EuclidClass`` instead of the shared default.
    :param verbose: Pass ``verbose=True`` to astroquery calls.
    """

    name = "euclid"
    default_band = "VIS"
    #: Nominal VIS (and MER mosaic) pixel scale; used only if a file lacks WCS.
    PIXEL_SCALE = 0.1 * u.arcsec
    #: Approximate centre of the Euclid Deep Field North (covered by Q1).
    EDF_NORTH = SkyCoord(269.73, 66.02, unit="deg", frame="icrs")
    MOSAIC_TABLE = "sedm.mosaic_product"
    #: Number of retries for failed metadata queries, and seconds between them.
    query_retries = 2
    retry_wait = 2.0
    DEFAULT_COLUMNS = (
        "file_name",
        "file_path",
        "tile_index",
        "instrument_name",
        "filter_name",
        "release_name",
        "product_type",
        "ra",
        "dec",
        "zero_point",
    )

    def __init__(self, client: Any = None, environment: Optional[str] = None, verbose: bool = False):
        super().__init__(verbose=verbose)
        self._client = client
        self._environment = environment

    @property
    def client(self):
        """The underlying astroquery Euclid client (created lazily)."""
        if self._client is None:
            Euclid, EuclidClass = _require_astroquery()
            self._client = EuclidClass(environment=self._environment) if self._environment else Euclid
        return self._client

    # -- authentication ---------------------------------------------------
    def login(
        self,
        user: Optional[str] = None,
        password: Optional[str] = None,
        credentials_file: Optional[str] = None,
    ) -> None:
        """Log in to the Euclid archive (data and cutout services).

        If only ``user`` is given, astroquery prompts for the password
        interactively (``getpass``). The password is never stored by this class.

        :param user: Archive (Cosmos) user name.
        :param password: Password; prefer leaving it ``None`` to be prompted.
        :param credentials_file: File with user name and password on two lines.
        """
        self.client.login(
            user=user, password=password, credentials_file=credentials_file, verbose=self.verbose
        )
        self.logged_in = True

    def logout(self) -> None:
        """Log out of the Euclid archive."""
        if self._client is not None or self.logged_in:
            self.client.logout(verbose=self.verbose)
        self.logged_in = False

    # -- queries ------------------------------------------------------------
    def build_mosaic_query(
        self,
        coordinate: CoordinateLike,
        radius: AngleLike,
        band: Optional[str] = None,
        instrument: Optional[str] = None,
        release: Optional[str] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> str:
        """Build the ADQL query used by :meth:`query_images`.

        :return: ADQL query string.
        """
        coord = to_skycoord(coordinate)
        radius_deg = to_angle(radius).to_value(u.deg)
        cols = ", ".join(_validate_token(c, "column") for c in (columns or self.DEFAULT_COLUMNS))
        where = ["mosaic_product.fov IS NOT NULL"]
        band = normalize_band(band)
        if band is not None:
            where.append(f"filter_name='{band}'")
        instrument = _validate_token(instrument, "instrument")
        if instrument is not None:
            where.append(f"instrument_name='{instrument.upper()}'")
        release = _validate_token(release, "release")
        if release is not None:
            where.append(f"release_name='{release}'")
        where.append(
            f"INTERSECTS(CIRCLE('ICRS', {coord.ra.deg:.8f}, {coord.dec.deg:.8f}, {radius_deg:.8f}), "
            "mosaic_product.fov)=1"
        )
        return (
            f"SELECT {cols} FROM {self.MOSAIC_TABLE} WHERE "
            + " AND ".join(where)
            + " ORDER BY tile_index"
        )

    def query_images(
        self,
        coordinate: CoordinateLike,
        radius: AngleLike,
        band: Optional[str] = None,
        instrument: Optional[str] = None,
        release: Optional[str] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> Table:
        """Find background-subtracted MER mosaic tiles overlapping a sky region.

        Works without login.

        :param coordinate: Search centre (``SkyCoord`` or ``(ra, dec)`` in deg).
        :param radius: Search radius (Quantity, or arcsec for a bare number).
        :param band: Filter, e.g. ``"VIS"``, ``"NIR_H"`` / ``"H"``; ``None`` for all.
        :param instrument: Instrument name, e.g. ``"VIS"`` or ``"NISP"``.
        :param release: Release name, e.g. ``"Q1_R1"``.
        :param columns: Columns to return (default :attr:`DEFAULT_COLUMNS`).
        :return: One row per mosaic tile/band, ordered by ``tile_index``.
        :rtype: astropy.table.Table
        """
        query = self.build_mosaic_query(coordinate, radius, band, instrument, release, columns)
        return self._run_query(query)

    def _run_query(self, query: str) -> Table:
        self.logger.debug("Euclid ADQL: %s", query)
        # astroquery logs HTTP errors and returns None instead of raising; the
        # archive occasionally answers with transient HTTP 500s, so retry.
        for attempt in range(1 + self.query_retries):
            job = self.client.launch_job_async(query, verbose=self.verbose)
            if job is not None:
                return job.get_results()
            if attempt < self.query_retries:
                self.logger.warning("Euclid query failed; retrying (%d/%d)", attempt + 1, self.query_retries)
                time.sleep(self.retry_wait)
        raise SurveyError(f"Euclid archive query failed (see astroquery log). Query: {query}")

    #: MER source catalogue used by :meth:`estimate_gain`.
    CATALOGUE_TABLE = "catalogue.mer_catalogue"

    def estimate_gain(
        self,
        coordinate: CoordinateLike = None,
        radius: AngleLike = 0.15 * u.deg,
        band: str = "VIS",
        aperture: str = "2fwhm",
        zeropoint: float = 24.6,
        n_bins: int = 15,
    ) -> dict:
        """Estimate the effective gain (detected photoelectrons per mosaic unit) from the MER catalogue.

        For a fixed aperture, the catalogue flux error is ``err^2 = a + k * F``: ``a`` is the
        sky/read noise inside the aperture (the same for every source) and ``k * F`` is the
        source's own Poisson noise, with ``k`` = flux per detected electron. Fitting ``err^2``
        against ``F`` (medians in log-spaced flux bins, weighted fit) gives ``k``, and the gain in
        electrons per mosaic unit is ``(flux of one mosaic unit) / k``. Use it as
        ``Sim2Real(shot_noise_intensity=gain)`` for images in mosaic units. Works without login.

        :param coordinate: Centre of the source sample (default :attr:`EDF_NORTH`).
        :param radius: Radius of the source sample.
        :param band: Band whose aperture fluxes are used, e.g. ``"VIS"`` (catalogue column
            prefix ``flux_<band>_<aperture>_aper``).
        :param aperture: Aperture size label, e.g. ``"1fwhm"``, ``"2fwhm"``.
        :param zeropoint: AB zero point of the mosaic units (``MAGZERO``, 24.6 for Q1 VIS).
        :param n_bins: Number of log-spaced flux bins for the fit.
        :return: dict with ``gain`` (e-/mosaic unit), ``k`` (µJy per e-), ``sky_term`` (µJy²),
            ``n_sources``, and the binned ``flux``/``err2`` used in the fit (µJy, µJy²).
        :raises SurveyError: If the fit finds no positive Poisson term.
        """
        centre = to_skycoord(coordinate if coordinate is not None else self.EDF_NORTH)
        radius_deg = to_angle(radius).to_value(u.deg)
        band_col = _validate_token(band, "band").lower().replace("nir_", "")
        aperture = _validate_token(aperture, "aperture").lower()
        flux, err = f"flux_{band_col}_{aperture}_aper", f"fluxerr_{band_col}_{aperture}_aper"
        query = (
            f"SELECT {flux} AS f, {err} AS e FROM {self.CATALOGUE_TABLE} WHERE "
            f"1=CONTAINS(POINT('ICRS', right_ascension, declination), "
            f"CIRCLE('ICRS', {centre.ra.deg:.6f}, {centre.dec.deg:.6f}, {radius_deg:.6f})) "
            f"AND spurious_flag=0 AND det_quality_flag=0 AND {flux} > 0 AND {err} > 0"
        )
        table = self._run_query(query)
        f = np.asarray(table["f"], dtype=float)
        e2 = np.asarray(table["e"], dtype=float) ** 2
        ok = np.isfinite(f) & np.isfinite(e2)
        f, e2 = f[ok], e2[ok]

        edges = np.logspace(np.log10(np.percentile(f, 5)), np.log10(np.percentile(f, 99.5)), n_bins + 1)
        flux_b, err2_b = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bin = (f >= lo) & (f < hi)
            if in_bin.sum() >= 10:
                flux_b.append(np.median(f[in_bin]))
                err2_b.append(np.median(e2[in_bin]))
        flux_b, err2_b = np.array(flux_b), np.array(err2_b)
        if len(flux_b) < 3:
            raise SurveyError(f"Too few catalogue sources ({len(f)}) to fit the gain.")
        k, sky_term = np.polyfit(flux_b, err2_b, 1, w=1 / err2_b)
        if k <= 0:
            raise SurveyError("Catalogue errors show no Poisson term; cannot estimate the gain.")

        ujy_per_unit = 3631e6 * 10 ** (-0.4 * zeropoint)
        return {
            "gain": ujy_per_unit / k,
            "k": k,
            "sky_term": sky_term,
            "n_sources": int(len(f)),
            "flux": flux_b,
            "err2": err2_b,
        }

    def get_cutout(
        self,
        coordinate: CoordinateLike,
        size: AngleLike,
        band: Optional[str] = None,
        output_path: Optional[str] = None,
        tile: Any = None,
        release: Optional[str] = None,
        overwrite: bool = False,
    ) -> Cutout:
        """Download a cutout of a MER mosaic tile (may require :meth:`login`).

        The archive service takes a *radius*; ``radius = size / 2`` is sent and
        the service returns a square of roughly side ``size`` (a 20 arcsec
        request gave 202 x 202 VIS pixels). Rely on the returned
        :class:`Cutout` (shape, WCS, pixel scale) rather than on ``size``.

        :param coordinate: Cutout centre.
        :param size: Requested side length (Quantity, or arcsec). Max 1 deg.
        :param band: Filter (default ``"VIS"``).
        :param output_path: FITS file to write. If it exists and is non-empty and
            ``overwrite`` is False, it is loaded instead of downloading again.
            Defaults to a file in a fresh temporary directory.
        :param tile: Row (or dict) from :meth:`query_images` selecting the tile.
            By default the covering tile whose centre is closest to ``coordinate``.
        :param release: Restrict tile search to a release, e.g. ``"Q1_R1"``.
        :param overwrite: Re-download even if ``output_path`` exists.
        :return: The cutout.
        :rtype: Cutout
        :raises SurveyAuthenticationError: If the archive returned an empty file
            (login required).
        :raises SurveyError: If no tile covers the position or the download failed.
        """
        coord = to_skycoord(coordinate)
        size = to_angle(size)
        band = normalize_band(band or self.default_band)

        if output_path is not None and not overwrite and _nonempty_file(output_path):
            self.logger.info("Using cached Euclid cutout %s", output_path)
            return Cutout.from_fits(
                output_path, survey=self.name, band=band,
                default_pixel_scale=self.PIXEL_SCALE, meta={"cached": True},
            )

        if tile is None:
            tiles = self.query_images(coord, 1 * u.arcsec, band=band, release=release)
            if len(tiles) == 0:
                raise SurveyError(
                    f"No Euclid {band} mosaic tile covers RA={coord.ra.deg:.5f}, Dec={coord.dec.deg:.5f}."
                )
            tile = _closest_tile(tiles, coord)

        file_path = str(tile["file_path"]).rstrip("/")
        file_name = str(tile["file_name"])
        tile_index = str(tile["tile_index"])
        instrument = str(tile["instrument_name"]) if _has(tile, "instrument_name") else (
            "VIS" if band == "VIS" else "NISP"
        )

        if output_path is None:
            output_path = os.path.join(
                tempfile.mkdtemp(prefix="euclid_cutout_"),
                self.default_cache_filename(coord, size, band),
            )
        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
        if overwrite and os.path.exists(output_path):
            os.remove(output_path)

        self.logger.info(
            "Requesting Euclid %s cutout (tile %s, size %s) at RA=%.5f Dec=%.5f",
            band, tile_index, size, coord.ra.deg, coord.dec.deg,
        )
        result = self.client.get_cutout(
            file_path=f"{file_path}/{file_name}",
            instrument=instrument,
            id=tile_index,
            coordinate=coord,
            radius=(size / 2).to(u.arcsec),
            output_file=output_path,
            verbose=self.verbose,
        )
        saved = _pick_saved_file(result, output_path)
        if saved is None:
            raise SurveyError(
                f"Euclid cutout download failed for tile {tile_index} ({file_name}); "
                "see the astroquery log for the HTTP error."
            )
        if not _nonempty_file(saved):
            if os.path.exists(saved):
                os.remove(saved)
            raise SurveyAuthenticationError(
                "The Euclid cutout service returned an empty file. Cutouts require an "
                "authenticated session: call EuclidSurvey.login(user=...) first."
            )

        return Cutout.from_fits(
            saved,
            survey=self.name,
            band=band,
            default_pixel_scale=self.PIXEL_SCALE,
            meta={
                "tile_index": tile_index,
                "file_name": file_name,
                "release": str(tile["release_name"]) if _has(tile, "release_name") else None,
                "requested_size": size,
                "cached": False,
            },
        )

    def find_empty_sky(
        self,
        coordinate: Optional[CoordinateLike] = None,
        n: int = 16,
        size_pix: int = 256,
        cutout_size: AngleLike = 4 * u.arcmin,
        band: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        """Find the least source-contaminated patches in a Euclid cutout.

        Same as :meth:`BaseSurvey.find_empty_sky` but defaults to the Euclid Deep
        Field North and the VIS band. May require :meth:`login` unless the
        cutout is already cached in ``cache_dir``. At 0.1 arcsec/px the default 4 arcmin
        cutout is ~2400 px on a side; ``size_pix=256`` is ~26 arcsec.

        :param coordinate: Cutout centre (default :attr:`EDF_NORTH`).
        :param n: Number of patches.
        :param size_pix: Patch side length in pixels.
        :param cutout_size: Side length of the downloaded cutout (max 1 deg).
        :param band: Filter (default ``"VIS"``).
        :param cache_dir: Directory to cache the cutout FITS file.
        :param kwargs: ``nsigma``, ``smoothing_sigma``, ``stride``,
            ``max_source_fraction`` and :meth:`get_cutout` options.
        :rtype: astro_datatools.surveys.empty_regions.EmptyRegions
        """
        return super().find_empty_sky(
            coordinate if coordinate is not None else self.EDF_NORTH,
            n=n,
            size_pix=size_pix,
            cutout_size=cutout_size,
            band=band,
            cache_dir=cache_dir,
            **kwargs,
        )


def _has(row: Any, key: str) -> bool:
    try:
        names = row.colnames if hasattr(row, "colnames") else row.keys()
    except Exception:
        return False
    return key in names


def _nonempty_file(path: Optional[str]) -> bool:
    return path is not None and os.path.isfile(path) and os.path.getsize(path) > 0


def _closest_tile(tiles: Table, coord: SkyCoord):
    if len(tiles) == 1 or "ra" not in tiles.colnames or "dec" not in tiles.colnames:
        return tiles[0]
    centres = SkyCoord(np.asarray(tiles["ra"], float), np.asarray(tiles["dec"], float), unit="deg")
    return tiles[int(np.argmin(coord.separation(centres).deg))]


def _pick_saved_file(result: Any, output_path: str) -> Optional[str]:
    """Pick the FITS file path from astroquery's ``get_cutout`` return value."""
    if result is None:
        return output_path if os.path.exists(output_path) else None
    if isinstance(result, (str, os.PathLike)):
        return str(result)
    files = [str(f) for f in result]
    if not files:
        return output_path if os.path.exists(output_path) else None
    fits_files = [f for f in files if ".fits" in os.path.basename(f).lower()]
    return (fits_files or files)[0]
