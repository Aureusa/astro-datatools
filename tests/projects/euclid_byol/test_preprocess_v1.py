"""Tests for the frozen Euclid BYOL preprocessing (projects/euclid_byol/preprocess_v1.py)."""
import hashlib
import io
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from projects.euclid_byol import preprocess_v1 as prep

SIZE = prep.OUTPUT_SIZE
C = SIZE // 2

# SHA-256 of preprocess(synthetic_galaxy()). If this fails, the output of v1 has
# changed (edited function, or a numpy/astropy behaviour change): do NOT update
# the hash for a dataset that already exists; create preprocess_v2 instead.
PINNED_OUTPUT_SHA256 = "aec79c1dcc8c64303a4ab5401635bda12cde1df02a082657cce87e4192aede55"


def hash_noise(shape, seed=12345):
    """Approximately standard-normal noise from integer hashing only.

    Unlike numpy's random generators, this sequence can never change between
    numpy versions, so the pinned output hash only tests the preprocessing.
    """
    n = int(np.prod(shape)) * 4
    with np.errstate(over="ignore"):
        z = np.arange(n, dtype=np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    u = (z >> np.uint64(11)).astype(np.float64) / 2.0 ** 53
    # Sum of 4 uniforms, scaled to unit variance.
    return ((u.reshape(4, -1).sum(axis=0) - 2.0) * np.sqrt(3.0)).reshape(shape)


def synthetic_galaxy(noise_sigma=1.0, arm_amplitude=3.0, peak=400.0, disc_scale=25.0):
    """Disc galaxy with two faint spiral arms, a bright bulge and noise, centred at (C, C)."""
    yy, xx = np.mgrid[:SIZE, :SIZE].astype(np.float64)
    dx, dy = xx - C, yy - C
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    disc = 20.0 * np.exp(-r / disc_scale)
    bulge = peak * np.exp(-((r / 3.0) ** 2))
    arms = arm_amplitude * np.cos(2 * theta - 0.25 * r) ** 8 * np.exp(-r / 60.0) * (r > 12)
    return disc + bulge + arms + noise_sigma * hash_noise((SIZE, SIZE))


@pytest.fixture(scope="module")
def galaxy():
    return synthetic_galaxy()


def test_output_is_uint8_of_input_shape(galaxy):
    out = prep.preprocess(galaxy)
    assert out.dtype == np.uint8 and out.shape == galaxy.shape


def test_is_deterministic_and_does_not_modify_input(galaxy):
    before = galaxy.copy()
    assert np.array_equal(prep.preprocess(galaxy), prep.preprocess(galaxy))
    assert np.array_equal(galaxy, before)


def test_float32_input_gives_same_result(galaxy):
    as32 = galaxy.astype(np.float32)
    assert np.array_equal(prep.preprocess(as32), prep.preprocess(as32.astype(np.float64)))


def test_output_is_pinned(galaxy):
    digest = hashlib.sha256(prep.preprocess(galaxy).tobytes()).hexdigest()
    assert digest == PINNED_OUTPUT_SHA256, (
        "preprocess_v1 output changed. If a dataset was generated with v1, do not "
        f"re-pin: create preprocess_v2. New digest: {digest}"
    )


def test_noise_estimate_ignores_a_compact_galaxy():
    _, stats = prep.preprocess_with_stats(synthetic_galaxy(disc_scale=8.0))
    assert stats["noise_sigma"] == pytest.approx(1.0, rel=0.1)
    assert not stats["degenerate"]


def test_noise_estimate_of_a_galaxy_filling_the_cutout_is_biased_high_but_bounded(galaxy):
    # Known property of v1: when a galaxy covers most of the cutout, clipping cannot
    # separate its faint disc from the noise, so sigma (and hence the stretch
    # softening) comes out somewhat high. The image stays well-behaved (tests below).
    _, stats = prep.preprocess_with_stats(galaxy)
    assert 1.0 < stats["noise_sigma"] < 1.6


def test_faint_outskirts_not_clipped(galaxy):
    out, stats = prep.preprocess_with_stats(galaxy)
    # Background sits clearly above black, so faint signal and noise texture survive.
    background = np.median(out[:20, :20])
    assert background >= 10
    assert (out == 0).mean() < 0.05
    # Outskirts at a few sigma are brighter than the background.
    yy, xx = np.mgrid[:SIZE, :SIZE]
    r = np.hypot(xx - C, yy - C)
    outskirt = np.median(out[(r > 50) & (r < 55)])  # disc there is ~2-3 sigma
    assert outskirt > background + 5


def test_bright_core_not_saturated(galaxy):
    out = prep.preprocess(galaxy)
    assert (out == 255).sum() <= 2
    core = out[C - 5:C + 6, C - 5:C + 6]
    assert len(np.unique(core)) > 15  # the core keeps its gradient


def test_spiral_arms_visible():
    clean = synthetic_galaxy(noise_sigma=1.0, arm_amplitude=4.0)
    no_arms = synthetic_galaxy(noise_sigma=1.0, arm_amplitude=0.0)
    arms_region = (synthetic_galaxy(noise_sigma=0.0, arm_amplitude=4.0)
                   - synthetic_galaxy(noise_sigma=0.0, arm_amplitude=0.0)) > 2.0
    diff = prep.preprocess(clean).astype(int) - prep.preprocess(no_arms).astype(int)
    assert diff[arms_region].mean() > 8  # arms add clearly visible grey levels


def test_nan_and_zero_pixels_are_blank(galaxy):
    image = galaxy.copy()
    image[:, :SIZE // 4] = np.nan
    image[:SIZE // 4, :] = 0.0
    out, stats = prep.preprocess_with_stats(image)
    assert 0.4 < stats["blank_fraction"] < 0.5
    assert not stats["degenerate"]
    # Blank pixels are set to the background level, not to black.
    assert np.median(out[:, :SIZE // 4]) > 0


@pytest.mark.parametrize("image", [np.full((SIZE, SIZE), np.nan), np.zeros((SIZE, SIZE)), np.ones((SIZE, SIZE))])
def test_degenerate_images(image):
    out, stats = prep.preprocess_with_stats(image)
    assert stats["degenerate"]
    assert out.dtype == np.uint8 and not out.any()


def _wcs(ra, dec, shape, scale_arcsec=0.1):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [shape[1] / 2 + 0.5, shape[0] / 2 + 0.5]
    w.wcs.cdelt = [-scale_arcsec / 3600, scale_arcsec / 3600]
    return w


def test_extract_cutout_centres_the_source():
    data = np.zeros((300, 280))
    data[140, 150] = 1.0
    wcs = _wcs(52.0, -27.5, data.shape)
    ra, dec = wcs.pixel_to_world_values(150.2, 139.8)
    cut = prep.extract_cutout(data, wcs, float(ra), float(dec))
    assert cut.shape == (SIZE, SIZE)
    assert cut[C, C] == 1.0 and np.nansum(cut) == 1.0


def test_extract_cutout_pads_outside_the_image():
    data = np.ones((100, 100))
    wcs = _wcs(52.0, -27.5, data.shape)
    ra, dec = wcs.pixel_to_world_values(5.0, 90.0)
    cut = prep.extract_cutout(data, wcs, float(ra), float(dec))
    assert cut.shape == (SIZE, SIZE)
    assert np.isnan(cut[0, 0]) and cut[C, C] == 1.0
    assert np.isfinite(cut).sum() == 100 * 100


def _fits_bytes(data, wcs):
    buf = io.BytesIO()
    fits.PrimaryHDU(data=data.astype(np.float32), header=wcs.to_header()).writeto(buf)
    return buf.getvalue()


def test_preprocess_fits_from_bytes(galaxy):
    padded = np.pad(galaxy, 8)
    wcs = _wcs(52.0, -27.5, padded.shape)
    ra, dec = wcs.pixel_to_world_values(C + 8, C + 8)
    image, raw, stats = prep.preprocess_fits(_fits_bytes(padded, wcs), float(ra), float(dec))
    assert image.shape == (SIZE, SIZE) and raw.dtype == np.float32
    assert np.array_equal(raw, galaxy.astype(np.float32))
    assert np.array_equal(image, prep.preprocess(raw))


def test_preprocess_fits_rejects_other_pixel_scales(galaxy):
    wcs = _wcs(52.0, -27.5, galaxy.shape, scale_arcsec=0.3)
    with pytest.raises(ValueError, match="Pixel scale"):
        prep.preprocess_fits(_fits_bytes(galaxy, wcs))


def test_fingerprint_is_hash_of_module_source():
    source = Path(prep.__file__).read_bytes()
    assert prep.FINGERPRINT == hashlib.sha256(source).hexdigest()
    assert prep.VERSION == "v1" and prep.OUTPUT_SIZE == 224
