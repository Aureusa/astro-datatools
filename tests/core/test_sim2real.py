import astropy.units as u
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from astro_datatools.core import STAGE_NAMES, Sim2Real


COSMO = FlatLambdaCDM(H0=68.1, Om0=0.304611, Tcmb0=2.7255)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_s2r(output_size=224, **kwargs):
    return Sim2Real(0.2 * u.kpc, 0.1 * u.arcsec, output_size, cosmology=COSMO, **kwargs)


def gaussian_psf(size=7, sigma=1.2):
    y, x = np.mgrid[:size, :size] - size // 2
    psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return psf / psf.sum()


def delta_psf(size=5):
    psf = np.zeros((size, size))
    psf[size // 2, size // 2] = 1.0
    return psf


def galaxy(n=500, channels=3, sigma=40.0):
    y, x = np.mgrid[:n, :n] - (n - 1) / 2
    img = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return np.repeat(img[None], channels, axis=0) * np.linspace(0.5, 1.0, channels)[:, None, None]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("z, expected", [(0.1, 0.190), (1.0, 0.821), (3.0, 0.788)])
def test_target_pixel_kpc(z, expected):
    value = make_s2r().target_pixel_kpc(z)
    assert value.unit == u.kpc
    assert value.value == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("z", [0, 0.0, -0.5])
def test_non_positive_redshift_raises(z):
    s2r = make_s2r()
    with pytest.raises(ValueError, match="angular diameter distance"):
        s2r.target_pixel_kpc(z)
    with pytest.raises(ValueError, match="Redshift must be > 0"):
        s2r(np.zeros((1, 50, 50)), z, delta_psf())


def test_rebinned_shape_crop_low_z_pad_high_z():
    s2r = make_s2r(output_size=224)
    assert s2r.rebinned_shape((3, 500, 500), 0.1) == (528, 528)
    assert s2r.rebinned_shape((500, 500), 1.0) == (122, 122)

    image = galaxy()
    low = s2r.run(image, 0.1, gaussian_psf(), return_stages=True)
    high = s2r.run(image, 1.0, gaussian_psf(), return_stages=True)
    assert low["rebinned"].shape == (3, 528, 528)
    assert high["rebinned"].shape == (3, 122, 122)
    assert low["final"].shape == high["final"].shape == (3, 224, 224)
    # Padded border stays empty at high z (no noise, no background).
    assert np.all(high["final"][:, :40, :] == 0)


def test_quantity_units_are_validated():
    with pytest.raises(TypeError, match="Quantity"):
        Sim2Real(0.2, 0.1 * u.arcsec, 64)
    with pytest.raises(u.UnitConversionError):
        Sim2Real(0.2 * u.arcsec, 0.1 * u.arcsec, 64)
    # Other compatible units are converted.
    s2r = Sim2Real(200 * u.pc, 100 * u.mas, 64, cosmology=COSMO)
    assert s2r.target_pixel_kpc(1.0).value == pytest.approx(0.821, abs=1e-3)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def test_output_shape_and_dtype():
    s2r = make_s2r(output_size=64, shot_noise_intensity=1000, seed=0)
    bg = np.random.default_rng(0).normal(0, 0.01, (64, 64))
    out = s2r(galaxy(200), 1.0, gaussian_psf(), bg)
    assert out.shape == (3, 64, 64)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    assert out.min() < 0  # sky noise is not clipped


def test_batched_and_2d_input():
    s2r = make_s2r(output_size=64)
    batch = np.stack([galaxy(200), galaxy(200, sigma=10)])
    out = s2r(batch, 1.0, gaussian_psf(), np.zeros((64, 64)))
    assert out.shape == (2, 3, 64, 64)
    np.testing.assert_allclose(out[0], s2r(batch[0], 1.0, gaussian_psf()), atol=1e-6)

    stages = s2r.run(galaxy(200)[0], 1.0, gaussian_psf(), np.zeros((64, 64)), return_stages=True)
    for name, stage in stages.items():
        assert stage is None or stage.ndim == 2, name
    assert stages["final"].shape == (64, 64)


def test_rejects_unsupported_image_dims():
    with pytest.raises(ValueError, match="Expected image"):
        make_s2r()(np.zeros((1, 1, 1, 10, 10)), 1.0, delta_psf())


def test_stages_contain_every_step():
    s2r = make_s2r(output_size=64, shot_noise_intensity=500, seed=1)
    bg = np.random.default_rng(0).normal(0, 1.0, (3, 64, 64))
    stages = s2r.run(galaxy(200), 1.0, gaussian_psf(), bg, return_stages=True)
    assert tuple(stages) == STAGE_NAMES
    assert all(stages[name] is not None for name in STAGE_NAMES)
    np.testing.assert_allclose(stages["final"], stages["noisy"] + stages["background"], atol=1e-5)

    skipped = make_s2r(output_size=64).run(galaxy(200), 1.0, gaussian_psf(), return_stages=True)
    assert tuple(skipped) == STAGE_NAMES
    assert skipped["noisy"] is None and skipped["background"] is None
    np.testing.assert_allclose(skipped["final"], skipped["cropped"], atol=1e-6)


# ---------------------------------------------------------------------------
# Physics sanity
# ---------------------------------------------------------------------------


def test_point_source_stays_centred():
    s2r = make_s2r(output_size=101)
    for n in (500, 499):
        image = np.zeros((1, n, n))
        image[0, n // 2, n // 2] = 1.0
        for z in (0.1, 1.0):
            out = s2r(image, z, gaussian_psf())
            y, x = np.unravel_index(np.argmax(out[0]), out[0].shape)
            assert abs(y - 50) <= 1 and abs(x - 50) <= 1, (n, z, y, x)


def test_surface_brightness_preserved_without_noise():
    """Delta PSF, no crop, no noise/background: mean over the galaxy region is unchanged."""
    s2r = make_s2r(output_size=224)
    image = galaxy(500)
    z = 1.0
    stages = s2r.run(image, z, delta_psf(), return_stages=True)
    ny, nx = s2r.rebinned_shape(image.shape, z)
    assert ny <= 224  # padded, nothing cropped away
    # Galaxy region = the rebinned footprint inside the padded frame.
    top, left = (224 - ny) // 2, (224 - nx) // 2
    region = stages["final"][:, top:top + ny, left:left + nx]
    np.testing.assert_allclose(region.mean(axis=(-2, -1)), image.mean(axis=(-2, -1)), rtol=1e-5)
    # Nothing leaked outside the footprint.
    assert stages["final"].sum() == pytest.approx(region.sum(), rel=1e-6)

    # Uniform patch keeps its surface brightness pixel by pixel.
    flat = np.full((1, 500, 500), 0.4)
    out = s2r(flat, z, delta_psf())
    np.testing.assert_allclose(out[0, top:top + ny, left:left + nx], 0.4, rtol=1e-5)


def test_flux_mode_preserves_total():
    s2r = make_s2r(output_size=224, conserve="flux")
    image = galaxy(500)
    out = s2r(image, 1.0, delta_psf())
    np.testing.assert_allclose(out.sum(axis=(-2, -1)), image.sum(axis=(-2, -1)), rtol=1e-5)


def test_psf_is_normalised_and_clipped():
    # Unnormalised PSF with a negative wing: normalised to unit sum, then clipped to [0, 1].
    psf = gaussian_psf() * 7.0
    psf[0, 0] = -0.05
    image = np.zeros((1, 200, 200))
    image[:, 80:120, 80:120] = 1.0
    stages = make_s2r(output_size=64).run(image, 1.0, psf, return_stages=True)
    assert stages["convolved"].min() >= 0 and stages["convolved"].max() <= 1

    unclipped = make_s2r(output_size=64, clip=None).run(image, 1.0, psf, return_stages=True)
    assert unclipped["convolved"].min() < 0

    raw = make_s2r(output_size=64, clip=None, normalise_psf=False)(image, 1.0, gaussian_psf() * 7.0)
    assert raw.max() > 1.5


def test_channel_specific_psf():
    image = np.zeros((2, 200, 200))
    image[:, 100, 100] = 1.0
    psf = np.stack([delta_psf(7), gaussian_psf(7, 1.5)])
    out = make_s2r(output_size=64)(image, 1.0, psf)
    # Channel 0 unblurred (single pixel), channel 1 spread out; both conserve the sum.
    assert np.count_nonzero(out[0] > 1e-8) < np.count_nonzero(out[1] > 1e-8)
    np.testing.assert_allclose(out[0].sum(), out[1].sum(), rtol=1e-5)

    with pytest.raises(ValueError, match="PSF has 3 channels"):
        make_s2r(output_size=64)(image, 1.0, np.stack([delta_psf()] * 3))


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------


def test_background_size_mismatch_raises():
    s2r = make_s2r(output_size=64)
    with pytest.raises(ValueError, match="does not match output_size"):
        s2r(galaxy(200), 1.0, gaussian_psf(), np.zeros((3, 32, 32)))
    with pytest.raises(ValueError, match="channels"):
        s2r(galaxy(200), 1.0, gaussian_psf(), np.zeros((2, 64, 64)))


def test_background_sigma_rescaling():
    rng = np.random.default_rng(3)
    bg = rng.normal(100.0, [[[5.0]], [[20.0]]], (2, 128, 128))
    s2r = make_s2r(output_size=128, background_sigma=0.01)
    stages = s2r.run(np.zeros((2, 200, 200)), 1.0, delta_psf(), bg, return_stages=True)
    np.testing.assert_allclose(stages["background"].std(axis=(-2, -1)), 0.01, rtol=0.05)

    # Given noise level instead of the sigma-clipped estimate.
    given = s2r.run(np.zeros((2, 200, 200)), 1.0, delta_psf(), bg,
                    background_noise=np.array([5.0, 20.0]), return_stages=True)
    np.testing.assert_allclose(given["background"], bg * (0.01 / np.array([5.0, 20.0]))[:, None, None])


def test_fill_embedding():
    s2r = make_s2r(output_size=64, embed_type="fill")
    bg = np.full((64, 64), -1.0)
    out = s2r(galaxy(200, sigma=10), 1.0, delta_psf(), bg)
    assert np.all((out == -1.0) | (out > 0))
    assert np.any(out == -1.0) and np.any(out > 0)


# ---------------------------------------------------------------------------
# Shot noise RNG
# ---------------------------------------------------------------------------


def test_seeded_noise_is_reproducible_and_leaves_global_rng_alone():
    image = galaxy(200)
    np.random.seed(123)
    expected_global = np.random.random()

    np.random.seed(123)
    a = make_s2r(output_size=64, shot_noise_intensity=100, seed=7)(image, 1.0, gaussian_psf())
    b = make_s2r(output_size=64, shot_noise_intensity=100, seed=7)(image, 1.0, gaussian_psf())
    assert np.random.random() == expected_global  # global stream untouched
    np.testing.assert_array_equal(a, b)

    s2r = make_s2r(output_size=64, shot_noise_intensity=100, seed=7)
    first, second = s2r(image, 1.0, gaussian_psf()), s2r(image, 1.0, gaussian_psf())
    assert not np.array_equal(first, second)  # stream continues across calls
    np.testing.assert_array_equal(first, a)

    c = s2r(image, 1.0, gaussian_psf(), seed=99)
    d = s2r(image, 1.0, gaussian_psf(), seed=99)
    np.testing.assert_array_equal(c, d)


def test_shot_noise_on_calibrated_data_is_unclipped_with_poisson_variance():
    gain = 50.0
    s2r = make_s2r(output_size=64, conserve="flux", clip=None, shot_noise_intensity=gain, seed=3)
    stages = s2r.run(np.full((200, 200), 2.0), 1.0, delta_psf(), return_stages=True)
    clean, noisy = stages["cropped"], stages["noisy"]
    assert clean.max() > 1 and noisy.max() > 1  # physical units, not clipped to [0, 1]
    inside = clean > 0
    residual = (noisy - clean)[inside]
    assert abs(residual.mean()) < 0.05 * np.sqrt(clean[inside].mean() / gain)
    np.testing.assert_allclose(residual.var(), clean[inside].mean() / gain, rtol=0.1)
    np.testing.assert_array_equal(noisy[~inside], clean[~inside])  # no expected photons -> no noise


def test_noise_correlation_kernel_is_used_for_shot_noise():
    kernel = np.array([[0.0, 0.2, 0.0], [0.2, 1.0, 0.2], [0.0, 0.2, 0.0]])
    kw = dict(output_size=128, conserve="flux", clip=None, shot_noise_intensity=20.0)
    white = make_s2r(**kw, seed=1).run(np.full((200, 200), 1.0), 1.0, delta_psf(), return_stages=True)
    corr = make_s2r(**kw, seed=1, noise_correlation_kernel=kernel).run(
        np.full((200, 200), 1.0), 1.0, delta_psf(), return_stages=True
    )
    inner = np.s_[40:88, 40:88]  # inside the rebinned footprint
    def neighbour_corr(stages):
        r = (stages["noisy"] - stages["cropped"])[inner]
        return np.corrcoef(r[:, :-1].ravel(), r[:, 1:].ravel())[0, 1]
    assert abs(neighbour_corr(white)) < 0.1
    assert neighbour_corr(corr) > 0.25
