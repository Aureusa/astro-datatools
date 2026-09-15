import numpy as np
import pytest

from astro_datatools.core.sim2real import sim2real


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_psf(size=3):
    """Create a simple normalized PSF."""
    psf = np.ones((size, size), dtype=float)
    return psf / psf.sum()


# ---------------------------------------------------------------------------
# Single image
# ---------------------------------------------------------------------------


def test_sim2real_single_image(cropped=False):
    rng = np.random.default_rng(42)

    image = rng.random((3, 32, 32))
    if cropped:
        background = np.full((3, 16, 16), 0.1)
    else:
        background = np.full((3, 32, 32), 0.1)
    psf = make_psf()

    result = sim2real(
        image,
        psf,
        background,
        crop_size=16,
        shot_noise_intensity=1000,
    )

    assert result.shape == (3, 16, 16)
    assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Batched images
# ---------------------------------------------------------------------------


def test_sim2real_batched_images(cropped=False):
    rng = np.random.default_rng(42)

    batch_size = 8
    channels = 3

    image = rng.random((batch_size, channels, 32, 32))
    if cropped:
        background = np.full(
            (batch_size, channels, 16, 16),
            0.1,
        )
    else:
        background = np.full(
            (batch_size, channels, 32, 32),
            0.1,
        )

    psf = make_psf()

    result = sim2real(
        image,
        psf,
        background,
        crop_size=16,
        shot_noise_intensity=1000,
    )

    assert result.shape == (batch_size, channels, 16, 16)
    assert np.all(np.isfinite(result))


def test_sim2real_batched_images_with_channel_psf(cropped=False):
    rng = np.random.default_rng(42)

    batch_size = 4
    channels = 3

    image = rng.random((batch_size, channels, 32, 32))
    if cropped:
        background = np.full(
            (batch_size, channels, 16, 16),
            0.1,
        )
    else:
        background = np.full(
            (batch_size, channels, 32, 32),
            0.1,
        )

    # One independent PSF per channel.
    psf = np.ones((channels, 3, 3), dtype=float)
    psf /= psf.sum(axis=(-2, -1), keepdims=True)

    result = sim2real(
        image,
        psf,
        background,
        crop_size=16,
        shot_noise_intensity=1000,
    )

    assert result.shape == (batch_size, channels, 16, 16)
    assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Batch consistency
# ---------------------------------------------------------------------------


def test_sim2real_batch_processes_each_image(cropped=False):
    """Test that sim2real correctly processes a batch of images."""
    rng = np.random.default_rng(42)

    images = rng.random((4, 3, 32, 32))
    if cropped:
        backgrounds = np.full((4, 3, 16, 16), 0.1)
    else:
        backgrounds = np.full((4, 3, 32, 32), 0.1)

    psf = make_psf()

    result = sim2real(
        images,
        psf,
        backgrounds,
        crop_size=16,
        shot_noise_intensity=1000,
    )

    # Batch dimension and crop size should be preserved correctly.
    assert result.shape == (4, 3, 16, 16)

    # Pipeline should not produce NaN or infinite values.
    assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Crop behavior
# ---------------------------------------------------------------------------


def test_sim2real_crop_is_applied_to_batch(cropped=False):
    rng = np.random.default_rng(42)

    image = rng.random((2, 3, 64, 64))
    if cropped:
        background = np.zeros((2, 3, 32, 32))
    else:
        background = np.zeros_like(image)

    result = sim2real(
        image,
        make_psf(),
        background,
        crop_size=32,
        shot_noise_intensity=1000,
    )

    assert result.shape == (2, 3, 32, 32)


# ---------------------------------------------------------------------------
# Background validation
# ---------------------------------------------------------------------------


def test_sim2real_rejects_background_with_wrong_shape(cropped=False):
    image = np.ones((4, 3, 32, 32))
    if cropped:
        background = np.ones((3, 16, 16))
    else:
        background = np.ones((3, 32, 32))

    with pytest.raises(
        ValueError,
        match="Image and background must have the same dimensions",
    ):
        sim2real(
            image,
            make_psf(),
            background,
            crop_size=16,
        )


# ---------------------------------------------------------------------------
# Invalid input dimensions
# ---------------------------------------------------------------------------


def test_sim2real_rejects_2d_image(cropped=False):
    image = np.ones((32, 32))
    if cropped:
        background = np.ones((16, 16))
    else:
        background = np.ones((32, 32))

    with pytest.raises(
        ValueError,
        match="Expected 3.*or 4",
    ):
        sim2real(
            image,
            make_psf(),
            background,
            crop_size=16,
        )


def test_sim2real_rejects_5d_image(cropped=False):
    image = np.ones((2, 3, 4, 32, 32))
    if cropped:
        background = np.ones((2, 3, 4, 16, 16))
    else:
        background = np.ones_like(image)

    with pytest.raises(
        ValueError,
        match="Expected 3.*or 4",
    ):
        sim2real(
            image,
            make_psf(),
            background,
            crop_size=16,
        )


# ---------------------------------------------------------------------------
# Different embedding modes
# ---------------------------------------------------------------------------


def test_sim2real_fill_embedding(cropped=False):
    rng = np.random.default_rng(42)

    image = rng.random((2, 3, 32, 32))
    if cropped:
        background = np.full((2, 3, 16, 16), 0.1)
    else:
        background = np.full_like(image, 0.1)

    result = sim2real(
        image,
        make_psf(),
        background,
        crop_size=16,
        shot_noise_intensity=1000,
        embed_type="fill",
    )

    assert result.shape == (2, 3, 16, 16)
    assert np.all(np.isfinite(result))


def test_sim2real_default_embedding(cropped=False):
    rng = np.random.default_rng(42)

    image = rng.random((2, 3, 32, 32))
    if cropped:
        background = np.full((2, 3, 16, 16), 0.1)
    else:
        background = np.full_like(image, 0.1)

    result = sim2real(
        image,
        make_psf(),
        background,
        crop_size=16,
        shot_noise_intensity=1000,
        embed_type="default",
    )

    assert result.shape == (2, 3, 16, 16)
    assert np.all(np.isfinite(result))

def test_sim2real_all_cropped():
    """Test all sim2real functions with cropped background images."""
    test_sim2real_single_image(cropped=True)
    test_sim2real_rejects_2d_image(cropped=True)
    test_sim2real_rejects_5d_image(cropped=True)
    test_sim2real_fill_embedding(cropped=True)
    test_sim2real_default_embedding(cropped=True)
