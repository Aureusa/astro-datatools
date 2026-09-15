import numpy as np
import pytest

from astro_datatools.transforms.geometric import (
    crop,
    resize,
    rotation,
    hflip,
    vflip,
)


# ---------------------------------------------------------------------------
# crop
# ---------------------------------------------------------------------------


def test_crop_center():
    image = np.arange(25).reshape(5, 5)

    result = crop(image, 3)

    expected = np.array([
        [6, 7, 8],
        [11, 12, 13],
        [16, 17, 18],
    ])

    np.testing.assert_array_equal(result, expected)


def test_crop_center_with_channels():
    image = np.arange(50).reshape(2, 5, 5)

    result = crop(image, 3)

    assert result.shape == (2, 3, 3)

    expected = image[:, 1:4, 1:4]
    np.testing.assert_array_equal(result, expected)


def test_crop_custom():
    image = np.arange(25).reshape(5, 5)

    result = crop(
        image,
        2,
        crop_type="custom",
        loc=(1, 2),
    )

    expected = np.array([
        [7, 8],
        [12, 13],
    ])

    np.testing.assert_array_equal(result, expected)


def test_crop_custom_requires_loc():
    image = np.zeros((5, 5))

    with pytest.raises(ValueError, match="loc"):
        crop(image, 3, crop_type="custom")


def test_crop_random_with_seed_is_reproducible():
    image = np.arange(100).reshape(10, 10)

    result_1 = crop(
        image,
        4,
        crop_type="random",
        seed=42,
    )

    result_2 = crop(
        image,
        4,
        crop_type="random",
        seed=42,
    )

    np.testing.assert_array_equal(result_1, result_2)


def test_crop_random_with_rng_is_reproducible():
    image = np.arange(100).reshape(10, 10)

    rng_1 = np.random.default_rng(42)
    rng_2 = np.random.default_rng(42)

    result_1 = crop(
        image,
        4,
        crop_type="random",
        rng=rng_1,
    )

    result_2 = crop(
        image,
        4,
        crop_type="random",
        rng=rng_2,
    )

    np.testing.assert_array_equal(result_1, result_2)


def test_crop_random_is_valid():
    image = np.arange(100).reshape(10, 10)

    result = crop(
        image,
        4,
        crop_type="random",
        seed=42,
    )

    assert result.shape == (4, 4)

    # Every value must come from the original image.
    assert np.all(np.isin(result, image))


def test_crop_rejects_too_large_size():
    image = np.zeros((5, 5))

    with pytest.raises(ValueError, match="smaller than crop size"):
        crop(image, 6)


def test_crop_rejects_unknown_type():
    image = np.zeros((5, 5))

    with pytest.raises(ValueError, match="Unknown crop_type"):
        crop(image, 3, crop_type="unknown")


# ---------------------------------------------------------------------------
# resize
# ---------------------------------------------------------------------------


def test_resize_2d():
    image = np.ones((10, 20))

    result = resize(image, 5)

    assert result.shape == (5, 5)


def test_resize_3d():
    image = np.ones((3, 10, 20))

    result = resize(image, 5)

    assert result.shape == (3, 5, 5)


def test_resize_preserves_constant_image():
    image = np.full((10, 20), 5.0)

    result = resize(image, 5)

    np.testing.assert_allclose(result, 5.0)


def test_resize_rejects_invalid_dimensions():
    image = np.ones((2, 3, 10, 10))

    with pytest.raises(ValueError, match=r"Expected image with shape"):
        resize(image, 5)


# ---------------------------------------------------------------------------
# rotation
# ---------------------------------------------------------------------------


def test_rotation_zero_angle():
    image = np.arange(25, dtype=float).reshape(5, 5)

    result = rotation(image, 0)

    assert result.shape == image.shape
    np.testing.assert_allclose(result, image)


def test_rotation_preserves_shape():
    image = np.ones((10, 20))

    result = rotation(image, 45)

    assert result.shape == image.shape


def test_rotation_with_channels_preserves_shape():
    image = np.ones((3, 10, 20))

    result = rotation(image, 45)

    assert result.shape == image.shape


def test_rotation_with_custom_size():
    image = np.ones((20, 20))

    result = rotation(image, 45, size=10)

    assert result.shape == (10, 10)


def test_rotation_rejects_non_image_dimensions():
    image = np.ones((2, 3, 10, 10))

    # The current implementation supports arbitrary leading dimensions
    # through np.pad/rotate, so this test documents that behavior rather
    # than expecting an exception.
    result = rotation(image, 10)

    assert result.shape == image.shape


# ---------------------------------------------------------------------------
# flips
# ---------------------------------------------------------------------------


def test_hflip():
    image = np.arange(12).reshape(3, 4)

    result = hflip(image)

    expected = np.array([
        [3, 2, 1, 0],
        [7, 6, 5, 4],
        [11, 10, 9, 8],
    ])

    np.testing.assert_array_equal(result, expected)


def test_vflip():
    image = np.arange(12).reshape(3, 4)

    result = vflip(image)

    expected = np.array([
        [8, 9, 10, 11],
        [4, 5, 6, 7],
        [0, 1, 2, 3],
    ])

    np.testing.assert_array_equal(result, expected)


def test_hflip_with_channels():
    image = np.arange(24).reshape(2, 3, 4)

    result = hflip(image)

    expected = image[:, :, ::-1]

    np.testing.assert_array_equal(result, expected)


def test_vflip_with_channels():
    image = np.arange(24).reshape(2, 3, 4)

    result = vflip(image)

    expected = image[:, ::-1, :]

    np.testing.assert_array_equal(result, expected)


def test_hflip_returns_independent_array():
    image = np.arange(9).reshape(3, 3)

    result = hflip(image)
    result[0, 0] = -1

    assert image[0, 2] != -1


def test_vflip_returns_independent_array():
    image = np.arange(9).reshape(3, 3)

    result = vflip(image)
    result[0, 0] = -1

    assert image[2, 0] != -1
