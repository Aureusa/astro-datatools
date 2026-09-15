import numpy as np
import pytest

from astro_datatools.transforms.embed import embed_in_background


# ---------------------------------------------------------------------------
# default embedding
# ---------------------------------------------------------------------------


def test_embed_default_adds_image_and_background():
    image = np.array([
        [1, 2],
        [3, 4],
    ])

    background = np.array([
        [10, 20],
        [30, 40],
    ])

    result = embed_in_background(
        image,
        background,
        embed_type="default",
    )

    expected = np.array([
        [11, 22],
        [33, 44],
    ])

    np.testing.assert_array_equal(result, expected)


def test_embed_default_is_same_as_addition():
    image = np.random.default_rng(42).random((10, 10))
    background = np.random.default_rng(43).random((10, 10))

    result = embed_in_background(
        image,
        background,
        embed_type="default",
    )

    np.testing.assert_array_equal(
        result,
        image + background,
    )


# ---------------------------------------------------------------------------
# fill embedding
# ---------------------------------------------------------------------------


def test_embed_fill_replaces_nonzero_pixels():
    image = np.array([
        [1, 0],
        [0, 4],
    ])

    background = np.array([
        [10, 20],
        [30, 40],
    ])

    result = embed_in_background(
        image,
        background,
        embed_type="fill",
    )

    expected = np.array([
        [1, 20],
        [30, 4],
    ])

    np.testing.assert_array_equal(result, expected)


def test_embed_fill_preserves_background_at_zero_pixels():
    image = np.zeros((3, 3))
    background = np.arange(9).reshape(3, 3)

    result = embed_in_background(
        image,
        background,
        embed_type="fill",
    )

    np.testing.assert_array_equal(result, background)


def test_embed_fill_preserves_all_image_pixels_when_nonzero():
    image = np.ones((3, 3))
    background = np.full((3, 3), 100)

    result = embed_in_background(
        image,
        background,
        embed_type="fill",
    )

    np.testing.assert_array_equal(result, image)


def test_embed_fill_works_with_negative_values():
    image = np.array([
        [-1, 0],
        [2, 0],
    ])

    background = np.array([
        [10, 20],
        [30, 40],
    ])

    result = embed_in_background(
        image,
        background,
        embed_type="fill",
    )

    expected = np.array([
        [-1, 20],
        [2, 40],
    ])

    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_embed_rejects_different_shapes():
    image = np.zeros((10, 10))
    background = np.zeros((8, 8))

    with pytest.raises(
        ValueError,
        match="Image and background must have the same shape",
    ):
        embed_in_background(image, background)


def test_embed_rejects_unknown_type():
    image = np.zeros((10, 10))
    background = np.zeros((10, 10))

    with pytest.raises(
        ValueError,
        match="Unsupported embedding type",
    ):
        embed_in_background(
            image,
            background,
            embed_type="unknown",
        )


def test_embed_works_with_channels():
    image = np.zeros((3, 10, 10))
    background = np.ones((3, 10, 10))

    result = embed_in_background(
        image,
        background,
        embed_type="fill",
    )

    np.testing.assert_array_equal(result, background)


def test_embed_preserves_shape():
    image = np.random.default_rng(42).random((3, 20, 20))
    background = np.random.default_rng(43).random((3, 20, 20))

    result = embed_in_background(image, background)

    assert result.shape == image.shape
