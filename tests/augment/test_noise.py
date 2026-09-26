import numpy as np
import pytest

from astro_datatools.augment import ShotNoiseAugment


def test_default_bounds_validate_and_clip():
    np.random.seed(0)
    image = np.full((50, 50), 0.99)
    out = ShotNoiseAugment(intensity=10).augment(image)
    assert out.min() >= 0 and out.max() <= 1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ShotNoiseAugment(intensity=10).augment(image * 2)


def test_unbounded_noise_is_zero_mean_with_poisson_variance():
    np.random.seed(1)
    gain = 200.0
    image = np.full((400, 400), 3.0)
    out = ShotNoiseAugment(intensity=gain, bounds=None).augment(image)
    residual = out - image
    assert out.max() > 1
    assert abs(residual.mean()) < 0.01 * np.sqrt(3.0 / gain)
    np.testing.assert_allclose(residual.var(), 3.0 / gain, rtol=0.05)


def test_unbounded_keeps_non_positive_pixels_unchanged():
    np.random.seed(2)
    image = np.array([[-0.01, 0.0], [0.5, 2.0]])
    out = ShotNoiseAugment(intensity=100, bounds=None).augment(image)
    assert out[0, 0] == -0.01 and out[0, 1] == 0.0


def test_rejects_non_positive_intensity():
    with pytest.raises(ValueError):
        ShotNoiseAugment(intensity=0)


# --------------------------------------------------------------------------- correlated noise
from astro_datatools.augment.noise import (  # noqa: E402
    _convolve_spatial,
    correlation_kernel_from_autocorrelation,
    estimate_correlation_kernel,
    kernel_autocorrelation,
    noise_autocorrelation,
)

TRUE_KERNEL = np.array([[0.02, 0.1, 0.02], [0.1, 1.0, 0.1], [0.02, 0.1, 0.02]])


def correlated_noise(shape=(600, 600), seed=0, floor_sigma=0.0):
    rng = np.random.default_rng(seed)
    noise = _convolve_spatial(rng.normal(size=shape), TRUE_KERNEL / np.sqrt((TRUE_KERNEL**2).sum()))
    if floor_sigma:
        noise = noise + floor_sigma * rng.normal()  # constant offset per image
    return noise


def test_kernel_reproduces_its_target_autocorrelation():
    target = np.array([[0.05, 0.2, 0.05], [0.2, 1.0, 0.2], [0.05, 0.2, 0.05]])
    kernel = correlation_kernel_from_autocorrelation(target, size=5)
    np.testing.assert_allclose((kernel**2).sum(), 1.0)
    np.testing.assert_allclose(kernel_autocorrelation(kernel, 1), target, atol=0.005)


def test_estimate_recovers_known_correlation_with_masked_pixels():
    stack = np.stack([correlated_noise(seed=s) for s in range(3)])
    mask = np.ones_like(stack, dtype=bool)
    mask[:, 100:200, 100:200] = False
    stack[:, 100:200, 100:200] = 1e3  # a masked "source" must not bias the estimate
    kernel, acf = estimate_correlation_kernel(stack, mask)
    expected = kernel_autocorrelation(TRUE_KERNEL, 1)
    np.testing.assert_allclose(acf, expected, atol=0.01)
    np.testing.assert_allclose(kernel_autocorrelation(kernel, 1), expected, atol=0.01)


def test_noise_autocorrelation_of_white_noise_is_a_delta():
    acf = noise_autocorrelation(np.random.default_rng(1).normal(size=(500, 500)), max_lag=2)
    expected = np.zeros((5, 5)); expected[2, 2] = 1
    np.testing.assert_allclose(acf, expected, atol=0.01)


def test_correlated_shot_noise_keeps_variance_and_mean_and_correlates_neighbours():
    np.random.seed(4)
    gain, level = 100.0, 5.0
    image = np.full((500, 500), level)
    out = ShotNoiseAugment(gain, bounds=None, correlation_kernel=TRUE_KERNEL).augment(image)
    residual = out - image
    assert abs(residual.mean()) < 0.01 * np.sqrt(level / gain)
    np.testing.assert_allclose(residual.var(), level / gain, rtol=0.05)
    acf = noise_autocorrelation(residual, max_lag=1, clip_sigma=None)
    np.testing.assert_allclose(acf, kernel_autocorrelation(TRUE_KERNEL, 1), atol=0.01)


def test_correlated_noise_works_channelwise_and_bounded():
    np.random.seed(5)
    image = np.full((3, 64, 64), 0.5)
    out = ShotNoiseAugment(1000, correlation_kernel=TRUE_KERNEL).augment(image)
    assert out.shape == image.shape and out.min() >= 0 and out.max() <= 1
    assert not np.allclose(out[0], out[1])  # channels get independent noise


def test_no_kernel_path_is_unchanged():
    image = np.random.default_rng(6).random((32, 32))
    np.random.seed(7)
    a = ShotNoiseAugment(50).augment(image)
    np.random.seed(7)
    b = np.clip(np.random.poisson(image * 50) / 50, 0, 1)
    np.testing.assert_array_equal(a, b)


def test_rejects_bad_kernels():
    with pytest.raises(ValueError):
        ShotNoiseAugment(10, correlation_kernel=np.zeros((3, 3)))
    with pytest.raises(ValueError):
        ShotNoiseAugment(10, correlation_kernel=np.ones(3))
