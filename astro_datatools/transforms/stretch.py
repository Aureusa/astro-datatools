"""
Module providing various data stretching functions for astronomical data processing.
These functions are motivated by common practices in computer vision and astronomy.
They do not perform any normalization; they only stretch the data.
"""
import numpy as np
import logging


logger = logging.getLogger(__name__)


def sqrt_stretch(data: np.ndarray, min_val: float = 0, max_val: float = None) -> np.ndarray:
    """
    Apply square-root stretch to the data between min_val and max_val.

    :param data: Input data array.
    :type data: np.ndarray
    :param min_val: Minimum value for stretching (default: 0).
    :type min_val: float
    :param max_val: Maximum value for stretching (default: None, which means no upper limit).
    :type max_val: float
    :return: Stretched data array.
    :rtype: np.ndarray
    """
    return np.sqrt(np.clip(data, min_val, max_val))


def asinh_stretch(data: np.ndarray, a: float) -> np.ndarray:
    """
    Apply asinh stretch to the data.

    :param data: Input data array.
    :type data: np.ndarray
    :param a: Scaling factor for asinh stretch.
    :type a: float
    """
    return np.arcsinh(data / a)


def linear_stretch(data: np.ndarray, min_val: float = -np.inf, max_val: float = np.inf) -> np.ndarray:
    """
    Apply linear stretch to the data between min_val and max_val.

    :param data: Input data array.
    :type data: np.ndarray
    :param min_val: Minimum value for stretching (default: 0).
    :type min_val: float
    :param max_val: Maximum value for stretching (default: 1).
    :type max_val: float
    :return: Stretched data array.
    :rtype: np.ndarray
    """
    return np.clip(data, min_val, max_val)


def log_like_stretch(data: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Apply log-like stretch to the data.

    :param data: Input data array.
    :type data: np.ndarray
    :param epsilon: Small constant to avoid log(0).
    :type epsilon: float
    :return: Stretched data array.
    :rtype: np.ndarray
    """
    return np.log10(data + epsilon)


def power_stretch(data: np.ndarray, exponent: float) -> np.ndarray:
    """
    Apply power-law stretch to the data.

    :param data: Input data array.
    :type data: np.ndarray
    :param exponent: Exponent for power-law stretch.
    :type exponent: float
    :return: Stretched data array.
    :rtype: np.ndarray
    """
    return np.power(data, exponent)


def z_score_stretch(data: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization to the data.

    :param data: Input data array.
    :type data: np.ndarray
    :return: Z-score normalized data array.
    :rtype: np.ndarray
    """
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


def robust_z_score_stretch(data: np.ndarray) -> np.ndarray:
    """
    Apply robust z-score normalization to the data.

    :param data: Input data array.
    :type data: np.ndarray
    :return: Robust z-score normalized data array.
    :rtype: np.ndarray
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))

    if mad == 0:
        logger.warning("MAD is zero in robust z-score stretch. Returning zeros array.")
        return np.zeros_like(data)

    return (data - median) / mad


def quantile_stretch(data: np.ndarray, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> np.ndarray:
    """
    Apply quantile stretch to the data.

    :param data: Input data array.
    :type data: np.ndarray
    :param lower_quantile: Lower quantile for stretching (default: 0.01).
    :type lower_quantile: float
    :param upper_quantile: Upper quantile for stretching (default: 0.99).
    :type upper_quantile: float
    :return: Stretched data array.
    :rtype: np.ndarray
    """
    nominator = data - np.quantile(data, lower_quantile)
    denominator = np.quantile(data, upper_quantile) - np.quantile(data, lower_quantile)

    # Protect against division by zero
    if denominator == 0:
        logger.warning("Denominator in quantile stretch is zero. Returning zeros array.")
        return np.zeros_like(data)
    
    return nominator / denominator
