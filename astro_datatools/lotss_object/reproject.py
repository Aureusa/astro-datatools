from skimage.transform import resize

from .cosmological_distances import luminosity_distance, angular_diameter_distance
from .lotss_object import LoTSSObject
from .utils import jyperbeam_to_jyperpixel, jyperpixel_to_jyperbeam


def _flux_scaling_factor(z1, z2, alpha=-0.7):
    """
    Calculate the flux scaling factor when reprojecting from redshift z1 to z2.

    The scaling factor is given by:
    (D_L1 / D_L2)^2 * ((1 + z2) / (1 + z1))^(alpha + 1)
    where:
    D_L: Luminosity distance
    alpha: Spectral index for K-correction (default is -0.7 for synchrotron emission)
    z: Redshift
    
    Parameters:
    - z1: Original redshift
    - z2: Target redshift
    - alpha: Spectral index (default is -0.7 for synchrotron emission)
    
    Returns:
    - Scaling factor to apply to the flux
    - D_L2: Luminosity distance at z2
    """
    D_L1 = luminosity_distance(z1)  # in Mpc
    D_L2 = luminosity_distance(z2)  # in Mpc

    # K_CORRECTION = ((1 + z2) / (1 + z1))**(alpha + 1)

    scaling_factor = (D_L1 / D_L2)**2# * K_CORRECTION

    return scaling_factor

def _angular_scaling_factor(z1, z2):
    """
    Calculate the angular scaling factor when reprojecting from redshift z1 to z2.

    The angular scaling factor is given by:
    D_A1 / D_A2
    where:
    D_A: Angular diameter distance
    
    Parameters:
    - z1: Original redshift
    - z2: Target redshift
    
    Returns:
    - Scaling factor to apply to the angular size
    """
    D_A1 = angular_diameter_distance(z1)  # in Mpc
    D_A2 = angular_diameter_distance(z2)  # in Mpc
    
    scaling_factor = D_A1 / D_A2

    return scaling_factor

def reproject(lotss_object: LoTSSObject, desired_redshift, alpha=-0.7):
    """
    Reproject a LoTSSObject to a desired redshift.

    The reprojected image accounts for both the angular and flux scaling due to the change in redshift.
    
    Here is a recepie of the steps involved:
    1. Ensure the image is in Jy/pixel units.
    2. Calculate the angular scaling factor S_ANG=(D_A1 / D_A2).
    3. Calculate the flux scaling factor S_FLUX=((D_L1 / D_L2)^2 * ((1 + z2) / (1 + z1))^(alpha + 1)).
    4. Resample the image to the new angular scale using bilinear interpolation, assuming the original pixel values are preserved.
       - New shape = (original_shape[0] * S_ANG, original_shape[1] * S_ANG)
    5. Adjust the flux values in the resampled image:
       - Each pixel value is multiplied by S_FLUX / (S_ANG^2) to account for both cosmological dimming and the change in pixel area.
    6. Normalize the final image to ensure total flux is conserved.
    7. Return the reprojected image along with original and new luminosity distances for reference.

    Theory:
    We need to perserve the intrinsic luminosity (power) of the source while accounting for how it appears at different redshifts.
    The intrinsic luminosity L_nu is related to the observed flux density S_nu by:

    L_nu = 4 * pi * D_L^2 * S_nu * (1 + z)^(alpha - 1)

    where D_L is the luminosity distance, z is the redshift, and alpha is the spectral index.
    When changing redshift from z1 to z2, we want to ensure that the intrinsic luminosity remains the same.
    Therefore, we adjust the observed flux density S_nu according to the change in luminosity distance and the K-correction factor.

    Additionally, the angular size of the source changes with redshift, which affects how the image is sampled.
    The angular diameter distance D_A relates the physical size of an object to its angular size on the sky.
    The angular size theta of an object of physical size l is given by:

    theta = l / D_A

    Thus, when reprojecting the image, we need to resample it according to the change in angular diameter distance.
    The new angular size theta' at redshift z2 is given by:

    theta' = theta * (D_A2 / D_A1)

    where D_A1 and D_A2 are the angular diameter distances at z1 and z2, respectively.

    :param lotss_object: LoTSSObject to be reprojected
    :type lotss_object: LoTSSObject
    :param desired_redshift: Target redshift for reprojection
    :type desired_redshift: float
    :param alpha: Spectral index for K-correction (default is -0.7 for synchrotron emission)
    :type alpha: float
    :return: Reprojected image as a 2D numpy array
    :rtype: np.ndarray
    """
    # Get the image data and the header from the LoTSSObject
    image = lotss_object.get_object()
    header = lotss_object.header

    image, header = jyperbeam_to_jyperpixel(image, header)  # Convert to Jy/pixel

    # Define the redshift of the original object and the desired redshift
    z1 = lotss_object.redshift if lotss_object.redshift is not None else 0.0
    z2 = desired_redshift

    # Compute scaling factors
    flx_scaling_factor = _flux_scaling_factor(z1, z2, alpha=alpha)
    ang_scaling_factor = _angular_scaling_factor(z1, z2)

    # Zoom the image to the new angular scale
    new_shape = (int(image.shape[0] * ang_scaling_factor), int(image.shape[1] * ang_scaling_factor))
    resampled_image = resize(image, new_shape, order=1, mode='edge', anti_aliasing=True)

    # Account for lost flux due to change in pixel area
    resampled_image /= ang_scaling_factor**2

    # Assume that there is some loss due to interpolation and numerical effects
    total_flux_original = image.sum()
    total_flux_resampled = resampled_image.sum()
    resampled_image *= (total_flux_original / total_flux_resampled)

    # Scale the image
    scaled_image = resampled_image * flx_scaling_factor

    # Convert back to Jy/beam if needed
    scaled_image, _ = jyperpixel_to_jyperbeam(scaled_image, header)
    return scaled_image
