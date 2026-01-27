import numpy as np
from astropy.io import fits

from .reproject import reproject
from ..lotss_annotations.segmentation import Segment


class LoTSSComponent(Segment):
    def __init__(self, positions: list[tuple], nr_sigmas: float, rms: float):
        """
        Defines a LoTSS component for segmentation and component map generation.
        
        :param positions: (x, y) tuple representing pixel positions of the component.
        :type positions: list[tuple]
        :param nr_sigmas: Number of sigmas above the RMS for segmentation threshold.
        :type nr_sigmas: float
        :param rms: Root Mean Square noise level of the data.
        :type rms: float
        """
        super().__init__([positions], nr_sigmas, rms) # The positions need to be in a list for the parent class

    def get_component(self, data: np.ndarray) -> np.ndarray:
        """
        Generate a component map for the given data.

        :param data: 2D numpy array representing the data to segment.
        :type data: np.ndarray
        :return: 2D numpy array representing the component map.
        :rtype: np.ndarray
        """
        # We first get the segmentation labels, where 1 corresponds to the component
        # and 0 to the background
        labels = self.get_segmentation(data)

        # Now we set all pixel values outside the component to zero
        component_map = np.where(labels == 1, data, 0)
        return component_map


class LoTSSObject:
    """Class representing a LoTSS object."""
    def __init__(
            self,
            data: np.ndarray,
            header: fits.Header,
            positions: list[tuple],
            nr_sigmas: float = 5,
            rms: float = 0.1*1e-3,
            metadata: dict = {}
        ) -> None:
        """
        Initialize a LoTSS object.

        :param data: 2D numpy array representing the image data.
        :type data: np.ndarray
        :param positions: List of (x, y) tuples representing pixel positions of components.
        :type positions: list[tuple]
        :param nr_sigmas: Number of sigmas above the RMS for segmentation threshold.
        :type nr_sigmas: float
        :param rms: Root Mean Square noise level of the data.
        :type rms: float
        :param redshift: Redshift of the object if known.
        :type redshift: float, optional
        """
        self.data = data
        self.positions = positions
        self.nr_sigmas = nr_sigmas
        self.rms = rms
        self.metadata = metadata

        if "redshift" in self.metadata:
            self.redshift = self.metadata["redshift"]
        else:
            self.redshift = None

        self.header = header

        self.object_data = None  # To be generated when get_object is called

    def redshift_reproject(self, desired_redshift: float, alpha: float = -0.7) -> np.ndarray:
        """
        Reproject the LoTSS object to a desired redshift.

        :param desired_redshift: Target redshift for reprojection.
        :type desired_redshift: float
        :param alpha: Spectral index for K-correction (default is -0.7 for synchrotron emission).
        :type alpha: float
        :return: Reprojected image as a 2D numpy array.
        :rtype: np.ndarray
        """
        return reproject(self, desired_redshift, alpha)

    def get_object(self) -> np.ndarray:
        """
        Generate the object data by summing up all its components.
        The background pixels are set to zero.

        :return: 2D numpy array representing the object data.
        :rtype: np.ndarray
        """
        if self.object_data is not None:
            return self.object_data
        
        object_data = np.zeros_like(self.data)
        object_data = LoTSSComponent(self.positions, self.nr_sigmas, self.rms).get_component(self.data)
        self.object_data = object_data
        return object_data
        