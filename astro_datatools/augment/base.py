from abc import ABC, abstractmethod


class BaseAugment(ABC):
    """Abstract base class for data augmentation operations."""

    @abstractmethod
    def augment(self, data):
        """Apply augmentation to the data.

        Parameters
        ----------
        data : array-like
            The input data to augment.

        Returns
        -------
        augmented_data : array-like
            The augmented data.
        """
        pass
    