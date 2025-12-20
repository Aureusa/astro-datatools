from abc import ABC, abstractmethod


class BaseIO(ABC):
    """Abstract base class for input/output operations."""

    @abstractmethod
    def read(self, filepath):
        """Read data from the specified file path."""
        pass

    @abstractmethod
    def write(self, filepath, data):
        """Write data to the specified file path."""
        pass
    