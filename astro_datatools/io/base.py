"""Base interfaces for AstroIO readers, writers, and format handlers."""
from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseReader(ABC):
    """Abstract base class for all file format readers."""

    @abstractmethod
    def read(self, filepath: str, **kwargs) -> Any:
        """Read data from the specified file path.

        :param filepath: Path to the file to read.
        :type filepath: str
        :param kwargs: Additional format-specific keyword arguments.
        :return: Data read from the file.
        """
        pass


class BaseWriter(ABC):
    """Abstract base class for all file format writers."""

    @abstractmethod
    def write(self, filepath: str, data: Any, **kwargs) -> None:
        """Write data to the specified file path.

        :param filepath: Path to the file to write.
        :type filepath: str
        :param data: Data to write to the file.
        :param kwargs: Additional format-specific keyword arguments.
        """
        pass


class BaseIOHandler(ABC):
    """Base class for format-specific IO handlers grouping a reader, writer,

    and format-specific convenience operations.
    """

    reader: Optional[BaseReader] = None
    writer: Optional[BaseWriter] = None

    def read(self, filepath: str, **kwargs) -> Any:
        """Read data using the registered reader."""
        if self.reader is None:
            raise NotImplementedError(f"{self.__class__.__name__} has no reader configured.")
        return self.reader.read(filepath, **kwargs)

    def write(self, filepath: str, data: Any, **kwargs) -> None:
        """Write data using the registered writer."""
        if self.writer is None:
            raise NotImplementedError(f"{self.__class__.__name__} has no writer configured.")
        self.writer.write(filepath, data, **kwargs)
