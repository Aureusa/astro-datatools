"""NumPy .npy and .npz I/O handler, reader, and writer."""
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .base import BaseIOHandler, BaseReader, BaseWriter
from .registry import register_handler

NUMPY_EXTENSIONS = [".npy", ".npz"]


class NpyReader(BaseReader):
    """Reader for NumPy .npy and .npz files."""

    def read(self, filepath: str, **kwargs: Any) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Read a NumPy array or archive from disk.

        :param filepath: Path to .npy or .npz file.
        :param kwargs: Arguments passed to `np.load`.
        :return: np.ndarray or NpzFile mapping.
        """
        data = np.load(filepath, **kwargs)
        if filepath.endswith(".npz"):
            return {k: data[k] for k in data.files}
        return data


class NpyWriter(BaseWriter):
    """Writer for NumPy .npy and .npz files."""

    def write(
        self,
        filepath: str,
        data: Union[np.ndarray, Dict[str, np.ndarray]],
        compressed: bool = False,
        **kwargs: Any,
    ) -> None:
        """Write a NumPy array or dictionary of arrays to disk.

        :param filepath: Path to output file.
        :param data: Array or dict of arrays.
        :param compressed: If True and writing .npz, uses np.savez_compressed.
        :param kwargs: Additional arguments passed to np.save/savez.
        """
        if filepath.endswith(".npz"):
            if isinstance(data, dict):
                if compressed:
                    np.savez_compressed(filepath, **data)
                else:
                    np.savez(filepath, **data)
            else:
                if compressed:
                    np.savez_compressed(filepath, data=data)
                else:
                    np.savez(filepath, data=data)
        else:
            if not isinstance(data, np.ndarray):
                data = np.asarray(data)
            np.save(filepath, data, **kwargs)


@register_handler(NUMPY_EXTENSIONS)
class NpyIO(BaseIOHandler):
    """Handler for NumPy array and archive I/O."""

    def __init__(self):
        self.reader = NpyReader()
        self.writer = NpyWriter()

    def save_npz(self, filepath: str, compressed: bool = True, **arrays: np.ndarray) -> None:
        """Convenience method to save multiple named arrays into an .npz archive."""
        if compressed:
            np.savez_compressed(filepath, **arrays)
        else:
            np.savez(filepath, **arrays)

    def load_npz(self, filepath: str) -> Dict[str, np.ndarray]:
        """Convenience method to load all arrays from an .npz archive into a dict."""
        with np.load(filepath) as npz:
            return {k: npz[k] for k in npz.files}
