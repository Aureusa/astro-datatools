"""HDF5 I/O handler, reader, writer, and astronomy dataset utilities."""
from typing import Any, Dict, List, Optional, Union
import h5py
import numpy as np

from .base import BaseIOHandler, BaseReader, BaseWriter
from .registry import register_handler

HDF5_EXTENSIONS = [".hdf5", ".h5", ".hdf", ".he5"]


def _recursively_read_group(group: Union[h5py.File, h5py.Group], load_attrs: bool = True) -> Dict[str, Any]:
    """Recursively read an HDF5 group into a nested dictionary."""
    result = {}
    if load_attrs and len(group.attrs) > 0:
        result["__attrs__"] = {k: v for k, v in group.attrs.items()}

    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            val = item[()]
            if load_attrs and len(item.attrs) > 0:
                result[key] = {
                    "data": val,
                    "__attrs__": {ak: av for ak, av in item.attrs.items()},
                }
            else:
                result[key] = val
        elif isinstance(item, h5py.Group):
            result[key] = _recursively_read_group(item, load_attrs=load_attrs)
    return result


def _recursively_write_group(
    group: Union[h5py.File, h5py.Group],
    data: Dict[str, Any],
    compression: Optional[str] = "gzip",
    **kwargs: Any,
) -> None:
    """Recursively write a dictionary structure into an HDF5 group."""
    for key, value in data.items():
        if key == "__attrs__" and isinstance(value, dict):
            for ak, av in value.items():
                group.attrs[ak] = av
        elif isinstance(value, dict):
            # Check if it's a dataset dict with {'data': ..., '__attrs__': ...}
            if "data" in value and isinstance(value.get("__attrs__"), dict):
                dset_data = np.asarray(value["data"])
                dset = group.create_dataset(key, data=dset_data, compression=compression, **kwargs)
                for ak, av in value["__attrs__"].items():
                    dset.attrs[ak] = av
            else:
                subgroup = group.create_group(key)
                _recursively_write_group(subgroup, value, compression=compression, **kwargs)
        elif isinstance(value, (np.ndarray, list, tuple)):
            arr = np.asarray(value)
            group.create_dataset(key, data=arr, compression=compression, **kwargs)
        elif isinstance(value, (int, float, str, bytes, bool, np.number)):
            # Store scalar as dataset or attribute
            group.create_dataset(key, data=value)
        else:
            raise TypeError(f"Unsupported value type for HDF5 key '{key}': {type(value)}")


class HDF5Reader(BaseReader):
    """Reader for astronomical HDF5 files."""

    def read(
        self,
        filepath: str,
        key: Optional[str] = None,
        as_dict: bool = False,
        load_attrs: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Read data from an HDF5 file.

        :param filepath: Path to the HDF5 file.
        :param key: Optional specific dataset or group path (e.g. '/data/image').
        :param as_dict: If True, recursively loads all contents into a python dict.
        :param load_attrs: Whether to include attributes when as_dict=True.
        :param kwargs: Additional arguments passed to h5py.File.
        :return: Numpy array, nested dict, or open h5py.File handle.
        """
        if key is not None:
            with h5py.File(filepath, "r", **kwargs) as f:
                target = f[key]
                if isinstance(target, h5py.Dataset):
                    return target[()]
                elif isinstance(target, h5py.Group):
                    return _recursively_read_group(target, load_attrs=load_attrs)

        if as_dict:
            with h5py.File(filepath, "r", **kwargs) as f:
                return _recursively_read_group(f, load_attrs=load_attrs)

        # Return open file handle for streaming/slicing
        return h5py.File(filepath, "r", **kwargs)


class HDF5Writer(BaseWriter):
    """Writer for astronomical HDF5 files."""

    def write(
        self,
        filepath: str,
        data: Union[Dict[str, Any], np.ndarray],
        mode: str = "w",
        compression: Optional[str] = "gzip",
        **kwargs: Any,
    ) -> None:
        """Write data to an HDF5 file.

        :param filepath: Destination filepath.
        :param data: Dictionary of datasets/groups or a single numpy array.
        :param mode: File open mode ('w', 'w-', 'x', 'a'). Default is 'w'.
        :param compression: Dataset compression filter (e.g., 'gzip', 'lzf', None).
        :param kwargs: Additional arguments passed to create_dataset.
        """
        with h5py.File(filepath, mode) as f:
            if isinstance(data, dict):
                _recursively_write_group(f, data, compression=compression, **kwargs)
            elif isinstance(data, (np.ndarray, list, tuple)):
                arr = np.asarray(data)
                f.create_dataset("data", data=arr, compression=compression, **kwargs)
            else:
                raise TypeError(
                    f"Unsupported data type for HDF5 writer: {type(data)}. "
                    f"Expected dict or numpy.ndarray."
                )


@register_handler(HDF5_EXTENSIONS)
class HDF5IO(BaseIOHandler):
    """High-level astronomical HDF5 handler with common query and manipulation methods."""

    def __init__(self):
        self.reader = HDF5Reader()
        self.writer = HDF5Writer()

    def open(self, filepath: str, mode: str = "r", **kwargs: Any) -> h5py.File:
        """Open an HDF5 file as a context manager for streaming/large data operations.

        Usage:
            with hdf5_io.open('cube.h5') as f:
                slice_data = f['spectra'][:, 10:20]
        """
        return h5py.File(filepath, mode=mode, **kwargs)

    def read_dataset(self, filepath: str, dataset_path: str, **kwargs: Any) -> np.ndarray:
        """Read a single dataset from HDF5 as a numpy array.

        :param filepath: Path to HDF5 file.
        :param dataset_path: Internal path to dataset (e.g. 'images/radio').
        :return: numpy.ndarray
        """
        with h5py.File(filepath, "r", **kwargs) as f:
            return f[dataset_path][()]

    def write_dataset(
        self,
        filepath: str,
        dataset_path: str,
        data: np.ndarray,
        attrs: Optional[Dict[str, Any]] = None,
        mode: str = "a",
        compression: Optional[str] = "gzip",
        **kwargs: Any,
    ) -> None:
        """Write or overwrite a specific dataset in an HDF5 file.

        :param filepath: Path to HDF5 file.
        :param dataset_path: Internal path for dataset.
        :param data: Data array.
        :param attrs: Optional dictionary of metadata attributes for this dataset.
        :param mode: Open mode ('a' to append/update, 'w' to overwrite file).
        :param compression: Compression algorithm ('gzip', 'lzf', etc.).
        """
        with h5py.File(filepath, mode) as f:
            if dataset_path in f:
                del f[dataset_path]
            dset = f.create_dataset(dataset_path, data=np.asarray(data), compression=compression, **kwargs)
            if attrs:
                for k, v in attrs.items():
                    dset.attrs[k] = v

    def read_attrs(self, filepath: str, path: str = "/", **kwargs: Any) -> Dict[str, Any]:
        """Read metadata attributes of a group or dataset.

        :param filepath: Path to HDF5 file.
        :param path: Internal path to group or dataset (default '/').
        :return: Dictionary of attribute key-value pairs.
        """
        with h5py.File(filepath, "r", **kwargs) as f:
            item = f[path]
            return {k: v for k, v in item.attrs.items()}

    def write_attrs(self, filepath: str, attrs: Dict[str, Any], path: str = "/", mode: str = "a") -> None:
        """Write metadata attributes to a group or dataset.

        :param filepath: Path to HDF5 file.
        :param attrs: Dictionary of attributes.
        :param path: Internal path to group or dataset.
        :param mode: File open mode.
        """
        with h5py.File(filepath, mode) as f:
            item = f[path]
            for k, v in attrs.items():
                item.attrs[k] = v

    def read_dict(self, filepath: str, group_path: str = "/", load_attrs: bool = True, **kwargs: Any) -> Dict[str, Any]:
        """Read the entire HDF5 file or group into a python dictionary of numpy arrays and metadata.

        :param filepath: Path to HDF5 file.
        :param group_path: Group path to start reading from.
        :param load_attrs: Whether to load metadata attributes into '__attrs__'.
        :return: Nested dictionary.
        """
        with h5py.File(filepath, "r", **kwargs) as f:
            target = f[group_path] if group_path != "/" else f
            return _recursively_read_group(target, load_attrs=load_attrs)

    def write_dict(
        self,
        filepath: str,
        data_dict: Dict[str, Any],
        mode: str = "w",
        compression: Optional[str] = "gzip",
        **kwargs: Any,
    ) -> None:
        """Write a nested dictionary of numpy arrays and metadata to an HDF5 file.

        :param filepath: Destination filepath.
        :param data_dict: Nested dictionary of data.
        :param mode: File open mode ('w', 'a').
        :param compression: Compression type.
        """
        self.writer.write(filepath, data_dict, mode=mode, compression=compression, **kwargs)

    def list_keys(self, filepath_or_file: Union[str, h5py.File, h5py.Group], path: str = "/") -> List[str]:
        """List keys (datasets and subgroups) at a given path in the HDF5 file."""
        if isinstance(filepath_or_file, str):
            with h5py.File(filepath_or_file, "r") as f:
                target = f[path] if path != "/" else f
                return list(target.keys())
        target = filepath_or_file[path] if path != "/" else filepath_or_file
        return list(target.keys())

    def tree(self, filepath_or_file: Union[str, h5py.File, h5py.Group]) -> str:
        """Generate an ASCII tree view of the HDF5 hierarchy with shapes and attribute info."""
        lines = []

        def _walk(item: Union[h5py.File, h5py.Group, h5py.Dataset], prefix: str = ""):
            if isinstance(item, (h5py.File, h5py.Group)):
                keys = list(item.keys())
                for i, k in enumerate(keys):
                    is_last = (i == len(keys) - 1)
                    connector = "└── " if is_last else "├── "
                    child = item[k]
                    if isinstance(child, h5py.Dataset):
                        attr_info = f" [{len(child.attrs)} attrs]" if len(child.attrs) > 0 else ""
                        lines.append(f"{prefix}{connector}{k}: Dataset shape={child.shape}, dtype={child.dtype}{attr_info}")
                    else:
                        attr_info = f" [{len(child.attrs)} attrs]" if len(child.attrs) > 0 else ""
                        lines.append(f"{prefix}{connector}{k}/ (Group){attr_info}")
                        new_prefix = prefix + ("    " if is_last else "│   ")
                        _walk(child, new_prefix)

        if isinstance(filepath_or_file, str):
            with h5py.File(filepath_or_file, "r") as f:
                lines.append(f"{filepath_or_file} (HDF5 Root) [{len(f.attrs)} attrs]")
                _walk(f)
        else:
            lines.append(f"HDF5 Root [{len(filepath_or_file.attrs)} attrs]")
            _walk(filepath_or_file)

        return "\n".join(lines)
