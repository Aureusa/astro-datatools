"""HDF5 I/O handler, reader, writer, and astronomy dataset utilities."""
import os
from typing import Any, Dict, List, Optional, Union
import itertools
import h5py
import numpy as np

# Disable HDF5 file locking by default on cluster / networked filesystems (ZFS/NFS/Lustre)
# if not explicitly configured in environment.
if "HDF5_USE_FILE_LOCKING" not in os.environ:
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

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

    def read_dict(self, filepath: str, group_path: str = "/", load_attrs: bool = True, **kwargs: Any) -> Any:
        """Read the entire HDF5 file or group/dataset into a python dictionary of numpy arrays and metadata.

        :param filepath: Path to HDF5 file.
        :param group_path: Group or dataset path to start reading from.
        :param load_attrs: Whether to load metadata attributes into '__attrs__'.
        :return: Nested dictionary or numpy array.
        """
        with h5py.File(filepath, "r", **kwargs) as f:
            target = f[group_path] if group_path != "/" else f
            if isinstance(target, h5py.Dataset):
                val = target[()]
                if load_attrs and len(target.attrs) > 0:
                    return {"data": val, "__attrs__": {ak: av for ak, av in target.attrs.items()}}
                return val
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
                try:
                    return list(target.keys())
                except Exception:
                    # Fallback to key iterator if symbol table len/keys() fails
                    return list(iter(target))
        target = filepath_or_file[path] if path != "/" else filepath_or_file
        try:
            return list(target.keys())
        except Exception:
            return list(iter(target))


    def tree(
        self,
        filepath_or_file: Union[str, h5py.File, h5py.Group],
        max_depth: Optional[int] = None,
        max_items_per_group: Optional[int] = 10,
    ) -> str:
        """Generate an ASCII tree view of the HDF5 hierarchy with shapes and attribute info.

        :param filepath_or_file: Path to HDF5 file or open h5py File/Group.
        :param max_depth: Maximum recursion depth (None for unlimited).
        :param max_items_per_group: Max child items to display per group (None for unlimited).
                                    Prevents hanging on files with thousands of groups/datasets.
        :return: Formatted ASCII tree string.
        """
        lines = []

        def _safe_get_attrs_count(obj) -> int:
            try:
                return len(obj.attrs) if hasattr(obj, "attrs") else 0
            except Exception:
                return 0

        def _walk(item: Union[h5py.File, h5py.Group, h5py.Dataset], prefix: str = "", current_depth: int = 0):
            if isinstance(item, (h5py.File, h5py.Group)):
                if max_depth is not None and current_depth >= max_depth:
                    return

                # Safely collect keys via iterator to handle HDF5 symbol table quirks
                keys = []
                try:
                    for idx, key in enumerate(item):
                        if max_items_per_group is not None and idx >= max_items_per_group:
                            break
                        keys.append(key)
                except Exception:
                    pass

                try:
                    total = len(item)
                except Exception:
                    total = len(keys)

                truncated = max(0, total - len(keys)) if total is not None else 0

                for i, k in enumerate(keys):
                    is_last = (i == len(keys) - 1) and (truncated == 0)
                    connector = "└── " if is_last else "├── "
                    try:
                        child = item[k]
                        attr_count = _safe_get_attrs_count(child)
                        attr_info = f" [{attr_count} attrs]" if attr_count > 0 else ""
                        if isinstance(child, h5py.Dataset):
                            lines.append(f"{prefix}{connector}{k}: Dataset shape={child.shape}, dtype={child.dtype}{attr_info}")
                        else:
                            try:
                                child_len = len(child)
                            except Exception:
                                child_len = "?"
                            lines.append(f"{prefix}{connector}{k}/ (Group, {child_len} items){attr_info}")
                            new_prefix = prefix + ("    " if is_last else "│   ")
                            _walk(child, new_prefix, current_depth + 1)
                    except Exception as err:
                        lines.append(f"{prefix}{connector}{k}: <Error reading item: {err}>")

                if truncated > 0:
                    lines.append(f"{prefix}└── ... ({truncated} more items)")

        if isinstance(filepath_or_file, str):
            with h5py.File(filepath_or_file, "r") as f:
                try:
                    root_len = len(f)
                except Exception:
                    root_len = "?"
                root_attrs = _safe_get_attrs_count(f)
                lines.append(f"{filepath_or_file} (HDF5 Root, {root_len} items) [{root_attrs} attrs]")
                _walk(f)
        else:
            try:
                root_len = len(filepath_or_file)
            except Exception:
                root_len = "?"
            root_attrs = _safe_get_attrs_count(filepath_or_file)
            lines.append(f"HDF5 Root, {root_len} items [{root_attrs} attrs]")
            _walk(filepath_or_file)

        return "\n".join(lines)
    