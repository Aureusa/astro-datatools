"""AstroIO unified facade for astronomical and scientific file formats."""
from typing import Any, Dict, List, Optional, Sequence, Union

from .base import BaseIOHandler, BaseReader, BaseWriter
from .fits import FitsIO
from .hdf5 import HDF5IO
from .image import ImageIO
from .numpy_io import NpyIO
from .registry import DEFAULT_REGISTRY, IORegistry


class _FormatAccessor:
    """Descriptor providing access to format-specific handlers on both class and instance."""

    def __init__(self, handler_cls):
        self._handler = handler_cls()

    def __get__(self, instance, owner):
        return self._handler


class _DualMethod:
    """Descriptor enabling a method to be called on either class or instance,

    correctly receiving either the class or the instance as the first argument.
    """

    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__

    def __get__(self, instance, owner):
        target = instance if instance is not None else owner

        def wrapper(*args, **kwargs):
            return self.func(target, *args, **kwargs)

        wrapper.__doc__ = self.func.__doc__
        return wrapper


class AstroIO:
    """Unified facade for reading and writing astronomical and standard scientific file formats.

    Usage:
        # Automatic dispatch by file extension:
        data = AstroIO.read('image.fits')
        AstroIO.write('output.fits', data)

        # Direct access to format-specific toolkits:
        img, header = AstroIO.fits.read_image('galaxy.fits', return_header=True)
        AstroIO.hdf5.write_dict('dataset.h5', my_dict)
    """

    registry: IORegistry = DEFAULT_REGISTRY

    # Format-specific toolkits accessible as class attributes or instance attributes
    fits: FitsIO = _FormatAccessor(FitsIO)
    hdf5: HDF5IO = _FormatAccessor(HDF5IO)
    numpy: NpyIO = _FormatAccessor(NpyIO)
    image: ImageIO = _FormatAccessor(ImageIO)

    def __init__(self, registry: Optional[IORegistry] = None):
        if registry is not None:
            self.registry = registry

    @_DualMethod
    def read(self_or_cls, filepath: str, **kwargs: Any) -> Any:
        """Read data from a file using the automatically resolved reader.

        :param filepath: Path to the file.
        :param kwargs: Additional format-specific keyword arguments.
        :return: Data read from the file.
        """
        registry = getattr(self_or_cls, "registry", DEFAULT_REGISTRY)
        reader = registry.resolve_reader(filepath)
        return reader.read(filepath, **kwargs)

    @_DualMethod
    def write(self_or_cls, filepath: str, data: Any, **kwargs: Any) -> None:
        """Write data to a file using the automatically resolved writer.

        :param filepath: Path to output file.
        :param data: Data to write.
        :param kwargs: Additional format-specific keyword arguments.
        """
        registry = getattr(self_or_cls, "registry", DEFAULT_REGISTRY)
        writer = registry.resolve_writer(filepath)
        writer.write(filepath, data, **kwargs)

    @_DualMethod
    def register_reader(self_or_cls, extensions: Union[str, Sequence[str]], reader: BaseReader) -> None:
        """Register a custom reader for one or more file extensions."""
        registry = getattr(self_or_cls, "registry", DEFAULT_REGISTRY)
        registry.register_reader(extensions, reader)

    @_DualMethod
    def register_writer(self_or_cls, extensions: Union[str, Sequence[str]], writer: BaseWriter) -> None:
        """Register a custom writer for one or more file extensions."""
        registry = getattr(self_or_cls, "registry", DEFAULT_REGISTRY)
        registry.register_writer(extensions, writer)

    @_DualMethod
    def register_handler(self_or_cls, extensions: Union[str, Sequence[str]], handler: BaseIOHandler) -> None:
        """Register a format handler (reader + writer + tools) for extensions."""
        registry = getattr(self_or_cls, "registry", DEFAULT_REGISTRY)
        registry.register_handler(extensions, handler)

    @_DualMethod
    def supported_formats(self_or_cls) -> Dict[str, List[str]]:
        """Return a summary of supported read and write file extensions."""
        registry = getattr(self_or_cls, "registry", DEFAULT_REGISTRY)
        return {
            "read": registry.supported_read_formats,
            "write": registry.supported_write_formats,
        }


