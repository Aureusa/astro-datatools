"""Registry managing format readers, writers, and format handlers."""
from typing import Dict, List, Optional, Sequence, Union

from .base import BaseIOHandler, BaseReader, BaseWriter


class IORegistry:
    """Registry managing format readers, writers, and format handlers by file extension."""

    def __init__(self):
        self._readers: Dict[str, BaseReader] = {}
        self._writers: Dict[str, BaseWriter] = {}
        self._handlers: Dict[str, BaseIOHandler] = {}

    def _normalize_ext(self, ext: str) -> str:
        ext = ext.lower().strip()
        if not ext.startswith("."):
            ext = "." + ext
        return ext

    def register_reader(self, extensions: Union[str, Sequence[str]], reader: BaseReader) -> None:
        """Register a reader for one or more file extensions."""
        if isinstance(extensions, str):
            extensions = [extensions]
        for ext in extensions:
            self._readers[self._normalize_ext(ext)] = reader

    def register_writer(self, extensions: Union[str, Sequence[str]], writer: BaseWriter) -> None:
        """Register a writer for one or more file extensions."""
        if isinstance(extensions, str):
            extensions = [extensions]
        for ext in extensions:
            self._writers[self._normalize_ext(ext)] = writer

    def register_handler(
        self,
        extensions: Union[str, Sequence[str]],
        handler: BaseIOHandler,
    ) -> None:
        """Register a format handler (and its reader/writer if available) for extensions."""
        if isinstance(extensions, str):
            extensions = [extensions]
        for ext in extensions:
            norm_ext = self._normalize_ext(ext)
            self._handlers[norm_ext] = handler
            if getattr(handler, "reader", None) is not None:
                self._readers[norm_ext] = handler.reader
            if getattr(handler, "writer", None) is not None:
                self._writers[norm_ext] = handler.writer

    def _find_matching_extension(self, filepath: str, mapping: dict) -> Optional[str]:
        # Match longest extensions first (e.g., '.fits.gz' before '.gz')
        filepath_lower = filepath.lower()
        sorted_exts = sorted(mapping.keys(), key=len, reverse=True)
        for ext in sorted_exts:
            if filepath_lower.endswith(ext):
                return ext
        return None

    def resolve_reader(self, filepath: str) -> BaseReader:
        """Find the appropriate reader for a given filepath."""
        matched_ext = self._find_matching_extension(filepath, self._readers)
        if matched_ext is None:
            raise ValueError(
                f"Unsupported reader file format for '{filepath}'. "
                f"Supported formats: {self.supported_read_formats}"
            )
        return self._readers[matched_ext]

    def resolve_writer(self, filepath: str) -> BaseWriter:
        """Find the appropriate writer for a given filepath."""
        matched_ext = self._find_matching_extension(filepath, self._writers)
        if matched_ext is None:
            raise ValueError(
                f"Unsupported writer file format for '{filepath}'. "
                f"Supported formats: {self.supported_write_formats}"
            )
        return self._writers[matched_ext]

    def resolve_handler(self, filepath: str) -> Optional[BaseIOHandler]:
        """Find the format handler for a given filepath, if registered."""
        matched_ext = self._find_matching_extension(filepath, self._handlers)
        return self._handlers.get(matched_ext) if matched_ext else None

    @property
    def supported_read_formats(self) -> List[str]:
        """List all supported read extensions."""
        return sorted(list(self._readers.keys()))

    @property
    def supported_write_formats(self) -> List[str]:
        """List all supported write extensions."""
        return sorted(list(self._writers.keys()))


DEFAULT_REGISTRY = IORegistry()


def register_reader(extensions: Union[str, Sequence[str]]):
    """Decorator to register a reader class or instance with the default registry."""
    def decorator(cls_or_instance):
        instance = cls_or_instance() if isinstance(cls_or_instance, type) else cls_or_instance
        DEFAULT_REGISTRY.register_reader(extensions, instance)
        return cls_or_instance
    return decorator


def register_writer(extensions: Union[str, Sequence[str]]):
    """Decorator to register a writer class or instance with the default registry."""
    def decorator(cls_or_instance):
        instance = cls_or_instance() if isinstance(cls_or_instance, type) else cls_or_instance
        DEFAULT_REGISTRY.register_writer(extensions, instance)
        return cls_or_instance
    return decorator


def register_handler(extensions: Union[str, Sequence[str]]):
    """Decorator to register a handler class or instance with the default registry."""
    def decorator(cls_or_instance):
        instance = cls_or_instance() if isinstance(cls_or_instance, type) else cls_or_instance
        DEFAULT_REGISTRY.register_handler(extensions, instance)
        return cls_or_instance
    return decorator
