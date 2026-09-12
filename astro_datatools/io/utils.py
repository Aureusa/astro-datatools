"""Utility functions for AstroIO."""
from .registry import DEFAULT_REGISTRY


def _resolve_reader(filepath: str):
    """Resolve reader from registry for given filepath."""
    return DEFAULT_REGISTRY.resolve_reader(filepath)


def _resolve_writer(filepath: str):
    """Resolve writer from registry for given filepath."""
    return DEFAULT_REGISTRY.resolve_writer(filepath)

