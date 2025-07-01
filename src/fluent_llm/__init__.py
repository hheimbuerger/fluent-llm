"""Top-level package for fluent-llm."""

from importlib import metadata

from .builder import llm

__all__ = [
    "llm",
]

try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "0.0.0"
