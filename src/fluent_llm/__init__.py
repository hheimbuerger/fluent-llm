"""Top-level package for fluent-llm."""

from importlib import metadata

from .builder import llm, ResponseType
from .usage_tracker import UsageTracker, track_usage, get_usage, reset_usage

__all__ = [
    "llm",
    "ResponseType",
    "UsageTracker",
    "track_usage",
    "get_usage",
    "reset_usage",
]

try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "0.0.0"
