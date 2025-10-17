"""Top-level package for fluent-llm."""

from importlib import metadata

from .builder import llm
from .conversation import MessageList, LLMConversation, MessageListDeserializationError, DeltaApplicationError, ConversationConfigurationError

__all__ = [
    "llm",
    "MessageList",
    "LLMConversation",
    "MessageListDeserializationError",
    "DeltaApplicationError",
    "ConversationConfigurationError",
]

try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "0.0.0"
