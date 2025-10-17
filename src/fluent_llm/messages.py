from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass
import pathlib
from typing import Any, Type
from collections.abc import Iterable
import base64

class Role(str, Enum):
    """Abstract chat message role."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class Message(ABC):
    """Base class for all message types."""
    @property
    @abstractmethod
    def role(self) -> Role:
        """Return the role of the message."""
    @property
    @abstractmethod
    def content(self) -> Any:
        """Return the content of the message."""

    def to_dict(self) -> dict[str, Any]:
        """Convert the message to a dictionary representation.

        Returns:
            A dictionary with 'role' and 'content' keys.
        """
        return {
            "role": self.role.value,
            "content": self.content
        }

@dataclass(slots=True)
class TextMessage(Message):
    """A text message in the prompt."""
    text: str
    role: Role = Role.USER

    @property
    def content(self) -> str:
        return self.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.text
        }

@dataclass(slots=True)
class AudioMessage(Message):
    """An audio message in the prompt."""
    audio_path: pathlib.Path
    role: Role = Role.USER

    @property
    def content(self) -> str:
        # In a real implementation, this would read and encode the audio file
        return str(self.audio_path)

    @property
    def content_b64(self) -> str:
        with open(self.audio_path, "rb") as audio_file:
            audio_data = audio_file.read()
        return base64.b64encode(audio_data).decode("utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": [
                {
                    "type": "audio",
                    "audio_url": f"file://{self.audio_path}"
                }
            ]
        }

@dataclass(slots=True)
class ImageMessage(Message):
    """An image message in the prompt.

    Either image_path or image_data must be provided, but not both.
    """
    image_path: pathlib.Path | None = None
    image_data: bytes | None = None
    role: Role = Role.USER

    def __post_init__(self):
        if self.image_path is None and self.image_data is None:
            raise ValueError("Either image_path or image_data must be provided")
        if self.image_path is not None and self.image_data is not None:
            raise ValueError("Only one of image_path or image_data can be provided")

    @property
    def content(self) -> str:
        """Return a string representation of the image source."""
        if self.image_path is not None:
            return str(self.image_path)
        return f"<{len(self.image_data or b'')} bytes of image data>"

    @property
    def media_type(self) -> str:
        """Return the media type of the image."""
        # TODO: pretty creative way to determine media type...
        return f"image/{self.image_path.suffix[1:].lower()}"

    @property
    def base64_data(self) -> str:
        """Return the image data as a base64-encoded data URL."""
        if self.image_path is not None:
            with open(self.image_path, "rb") as image_file:
                image_data = image_file.read()
        else:
            image_data = self.image_data or b''

        base64_encoded = base64.b64encode(image_data).decode("utf-8")
        return base64_encoded

    @property
    def base64_data_url(self) -> str:
        """Return the image data as a base64-encoded data URL."""
        if self.image_path is not None:
            with open(self.image_path, "rb") as image_file:
                image_data = image_file.read()
        else:
            image_data = self.image_data or b''

        base64_encoded = base64.b64encode(image_data).decode("utf-8")
        return f"data:{self.media_type};base64,{base64_encoded}"

    def to_dict(self) -> dict[str, Any]:
        """Convert the image message to a dictionary representation.

        Returns:
            A dictionary with 'role' and 'content' keys, where content is a list
            containing the image data as a data URL.
        """
        return {
            "role": self.role.value,
            "content": [
                {
                    "type": "image_url",
                    "image_url": self.base64_data_url if self.image_data is not None else f"file://{self.image_path}"
                }
            ]
        }

@dataclass(slots=True)
class AgentMessage(Message):
    """A message from an agent in the prompt (system role)."""
    text: str
    role: Role = Role.SYSTEM

    @property
    def content(self) -> str:
        return self.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.text
        }

@dataclass(slots=True)
class ToolCallMessage(Message):
    """A message representing a complete tool call with execution result."""
    message: str  # Any assistant message text accompanying the tool call
    tool_name: str
    tool_call_id: str
    arguments: dict
    result: Any | None = None  # Tool execution result (when successful)
    error: Exception | None = None  # Exception instance (when failed)
    role: Role = Role.ASSISTANT

    @property
    def content(self) -> dict:
        return {
            "message": self.message,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "arguments": self.arguments,
            "result": self.result,
            "error": str(self.error) if self.error else None
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content
        }

    def __str__(self) -> str:
        """String representation showing tool call and result."""
        if self.error:
            return f"Tool call: {self.tool_name}({self.arguments}) -> ERROR: {self.error}"
        return f"Tool call: {self.tool_name}({self.arguments}) -> {self.result}"


class ResponseType(Enum):
    """Expected response type."""
    TEXT = auto()
    AUDIO = auto()
    IMAGE = auto()
    JSON = auto()
