from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass
import pathlib
from typing import Any, Type, TypeVar, Generic, List as ListType, overload
from collections.abc import Iterable, Iterator
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
        return f"image/{self.image_path.suffix.lower()}"

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
        return f"data:image/png;base64,{base64_encoded}"

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

class MessageList(list[Message]):
    """A list of Message objects with additional type-checking methods.

    This class provides convenience methods to check for the presence of
    different message types in the list.
    """

    def __init__(self, iterable: Iterable[Message] = ()):
        super().__init__(iterable)

    def has_type(self, msg_type: Type[Message]) -> bool:
        """Check if the list contains any messages of the specified type."""
        return any(isinstance(msg, msg_type) for msg in self)

    @property
    def has_text(self) -> bool:
        """Check if the list contains any TextMessage instances."""
        return self.has_type(TextMessage)

    @property
    def has_audio(self) -> bool:
        """Check if the list contains any AudioMessage instances."""
        return self.has_type(AudioMessage)

    @property
    def has_image(self) -> bool:
        """Check if the list contains any ImageMessage instances."""
        return self.has_type(ImageMessage)

    @property
    def has_agent(self) -> bool:
        """Check if the list contains any AgentMessage instances."""
        return self.has_type(AgentMessage)

    def to_dict_list(self) -> list[dict]:
        """Convert all messages in the list to their dictionary representation."""
        return [msg.to_dict() for msg in self]

    def merge_all_text(self) -> str:
        """Merge all text messages into a single string."""
        return "\n".join(msg.text for msg in self if isinstance(msg, (AgentMessage, TextMessage)))

    def merge_all_agent(self) -> str:
        """Merge all agent messages into a single string."""
        return "\n".join(msg.text for msg in self if isinstance(msg, AgentMessage))

    def copy(self) -> 'MessageList':
        """Return a shallow copy of the MessageList.

        Returns:
            A new MessageList instance containing the same Message objects.
        """
        return MessageList(self)


class ResponseType(Enum):
    """Expected response type."""
    TEXT = auto()
    AUDIO = auto()
    IMAGE = auto()
    JSON = auto()
