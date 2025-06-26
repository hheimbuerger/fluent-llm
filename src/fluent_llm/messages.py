from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass
import pathlib
from typing import Any
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

@dataclass(slots=True)
class TextMessage(Message):
    """A text message in the prompt."""
    text: str
    role: Role = Role.USER
    @property
    def content(self) -> str:
        return self.text

@dataclass(slots=True)
class AudioMessage(Message):
    """An audio message in the prompt."""
    audio_path: pathlib.Path
    role: Role = Role.USER
    @property
    def content(self) -> str:
        # In a real implementation, this would read and encode the audio file
        return str(self.audio_path)

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
    def base64_data_url(self) -> str:
        """Return the image data as a base64-encoded data URL."""
        if self.image_path is not None:
            with open(self.image_path, "rb") as image_file:
                image_data = image_file.read()
        else:
            image_data = self.image_data or b''
            
        base64_encoded = base64.b64encode(image_data).decode("utf-8")
        return f"data:image/png;base64,{base64_encoded}"

@dataclass(slots=True)
class AgentMessage(Message):
    """A message from an agent in the prompt (system role)."""
    text: str
    role: Role = Role.SYSTEM
    @property
    def content(self) -> str:
        return self.text

class ResponseType(Enum):
    """Expected response type."""
    TEXT = auto()
    AUDIO = auto()
    IMAGE = auto()
    JSON = auto()
