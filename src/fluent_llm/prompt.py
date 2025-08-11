from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Type
from pydantic import BaseModel

from .messages import MessageList, ResponseType


@dataclass(slots=True)
class Prompt:
    """Unified prompt description used across selection and provider layers.

    Attributes:
        messages: The list of input messages for the prompt.
        expect_type: Expected output type. Can be a ResponseType enum
                     or a Pydantic BaseModel subclass for structured output.
        preferred_provider: Optional provider hint (e.g. 'openai', 'anthropic').
        preferred_model: Optional model hint (e.g. 'gpt-4o-mini').
    """
    messages: MessageList
    expect_type: ResponseType | Type[BaseModel]
    preferred_provider: Optional[str] = None
    preferred_model: Optional[str] = None

    # -------- Convenience capabilities --------
    @property
    def audio_in(self) -> bool:
        return self.messages.has_audio

    @property
    def audio_out(self) -> bool:
        return self.expect_type == ResponseType.AUDIO

    @property
    def image_in(self) -> bool:
        return self.messages.has_image

    @property
    def image_out(self) -> bool:
        return self.expect_type == ResponseType.IMAGE

    @property
    def text_in(self) -> bool:
        return self.messages.has_text

    @property
    def text_out(self) -> bool:
        return self.expect_type == ResponseType.TEXT

    @property
    def structured_out(self) -> bool:
        return isinstance(self.expect_type, type) and issubclass(self.expect_type, BaseModel)

    # -------- Involvement helpers --------
    @property
    def audio_involved(self) -> bool:
        return self.audio_in or self.audio_out

    @property
    def image_involved(self) -> bool:
        return self.image_in or self.image_out

    @property
    def text_involved(self) -> bool:
        return self.text_in or self.text_out
