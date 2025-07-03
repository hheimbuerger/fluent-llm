from abc import ABC, abstractmethod
from typing import Optional, NamedTuple, Tuple, Any, Type
import re
from decimal import Decimal
from pydantic import BaseModel
from ..messages import MessageList, ResponseType, TextMessage, ImageMessage, AudioMessage, AgentMessage


class LLMModel(NamedTuple):
    name: str
    text_input: bool
    image_input: bool
    audio_input: bool
    text_output: bool
    image_output: bool
    audio_output: bool
    structured_output: bool
    price_per_million_text_tokens_input: Decimal
    price_per_million_text_tokens_output: Decimal
    price_per_million_image_tokens_input: Decimal
    price_per_million_image_tokens_output: Decimal
    price_per_million_audio_tokens_input: Decimal
    price_per_million_audio_tokens_output: Decimal
    additional_pricing: dict


class LLMProvider(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_models(self) -> Tuple[LLMModel]:
        """Get all available models."""

    @abstractmethod
    async def prompt_via_api(
        self,
        model: str,
        messages: MessageList,
        expect_type: ResponseType | Type[BaseModel],
        **kwargs: Any
    ) -> Any:
        """Make an async call to the provider's API with the given messages
        and return the appropriate response.
        The model is always inferred from the input and expected output."""

    def check_capabilities(
            self,
            model_name: str,
            messages: MessageList,
            expect_type: ResponseType | Type[BaseModel]
        ) -> None:
        """
        Validate that the selected model has all required capabilities for the current request.

        Args:
            model_name: Name of the model to validate
            messages: MessageList containing the conversation history
            expect_type: Expected response type

        Raises:
            ValueError: If the model doesn't support required capabilities
        """
        model = self.get_model_by_name(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")

        # Validate model capabilities
        if messages.has_text and not model.text_input:
            raise ValueError(f"Model {model_name} does not support text input")

        if expect_type == ResponseType.TEXT and not model.text_output:
            raise ValueError(f"Model {model_name} does not support text output")

        if messages.has_audio and not model.audio_input:
            raise ValueError(f"Model {model_name} does not support audio input")

        if expect_type == ResponseType.AUDIO and not model.audio_output:
            raise ValueError(f"Model {model_name} does not support audio output")

        if messages.has_image and not model.image_input:
            raise ValueError(f"Model {model_name} does not support image input")

        if expect_type == ResponseType.IMAGE and not model.image_output:
            raise ValueError(f"Model {model_name} does not support image output")

    def normalize_model_name(self, model_name: str) -> str:
        """Normalize a model name by stripping any trailing ISO 8601 date.

        Example:
            "gpt-4o-2024-07-18" -> "gpt-4o"
            "gpt-4o" -> "gpt-4o"
        """
        # Match a dash followed by 4 digits, dash, 2 digits, dash, 2 digits at end of string
        return re.sub(r'-\d{4}-\d{2}-\d{2}$', '', model_name)

    def get_model_by_name(self, model_name: str) -> Optional[LLMModel]:
        """Get model by name, handling versioned model names.

        Args:
            model_name: The model name to look up, which may include a version date suffix.

        Returns:
            The matching OpenAIModel or None if not found.
        """
        # Normalize the name by removing version date
        normalized_name = self.normalize_model_name(model_name)
        for model in self.get_models():
            if model.name == normalized_name:
                return model

        return None
