"""
Model selection strategies for LLM prompt building.

This module provides an interface and implementations for model selection strategies
used in the LLM prompt building process.
"""
from abc import ABC, abstractmethod
from typing import Optional, Callable, Tuple

from .messages import (
    MessageList, ResponseType
)
from .providers.openai.gpt import OpenAIProvider


class UnresolvableModelError(Exception):
    """Raised when a model selection strategy cannot select a model."""


class ModelSelectionStrategy(ABC):
    """
    Abstract base class for model selection strategies.

    Implementations of this class define how to select an appropriate model
    based on the input messages and expected output type.
    """

    @abstractmethod
    def select_model(
        self,
        messages: MessageList,
        expect_type: Optional[str] = None
    ) -> Tuple[Callable, str]:
        """
        Select an appropriate model based on the input messages and expected output type.

        Args:
            messages: List of message dictionaries containing the conversation history.
            expect_type: The expected output type (e.g., 'text', 'image', 'audio').

        Returns:
            The name of the selected model.
        """
        pass


class DefaultModelSelectionStrategy(ModelSelectionStrategy):
    """
    Default model selection strategy based on the legacy implementation.

    Selection rules (in order of priority):
    1. If expecting image output: gpt-image-1
    2. If there's audio input/output and no image input: gpt-4o-mini-audio
    3. If there's any image input: gpt-4o-mini
    4. Default: gpt-4o-mini
    """
    def select_model(
        self,
        messages: MessageList,
        expect_type: Optional[ResponseType] = None
    ) -> Tuple[Callable, str]:
        """
        Select the appropriate model based on message content and expected response type.

        Selection rules (in order of priority):
        1. If expecting image output: gpt-4.1-mini
        2. If there's audio input/output and no image input: gpt-4o-mini-audio
        3. If there's any image input: gpt-4o-mini
        4. Default: gpt-4o-mini

        Args:
            messages: MessageList containing the conversation history
            expect_type: The expected response type

        Returns:
            A tuple of (call_llm_api_function, model_name)
        """
        # Check for image output first (highest priority)
        if expect_type == ResponseType.IMAGE:
            model = "gpt-image-1"

        else:
            # Check for audio input/output and no image input
            has_audio = (expect_type == ResponseType.AUDIO) or messages.has_audio
            has_image = messages.has_image

            model = "gpt-4o-mini"
            if has_audio:
                if has_image:
                    raise UnresolvableModelError("Audio and image input are not supported by the same model.")
                model = "gpt-4o-mini-audio"
            elif has_image:
                model = "gpt-4o-mini"

        provider = OpenAIProvider()

        return provider, model
