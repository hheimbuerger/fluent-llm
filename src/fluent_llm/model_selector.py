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
from .providers.anthropic.claude import AnthropicProvider


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
        expect_type: Optional[str] = None,
        preferred_provider: Optional[str] = None,
        preferred_model: Optional[str] = None
    ) -> Tuple[Callable, str]:
        """
        Select an appropriate model based on the input messages and expected output type.

        Args:
            messages: List of message dictionaries containing the conversation history.
            expect_type: The expected output type (e.g., 'text', 'image', 'audio').
            preferred_provider: Preferred provider name (e.g., 'openai', 'anthropic').
            preferred_model: Preferred model name (e.g., 'gpt-4o-mini', 'claude-3-sonnet').

        Returns:
            The name of the selected model.
        """
        raise NotImplementedError("Subclasses must implement select_model")


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
        expect_type: Optional[ResponseType] = None,
        preferred_provider: Optional[str] = None,
        preferred_model: Optional[str] = None
    ) -> Tuple[Callable, str]:
        """
        Select the appropriate model based on message content and expected response type.

        Selection rules (in order of priority):
        1. If preferred_model is specified, validate it supports required capabilities
        2. If preferred_provider is specified, select best model from that provider
        3. Otherwise, use default selection logic:
           - If expecting image output: gpt-4.1-mini
           - If there's audio input/output and no image input: gpt-4o-mini-audio
           - If there's any image input: gpt-4o-mini
           - Default: gpt-4o-mini

        Args:
            messages: MessageList containing the conversation history
            expect_type: The expected response type
            preferred_provider: Preferred provider name (e.g., 'openai', 'anthropic')
            preferred_model: Preferred model name (e.g., 'gpt-4o-mini', 'claude-3-sonnet')

        Returns:
            A tuple of (provider_instance, model_name)

        Raises:
            UnresolvableModelError: If the preferred provider/model cannot fulfill the request
        """
        # Get available providers
        providers = {
            'openai': OpenAIProvider(),
            'anthropic': AnthropicProvider()
        }

        # Determine required capabilities
        has_audio = (expect_type == ResponseType.AUDIO) or messages.has_audio
        has_image = messages.has_image
        needs_image_output = expect_type == ResponseType.IMAGE
        needs_audio_output = expect_type == ResponseType.AUDIO
        needs_text_output = expect_type == ResponseType.TEXT or expect_type is None

        # If specific model is requested, validate it
        if preferred_model:
            # Find which provider has this model
            selected_provider = None
            selected_model = None

            for provider_name, provider_instance in providers.items():
                for model in provider_instance.get_models():
                    if model.name == preferred_model:
                        selected_provider = provider_instance
                        selected_model = model
                        break
                if selected_provider:
                    break

            if not selected_provider:
                raise UnresolvableModelError(f"Model '{preferred_model}' not found in any provider")

            # Validate capabilities
            if has_image and not selected_model.image_input:
                raise UnresolvableModelError(f"Model '{preferred_model}' does not support image input")
            if has_audio and not selected_model.audio_input:
                raise UnresolvableModelError(f"Model '{preferred_model}' does not support audio input")
            if needs_image_output and not selected_model.image_output:
                raise UnresolvableModelError(f"Model '{preferred_model}' does not support image output")
            if needs_audio_output and not selected_model.audio_output:
                raise UnresolvableModelError(f"Model '{preferred_model}' does not support audio output")
            if needs_text_output and not selected_model.text_output:
                raise UnresolvableModelError(f"Model '{preferred_model}' does not support text output")

            return selected_provider, preferred_model

        # If specific provider is requested, find best model from that provider
        if preferred_provider:
            if preferred_provider not in providers:
                raise UnresolvableModelError(f"Provider '{preferred_provider}' not supported. Available: {list(providers.keys())}")

            provider_instance = providers[preferred_provider]
            available_models = provider_instance.get_models()

            # Find a suitable model from the preferred provider
            suitable_models = []
            for model in available_models:
                if (not has_image or model.image_input) and \
                   (not has_audio or model.audio_input) and \
                   (not needs_image_output or model.image_output) and \
                   (not needs_audio_output or model.audio_output) and \
                   (not needs_text_output or model.text_output):
                    suitable_models.append(model)

            if not suitable_models:
                capabilities = []
                if has_image: capabilities.append("image input")
                if has_audio: capabilities.append("audio input")
                if needs_image_output: capabilities.append("image output")
                if needs_audio_output: capabilities.append("audio output")
                if needs_text_output: capabilities.append("text output")

                raise UnresolvableModelError(
                    f"Provider '{preferred_provider}' has no models supporting required capabilities: {', '.join(capabilities)}"
                )

            # Select the first suitable model (could be enhanced with better selection logic)
            selected_model = suitable_models[0]
            return provider_instance, selected_model.name

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
