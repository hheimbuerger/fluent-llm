"""
Model selection strategies for LLM prompt building.

This module provides an interface and implementations for model selection strategies
used in the LLM prompt building process.
"""
from abc import ABC, abstractmethod
from typing import Callable, Tuple
from .prompt import Prompt
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
        p: Prompt,
    ) -> Tuple[Callable, str]:
        """
        Select an appropriate model based on the input messages and expected output type.

        Args:
            p: Prompt object containing messages, expected type, and optional provider/model hints.

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
        p: Prompt,
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
            p: Prompt containing the conversation history, expected response type,
               and optional provider/model preferences

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

        # Determine required capabilities using Prompt helpers (use directly below)

        # If specific model is requested, validate it
        if p.preferred_model:
            # Find which provider has this model
            selected_provider = None
            selected_model = None

            for provider_name, provider_instance in providers.items():
                for model in provider_instance.get_models():
                    if model.name == p.preferred_model:
                        selected_provider = provider_instance
                        selected_model = model
                        break
                if selected_provider:
                    break

            if not selected_provider:
                raise UnresolvableModelError(f"Model '{p.preferred_model}' not found in any provider")

            # Validate capabilities
            if p.image_involved and not selected_model.image_input:
                raise UnresolvableModelError(f"Model '{p.preferred_model}' does not support image input")
            if p.audio_involved and not selected_model.audio_input:
                raise UnresolvableModelError(f"Model '{p.preferred_model}' does not support audio input")
            if p.image_out and not selected_model.image_output:
                raise UnresolvableModelError(f"Model '{p.preferred_model}' does not support image output")
            if p.audio_out and not selected_model.audio_output:
                raise UnresolvableModelError(f"Model '{p.preferred_model}' does not support audio output")
            if p.text_out and not selected_model.text_output:
                raise UnresolvableModelError(f"Model '{p.preferred_model}' does not support text output")

            return selected_provider, p.preferred_model

        # If specific provider is requested, find best model from that provider
        if p.preferred_provider:
            if p.preferred_provider not in providers:
                raise UnresolvableModelError(f"Provider '{p.preferred_provider}' not supported. Available: {list(providers.keys())}")

            provider_instance = providers[p.preferred_provider]
            available_models = provider_instance.get_models()

            # Find a suitable model from the preferred provider
            suitable_models = []
            for model in available_models:
                if (not p.image_involved or model.image_input) and \
                   (not p.audio_involved or model.audio_input) and \
                   (not p.image_out or model.image_output) and \
                   (not p.audio_out or model.audio_output) and \
                   (not (p.text_out or p.expect_type is None) or model.text_output):
                    suitable_models.append(model)

            if not suitable_models:
                capabilities = []
                if p.image_involved: capabilities.append("image input")
                if p.audio_involved: capabilities.append("audio input")
                if p.image_out: capabilities.append("image output")
                if p.audio_out: capabilities.append("audio output")
                if p.text_out: capabilities.append("text output")

                raise UnresolvableModelError(
                    f"Provider '{p.preferred_provider}' has no models supporting required capabilities: {', '.join(capabilities)}"
                )

            # Select the first suitable model (could be enhanced with better selection logic)
            selected_model = suitable_models[0]
            return provider_instance, selected_model.name

        # === from here on actual autoselection algorithm, given no hints ===

        # Prefer Anthropic when tools are present or when using conversation mode
        if p.has_tools or p.is_conversation:
            # Use Anthropic for tool calling and conversations
            provider = AnthropicProvider()
            
            # Select appropriate Anthropic model based on capabilities
            if p.image_involved:
                model = "claude-3-5-sonnet-20241022"  # Supports vision
            else:
                model = "claude-3-5-haiku-20241022"  # Faster for text-only
            
            return provider, model

        # Check for image output first
        if p.image_out:
            assert not p.audio_involved, "Audio input is not supported on image generating models."
            model = "gpt-image-1"

        # Check for audio input or output
        elif p.audio_out or p.audio_involved:
            assert not p.image_involved and not p.image_out, "Image input/output is not supported on audio generating/processing models."
            assert not p.structured_out, "Structured output is not supported on audio generating/processing models."
            model = "gpt-4o-mini-audio-preview"

        else:
            model = "gpt-4o-mini"

        provider = OpenAIProvider()
        return provider, model
