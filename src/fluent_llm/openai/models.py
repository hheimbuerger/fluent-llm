"""
Model selector and capability table for OpenAI and other LLMs.

Edit the MODEL_CAPABILITIES table to add or update model capabilities.
"""
from enum import Enum
from typing import Dict, NamedTuple, Tuple, Optional
import re
from decimal import Decimal
from ..messages import ResponseType, Message, AudioMessage, ImageMessage, TextMessage, AgentMessage


def normalize_model_name(model_name: str) -> str:
    """Normalize a model name by stripping any trailing ISO 8601 date.

    Example:
        "gpt-4o-2024-07-18" -> "gpt-4o"
        "gpt-4o" -> "gpt-4o"
    """
    # Match a dash followed by 4 digits, dash, 2 digits, dash, 2 digits at end of string
    return re.sub(r'-\d{4}-\d{2}-\d{2}$', '', model_name)

class OpenAIModel(NamedTuple):
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

# Example: tuple of models for easy editing
OPENAI_MODELS: Tuple[OpenAIModel, ...] = (
    OpenAIModel(
        name="gpt-4o-mini",
        text_input=True,
        image_input=True,
        audio_input=False,
        text_output=True,
        image_output=False,
        audio_output=False,
        structured_output=True,
        price_per_million_text_tokens_input=Decimal("0.15"),
        price_per_million_text_tokens_output=Decimal("0.60"),
        price_per_million_image_tokens_input=Decimal("1"),
        price_per_million_image_tokens_output=None,  # Not available
        price_per_million_audio_tokens_input=None,   # Not available
        price_per_million_audio_tokens_output=None,  # Not available
        additional_pricing={},
    ),
    OpenAIModel(
        name="gpt-4.1-mini",
        text_input=True,
        image_input=True,
        audio_input=False,
        text_output=True,
        image_output=True,
        audio_output=False,
        structured_output=True,
        price_per_million_text_tokens_input=Decimal("0.15"),
        price_per_million_text_tokens_output=Decimal("0.60"),
        price_per_million_image_tokens_input=Decimal("1"),
        price_per_million_image_tokens_output=None,  # Not available
        price_per_million_audio_tokens_input=None,   # Not available
        price_per_million_audio_tokens_output=None,  # Not available
        additional_pricing={},
    ),
    OpenAIModel(
        name="gpt-4o-mini-audio",
        text_input=True,
        image_input=False,
        audio_input=True,
        text_output=True,
        image_output=False,
        audio_output=True,
        structured_output=False,
        price_per_million_text_tokens_input=Decimal("0.15"),  # FILL ME
        price_per_million_text_tokens_output=Decimal("0.60"), # FILL ME
        price_per_million_image_tokens_input=Decimal('NaN'),    # Not available
        price_per_million_image_tokens_output=Decimal('NaN'),   # Not available
        price_per_million_audio_tokens_input=Decimal("10.00"), # FILL ME
        price_per_million_audio_tokens_output=Decimal("20.00"),# FILL ME
        additional_pricing={},
    ),
    OpenAIModel(
        name="gpt-image-1",
        text_input=True,
        image_input=True,
        audio_input=False,
        text_output=False,
        image_output=True,
        audio_output=False,
        structured_output=False,
        price_per_million_text_tokens_input=Decimal("5.00"),  # FILL ME
        price_per_million_text_tokens_output=None, # Not available
        price_per_million_image_tokens_input=Decimal('10.00'),    # Not available
        price_per_million_image_tokens_output=Decimal('40.00'),   # Not available
        price_per_million_audio_tokens_input=None, # Not available
        price_per_million_audio_tokens_output=None,# Not available
        additional_pricing={},
    ),
    # Add more models as needed...
)


def select_model(messages, expect_type):
    """
    Legacy model selection logic based on messages and expect_type.
    Selection rules (in order of priority):
    1. If expecting image output: gpt-image-1
    2. If there's audio input/output and no image input: gpt-4o-mini-audio
    3. If there's any image input: gpt-4o-mini
    4. Default: gpt-4o-mini
    """
    def _has_audio_output(expect_type):
        return expect_type == ResponseType.AUDIO
    def _has_image_output(expect_type):
        return expect_type == ResponseType.IMAGE
    def _has_audio_input(messages):
        for msg in messages:
            if isinstance(msg, AudioMessage):
                return True
            if isinstance(msg, dict) and isinstance(msg.get("content"), list):
                if any(part.get("type") == "audio" for part in msg["content"]):
                    return True
        return False
    def _has_image_input(messages):
        for msg in messages:
            if isinstance(msg, ImageMessage):
                return True
            if isinstance(msg, dict) and isinstance(msg.get("content"), list):
                if any(part.get("type") == "image_url" for part in msg["content"]):
                    return True
        return False
    # Check for image output first (highest priority)
    if _has_image_output(expect_type):
        return "gpt-4.1-mini"
    # Check for audio input/output and no image input
    has_audio = _has_audio_output(expect_type) or _has_audio_input(messages)
    has_image = _has_image_input(messages)
    if has_audio and not has_image:
        return "gpt-4o-mini-audio"
    if has_image:
        return "gpt-4o-mini"
    return "gpt-4o-mini"  # default lightweight model

def get_model_by_name(model_name: str) -> Optional[OpenAIModel]:
    """Get model by name, handling versioned model names.

    Args:
        model_name: The model name to look up, which may include a version date suffix.

    Returns:
        The matching OpenAIModel or None if not found.
    """
    # First try exact match
    for model in OPENAI_MODELS:
        if model.name == model_name:
            return model

    # If no exact match, try normalizing the name by removing version date
    normalized_name = normalize_model_name(model_name)
    if normalized_name != model_name:
        for model in OPENAI_MODELS:
            if model.name == normalized_name:
                return model

    return None

def validate_model(model_name, messages, expect_type):
    """
    Validate that the selected model has all required capabilities for the current request.
    """
    model = get_model_by_name(model_name)
    if model is None:
        raise ValueError(f"Model {model_name} not found")

    def _has_text_input(messages):
        for msg in messages:
            if isinstance(msg, TextMessage):
                return True
            if isinstance(msg, dict) and isinstance(msg.get("content"), list):
                if any(part.get("type") == "text" for part in msg["content"]):
                    return True
        return False

    def _has_text_output(expect_type):
        return expect_type == ResponseType.TEXT

    def _has_audio_input(messages):
        for msg in messages:
            if isinstance(msg, dict) and isinstance(msg.get("content"), list):
                if any(part.get("type") == "audio" for part in msg["content"]):
                    return True
        return False

    def _has_audio_output(expect_type):
        return expect_type == ResponseType.AUDIO

    def _has_image_input(messages):
        for msg in messages:
            if isinstance(msg, dict) and isinstance(msg.get("content"), list):
                if any(part.get("type") == "image" for part in msg["content"]):
                    return True
        return False

    def _has_image_output(expect_type):
        return expect_type == ResponseType.IMAGE

    if _has_text_input(messages) and not model.text_input:
        raise ValueError(f"Model {model_name} does not support text input")
    if _has_text_output(expect_type) and not model.text_output:
        raise ValueError(f"Model {model_name} does not support text output")
    if _has_audio_input(messages) and not model.audio_input:
        raise ValueError(f"Model {model_name} does not support audio input")
    if _has_audio_output(expect_type) and not model.audio_output:
        raise ValueError(f"Model {model_name} does not support audio output")
    if _has_image_input(messages) and not model.image_input:
        raise ValueError(f"Model {model_name} does not support image input")
    if _has_image_output(expect_type) and not model.image_output:
        raise ValueError(f"Model {model_name} does not support image output")


def select_and_validate_model(messages, expect_type):
    model_name = select_model(messages, expect_type)
    validate_model(model_name, messages, expect_type)
    return model_name
