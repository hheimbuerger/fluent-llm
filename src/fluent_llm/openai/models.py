"""
Model selector and capability table for OpenAI and other LLMs.

Edit the MODEL_CAPABILITIES table to add or update model capabilities.
"""
from typing import Optional, List, Dict, Any, Tuple, NamedTuple
from decimal import Decimal
import re


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
        price_per_million_image_tokens_output=Decimal('NaN'),  # Not available
        price_per_million_audio_tokens_input=Decimal('NaN'),   # Not available
        price_per_million_audio_tokens_output=Decimal('NaN'),  # Not available
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
        price_per_million_text_tokens_output=Decimal('10.00'), # FIXME: actually unknown, not mentioned on https://platform.openai.com/docs/pricing#latest-models
        price_per_million_image_tokens_input=Decimal('10.00'),    # Not available
        price_per_million_image_tokens_output=Decimal('40.00'),   # Not available
        price_per_million_audio_tokens_input=Decimal('NaN'), # Not available
        price_per_million_audio_tokens_output=Decimal('NaN'),# Not available
        additional_pricing={},
    ),
    # Add more models as needed...
)


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
