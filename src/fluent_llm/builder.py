"""Fluent, async builder for Large Language Model (LLM) requests.

This builder lets you fluently construct an *abstract* prompt (vendor-agnostic)
and only converts it to OpenAI-specific message JSON inside ``call()``.

Example
-------
```python
from python_openai_sample import llm, ResponseType

response = await (
    llm()
        .agent("You are an art evaluator.")
        .context("Please review …")
        .file("painting.png")
        .expect(ResponseType.TEXT)
        .call()
)
print(response)
```
"""
from __future__ import annotations

import pathlib
from typing import Any, List, Sequence, Type
import inspect

from pydantic import BaseModel
from decimal import Decimal
from .usage_tracker import tracker
from .openai.models import OPENAI_MODELS, get_model_by_name
from .openai.invoker import call_llm_api
from .messages import Message, TextMessage, AudioMessage, ImageMessage, AgentMessage, ResponseType

__all__: Sequence[str] = [
    "llm",
]


class LLMPromptBuilder:
    """Fluent, async builder for LLM requests."""

    def __init__(self, *, client: Any | None = None, messages: List[Message] | None = None, expect: ResponseType | None = None) -> None:
        self._client: Any = client
        self._messages: List[Message] = messages or []
        self._expect: ResponseType = expect or ResponseType.TEXT

    def _copy(self) -> "LLMPromptBuilder":
        """Create a copy of this builder with the same state."""
        return self.__class__(
            client=self._client,
            messages=self._messages.copy(),
            expect=self._expect
        )

    # ------------------------------------------------------------------
    # Chainable mutators
    # ------------------------------------------------------------------
    def agent(self, prompt: str) -> LLMPromptBuilder:
        """Add a *system* role message describing the agent/persona."""
        new_instance = self._copy()
        new_instance._messages.append(AgentMessage(text=prompt))
        return new_instance

    def context(self, content: str) -> LLMPromptBuilder:
        """Add background context (user role)."""
        new_instance = self._copy()
        new_instance._messages.append(TextMessage(text=content))
        return new_instance

    def request(self, content: str) -> LLMPromptBuilder:
        """Add the primary user request message."""
        new_instance = self._copy()
        new_instance._messages.append(TextMessage(text=content))
        return new_instance

    def audio(self, path: str | pathlib.Path) -> LLMPromptBuilder:
        """Add an audio file to the request."""
        new_instance = self._copy()
        new_instance._messages.append(AudioMessage(audio_path=pathlib.Path(path)))
        return new_instance

    def image(self, source: str | pathlib.Path | bytes) -> LLMPromptBuilder:
        """Add an image to the request.

        Args:
            source: Either a file path (str or pathlib.Path) or bytes containing the image data.
        """
        new_instance = self._copy()
        if isinstance(source, bytes):
            new_instance._messages.append(ImageMessage(image_data=source))
        else:
            new_instance._messages.append(ImageMessage(image_path=pathlib.Path(source)))
        return new_instance

    def file(self, path: str | pathlib.Path) -> LLMPromptBuilder:
        """Attach an image file to the request (basic vision support)."""
        return self.image(path)

    def expect(self, response_type: ResponseType | Type[BaseModel]) -> "LLMPromptBuilder":
        """
        Specify the expected response type.
        Accepts either a ResponseType enum or a Pydantic BaseModel subclass for structured outputs.
        """
        new_instance = self._copy()
        # Accept ResponseType or BaseModel subclass
        if inspect.isclass(response_type) and issubclass(response_type, BaseModel):
            new_instance._expect = response_type
        elif isinstance(response_type, ResponseType):
            new_instance._expect = response_type
        else:
            raise TypeError("expect() argument must be a ResponseType or a Pydantic BaseModel subclass.")
        return new_instance

    # ------------------------------------------------------------------
    # Pricing / stats helpers
    # ------------------------------------------------------------------
    def _gather_token_counts(self, data: Any, path: str = '') -> list[tuple[str, int]]:
        """Recursively gather all token counts from a nested dictionary.
        
        Args:
            data: The dictionary or value to search for token counts
            path: Dot-separated path to the current location in the dictionary
            
        Returns:
            List of (token_key, count) tuples where token_key is the full path to the count
        """
        counts = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                # Skip 'total_tokens' as it's not a billable token type
                if key == 'total_tokens':
                    continue
                    
                current_path = f"{path}.{key}" if path else key
                if key.endswith(('_tokens', '_tokens_details')) or 'token' in key.lower():
                    if isinstance(value, int) and value > 0:
                        counts.append((current_path, value))
                    elif isinstance(value, dict):
                        counts.extend(self._gather_token_counts(value, current_path))
                elif isinstance(value, (dict, list)):
                    counts.extend(self._gather_token_counts(value, current_path))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                counts.extend(self._gather_token_counts(item, f"{path}[{i}]"))
                
        return counts

    def get_last_call_stats(self) -> str:
        """Get token usage and price breakdown for the **last** API call as a string.

        Uses ``usage_tracker.tracker`` to obtain the latest usage info recorded
        by ``call_llm_api`` and looks up pricing in :data:`fluent_llm.openai.models.OPENAI_MODELS`.

        All non-zero token categories in the usage dictionary are included, including
        nested fields. If any token category present in the usage stats cannot be 
        priced, a ``RuntimeError`` is raised to surface the missing information.

        Returns:
            A formatted string with token usage and pricing information.
        """
        usage = tracker.last_call_usage
        if not usage:
            return "[fluent-llm] No usage information available for last call."

        model_name: str = usage.get("model")  # type: ignore[arg-type]
        model = get_model_by_name(model_name)
        if model is None:
            raise RuntimeError(f"Pricing data for model '{model_name}' not found.")

        # Build a mapping of all possible token types to their prices
        price_map: dict[str, Decimal | None] = {
            # Core text pricing (OpenAI naming variations)
            "input_tokens": model.price_per_million_text_tokens_input,
            "prompt_tokens": model.price_per_million_text_tokens_input,  # Alias for compatibility
            "output_tokens": model.price_per_million_text_tokens_output,
            "completion_tokens": model.price_per_million_text_tokens_output,  # Alias for compatibility
            # Image pricing (hypothetical keys)
            "image_tokens_input": model.price_per_million_image_tokens_input,
            "image_tokens_output": model.price_per_million_image_tokens_output,
            # Audio pricing (hypothetical keys)
            "audio_tokens_input": model.price_per_million_audio_tokens_input,
            "audio_tokens_output": model.price_per_million_audio_tokens_output,
        }
        # Merge any custom / detailed pricing the model defines
        price_map.update(model.additional_pricing)

        # Gather all token counts from the usage dictionary, including nested ones
        token_counts = self._gather_token_counts(usage)
        
        # Check that all token types have pricing
        missing_pricing = []
        for token_key, count in token_counts:
            # Try to find a matching price key - either the full path or just the last segment
            price_key = token_key
            if price_key not in price_map:
                # Try with just the last part of the path
                last_part = token_key.split('.')[-1]
                if last_part in price_map:
                    price_key = last_part
            
            if price_key not in price_map or (isinstance(price_map[price_key], Decimal) and price_map[price_key].is_nan()):
                missing_pricing.append((token_key, count))

        if missing_pricing:
            missing_list = '\n'.join(f"- {key}: {count} tokens" for key, count in missing_pricing)
            raise RuntimeError(
                f"No pricing configured for the following token types:\n{missing_list}"
            )

        # Now format all the token counts with their prices
        output = []
        for token_key, count in token_counts:
            # Try to find a matching price key - either the full path or just the last segment
            price_key = token_key
            if price_key not in price_map:
                # Try with just the last part of the path
                last_part = token_key.split('.')[-1]
                if last_part in price_map:
                    price_key = last_part
            
            price_per_million = price_map[price_key]
            cost = (Decimal(count) * price_per_million) / Decimal(1_000_000)
            output.append(f"{token_key}: {count} tokens → ${cost:.6f}")

        return "\n".join(output)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    async def call(self, **kwargs: Any) -> Any:
        """
        Convert abstract prompt to OpenAI format and execute asynchronously.

        Args:
            model: The model to use for the completion
            **kwargs: Additional arguments to pass to the OpenAI API

        Returns:
            The response from the OpenAI API, with the format depending on expect_type
        """
        # Call the API
        try:
            # If _expect is a BaseModel subclass, pass as structured_output_model
            return await call_llm_api(
                client=self._client,
                messages=self._messages,
                expect_type=ResponseType.JSON if issubclass(self._expect, BaseModel) else self._expect,
                text={
                    "type": "json_schema",
                    "name": self._expect.__name__,
                    "schema": self._expect.model_json_schema(),
                } if issubclass(self._expect, BaseModel) else None,
                **kwargs,
            )
        except () as e:
            # Handle any API errors here
            raise RuntimeError(f"Error calling LLM API: {str(e)}") from e


# ---------------------------------------------------------------------------
# Public instance
# ---------------------------------------------------------------------------

# Pre-instantiated LLMPromptBuilder for direct use
llm: LLMPromptBuilder = LLMPromptBuilder()
