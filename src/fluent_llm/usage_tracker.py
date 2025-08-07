"""Simple usage tracker for LLM API calls.

This module provides functionality to track the last API call usage
and calculate pricing based on provider model data.
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass
from decimal import Decimal
from io import StringIO

from .providers.provider import LLMProvider


@dataclass(frozen=True)
class TokenCostBreakdown:
    """Cost breakdown for a specific token type."""
    count: int
    price_per_million_usd: float
    cost_usd: float


@dataclass(frozen=True)
class UsageCost:
    """Complete cost information for an API call."""
    model: str
    breakdown: Dict[str, TokenCostBreakdown]
    total_call_cost_usd: float


class LastCallUsage:
    """Stores usage data for the most recent API call as a flat dictionary."""

    def __init__(self, provider: LLMProvider, model: str, usage_dict: Dict[str, int]):
        self.provider = provider
        self.model = model
        self.usage_dict = usage_dict  # Flat dictionary of usage descriptors to counts


class UsageTrackingError(Exception):
    """Raised when usage tracking encounters an error."""


class UsageTracker:
    """Simple tracker that only stores the last API call usage."""

    def __init__(self):
        """Initialize tracker with no stored usage."""
        self._last_call: Optional[LastCallUsage] = None
        self._cached_cost: Optional[Dict[str, Any]] = None

    def _flatten_usage(self, usage: Any) -> Dict[str, int]:
        """Flatten usage object into a dictionary of usage descriptors to counts.

        Args:
            usage: Usage object from the API response.

        Returns:
            Dictionary mapping usage descriptors to token counts (only non-zero counts).
        """
        usage_dict = {}

        # Extract basic token counts
        input_tokens = getattr(usage, 'input_tokens', None) or getattr(usage, 'prompt_tokens', 0)
        output_tokens = getattr(usage, 'output_tokens', None) or getattr(usage, 'completion_tokens', 0)

        if input_tokens > 0:
            usage_dict['input_tokens'] = input_tokens
        if output_tokens > 0:
            usage_dict['output_tokens'] = output_tokens

        # TODO: add here for Anthropic: cache_creation_input_tokens, cache_read_input_tokens, server_tool_use

        # Extract and flatten input token details
        if hasattr(usage, 'input_tokens_details'):
            input_details_iter = getattr(usage.input_tokens_details, 'items', usage.input_tokens_details)
            for detail_key, count in input_details_iter:
                if isinstance(count, int) and count > 0:
                    usage_dict[f'input_tokens_details.{detail_key}'] = count

        # Extract and flatten output token details
        if hasattr(usage, 'output_tokens_details'):   # yes, at least the OpenAI API sometimes doesn't have this!
            output_details_iter = getattr(usage.output_tokens_details, 'items', usage.output_tokens_details)
            for detail_key, count in output_details_iter:
                if isinstance(count, int) and count > 0:
                    usage_dict[f'output_tokens_details.{detail_key}'] = count

        return usage_dict

    def track_usage(self, provider: LLMProvider, model: str, usage: Any) -> None:
        """Track usage from the most recent API call.

        Args:
            provider: The provider instance used in the API call.
            model: The model name used in the API call.
            usage: Usage object from the API response.
        """
        # Flatten usage data into a simple dictionary
        usage_dict = self._flatten_usage(usage)

        # Store the flattened usage data
        self._last_call = LastCallUsage(provider, model, usage_dict)
        # Clear cached cost since we have new data
        self._cached_cost = None

    def get_last_call_usage(self) -> Optional[LastCallUsage]:
        """Get the usage data for the last API call."""
        return self._last_call

    @property
    def cost(self) -> UsageCost:
        """Get the cost breakdown for the last API call (cached).

        Returns:
            UsageCost object with detailed cost breakdown.

        Raises:
            UsageTrackingError: If no calls tracked or pricing information is missing.
        """
        if not self._last_call:
            raise UsageTrackingError("No API calls tracked yet.")

        # Return cached result if available
        if self._cached_cost is not None:
            return self._cached_cost

        # Calculate and cache the cost
        cost_breakdown = {}
        token_type_mapping = self._last_call.provider.get_token_type_to_price_mapping()
        llm_model = self._last_call.provider.get_model_by_name(self._last_call.model)

        # Process each usage descriptor in the flattened dictionary
        for usage_descriptor, count in self._last_call.usage_dict.items():
            # Map token type to pricing field
            if usage_descriptor not in token_type_mapping:
                raise UsageTrackingError(
                    f"Unknown token type '{usage_descriptor}' for provider '{self._last_call.provider.__class__.__name__}'. "
                    f"Add mapping to provider's get_token_type_mapping(). Available mappings: {list(token_type_mapping.keys())}"
                )

            pricing_field = token_type_mapping[usage_descriptor]

            # Get the pricing value
            pricing_value = getattr(llm_model, pricing_field, None)
            if pricing_value is None or pricing_value.is_nan():
                raise UsageTrackingError(
                    f"Missing pricing for {usage_descriptor} tokens in model '{llm_model.name}'. "
                    f"Expected field: {pricing_field}"
                )

            # Calculate cost
            cost_calc = (Decimal(count) / Decimal('1000000')) * pricing_value

            # Add to breakdown with price per million tokens
            price_per_million = float(pricing_value)
            cost_breakdown[usage_descriptor] = TokenCostBreakdown(
                count=count,
                price_per_million_usd=price_per_million,
                cost_usd=float(cost_calc)
            )

        # Calculate total cost
        total_cost = sum(item.cost_usd for item in cost_breakdown.values())

        # Cache the result
        self._cached_cost = UsageCost(
            model=self._last_call.model,
            breakdown=cost_breakdown,
            total_call_cost_usd=total_cost
        )

        return self._cached_cost

    def __str__(self):
        return self.generate_report()

    def generate_report(self) -> str:
        """Return a formatted report of the last API call usage and cost."""
        result = StringIO()

        if not self._last_call:
            raise UsageTrackingError("No API calls tracked yet.")

        cost_info = self.cost

        result.write("=== Last API Call Usage ===\n")
        result.write(f"Model: {self._last_call.model}\n")

        # Show token counts from the flattened usage dictionary
        for usage_descriptor, count in self._last_call.usage_dict.items():
            result.write(f"{usage_descriptor}: {count:,} tokens\n")

        result.write("\n💰 Cost Breakdown:\n")
        for usage_descriptor, details in cost_info.breakdown.items():
            result.write(f"  {usage_descriptor}: {details.count:,} tokens @ ${details.price_per_million_usd}/MTok → ${details.cost_usd:.6f}\n")

        result.write(f"\n💵 Total Call Cost: ${cost_info.total_call_cost_usd:.6f}\n")

        result.write("" + "="*30 + "\n")

        return result.getvalue()


# Global instance for convenience
tracker = UsageTracker()
