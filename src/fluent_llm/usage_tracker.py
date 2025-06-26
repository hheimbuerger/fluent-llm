"""
Module for tracking LLM API usage statistics.

This module provides functionality to track and manage usage statistics for different LLM models.
It can extract usage information from API responses and maintain cumulative usage statistics.
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class UsageStats:
    """Dataclass to store usage statistics for a model (responses API fields)."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0
    # Optionally store details (last call only)
    input_tokens_details: dict = None
    output_tokens_details: dict = None


class UsageTracker:
    """Tracks usage statistics for LLM API calls.

    This class maintains cumulative usage statistics per model and provides methods
    to update these statistics from API responses.
    """

    def __init__(self):
        """Initialize a new UsageTracker with empty statistics."""
        self._usage_stats: Dict[str, UsageStats] = {}
        self.last_call_usage: Optional[dict] = None

    def track_usage(self, response: Any) -> None:
        """Track usage from an API response object.

        Args:
            response: The API response object containing usage information.
                     Expected to have 'model' and 'usage' attributes with token counts.
        """
        if not response or not hasattr(response, 'model') or not hasattr(response, 'usage'):
            return

        model = response.model
        usage = response.usage

        # Convert usage to dictionary, handling both object and dict
        usage_dict = {}
        usage_attrs = [
            'input_tokens', 'output_tokens', 'total_tokens',
            'prompt_tokens', 'completion_tokens',  # For backward compatibility
            'input_tokens_details', 'output_tokens_details'
        ]

        for attr in usage_attrs:
            value = getattr(usage, attr, None)
            if value is not None:
                usage_dict[attr] = value

        # Handle backward compatibility for prompt/completion tokens
        if 'prompt_tokens' in usage_dict and 'input_tokens' not in usage_dict:
            usage_dict['input_tokens'] = usage_dict['prompt_tokens']
        if 'completion_tokens' in usage_dict and 'output_tokens' not in usage_dict:
            usage_dict['output_tokens'] = usage_dict['completion_tokens']

        # Ensure total_tokens is set
        if 'total_tokens' not in usage_dict and 'input_tokens' in usage_dict and 'output_tokens' in usage_dict:
            usage_dict['total_tokens'] = usage_dict['input_tokens'] + usage_dict['output_tokens']

        if model not in self._usage_stats:
            self._usage_stats[model] = UsageStats()

        stats = self._usage_stats[model]

        # Update cumulative stats
        stats.input_tokens += usage_dict.get('input_tokens', 0)
        stats.output_tokens += usage_dict.get('output_tokens', 0)
        stats.total_tokens += usage_dict.get('total_tokens', 0)
        stats.call_count += 1

        # Store details from the last call
        if 'input_tokens_details' in usage_dict:
            stats.input_tokens_details = usage_dict.get('input_tokens_details')
        if 'output_tokens_details' in usage_dict:
            stats.output_tokens_details = usage_dict.get('output_tokens_details')

        # Store raw usage for last call
        self.last_call_usage = {'model': model, 'usage': usage_dict}

    def get_usage(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Get usage statistics.

        Args:
            model: If provided, get statistics for this specific model.
                  If None, get combined statistics for all models.

        Returns:
            A dictionary containing the usage statistics.
        """
        if model:
            if model not in self._usage_stats:
                return {}
            return self._format_stats(self._usage_stats[model])

        # Combine stats for all models
        combined = UsageStats()
        for stats in self._usage_stats.values():
            combined.prompt_tokens += stats.prompt_tokens
            combined.completion_tokens += stats.completion_tokens
            combined.total_tokens += stats.total_tokens
            combined.call_count += stats.call_count

        return self._format_stats(combined)

    def reset_usage(self, model: Optional[str] = None) -> None:
        """Reset usage statistics.

        Args:
            model: If provided, reset statistics for this specific model.
                  If None, reset all statistics.
        """
        if model:
            if model in self._usage_stats:
                del self._usage_stats[model]
        else:
            self._usage_stats.clear()

    def _format_stats(self, stats: UsageStats) -> Dict[str, Any]:
        """Format usage statistics into a dictionary."""
        return {
            'input_tokens': stats.input_tokens,
            'output_tokens': stats.output_tokens,
            'total_tokens': stats.total_tokens,
            'call_count': stats.call_count,
            'input_tokens_details': stats.input_tokens_details,
            'output_tokens_details': stats.output_tokens_details,
        }

    def get_models(self) -> list:
        """Get a list of all tracked models.

        Returns:
            A list of model names that have usage statistics.
        """
        return list(self._usage_stats.keys())


# Global instance for convenience
tracker = UsageTracker()

def get_last_call_usage() -> Optional[dict]:
    """Get the usage statistics for the last API call."""
    return tracker.last_call_usage

def track_usage(response: Dict[str, Any]) -> None:
    """Convenience function to track usage from a response using the global tracker.

    Args:
        response: The API response dictionary containing usage information.
    """
    tracker.track_usage(response)

def get_usage(model: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to get usage statistics from the global tracker.

    Args:
        model: If provided, get statistics for this specific model.
              If None, get combined statistics for all models.

    Returns:
        A dictionary containing the usage statistics.
    """
    return tracker.get_usage(model)

def reset_usage(model: Optional[str] = None) -> None:
    """Convenience function to reset usage statistics in the global tracker.

    Args:
        model: If provided, reset statistics for this specific model.
              If None, reset all statistics.
    """
    tracker.reset_usage(model)
