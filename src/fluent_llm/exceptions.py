"""Exceptions used throughout the fluent-llm package."""

class LLMRefusalError(RuntimeError):
    """Raised when the LLM refuses to process a request.
    
    This typically happens when the request violates the model's content policy
    or other safety constraints.
    """
    def __init__(self, message: str, *args, **kwargs):
        super().__init__(message, *args, **kwargs)
        self.message = message

    def __str__(self) -> str:
        return f"LLM refused to process the request: {self.message}"
