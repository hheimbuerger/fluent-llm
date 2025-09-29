"""Tool calling support for fluent LLM library.

This module provides the core infrastructure for defining and managing tools
that can be called by AI models during conversations.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Union, get_type_hints, get_origin, get_args
from pydantic import TypeAdapter


@dataclass
class Tool:
    """Represents a tool that can be called by an AI model.
    
    A tool consists of a name, description, the actual function to call,
    and a JSON schema that describes the function's parameters.
    """
    name: str
    description: str
    function: Callable[..., Any]
    schema: dict
    
    @classmethod
    def from_function(cls, func: Callable[..., Any]) -> "Tool":
        """Create a Tool from a function, auto-deriving metadata.
        
        Args:
            func: The function to create a tool from. Must have type annotations.
            
        Returns:
            A Tool instance with auto-generated name, description, and schema.
            
        Raises:
            ValueError: If the function lacks proper type annotations or docstring.
        """
        name = func.__name__
        description = func.__doc__ or f"Tool: {func.__name__}"
        schema = generate_tool_schema(func)
        
        return cls(
            name=name,
            description=description.strip(),
            function=func,
            schema=schema
        )


def generate_tool_schema(func: Callable[..., Any]) -> dict:
    """Generate JSON schema from function signature using Pydantic TypeAdapter.
    
    Args:
        func: The function to generate a schema for.
        
    Returns:
        A JSON schema dictionary describing the function's parameters.
        
    Raises:
        ValueError: If the function has parameters without type annotations.
    """
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        if param_name not in type_hints:
            raise ValueError(f"Parameter '{param_name}' in function '{func.__name__}' lacks type annotation")
        
        param_type = type_hints[param_name]
        
        # Generate schema for this parameter using Pydantic TypeAdapter
        try:
            adapter = TypeAdapter(param_type)
            param_schema = adapter.json_schema()
            properties[param_name] = param_schema
        except Exception as e:
            raise ValueError(f"Could not generate schema for parameter '{param_name}' of type {param_type}: {e}")
        
        # Determine if parameter is required
        is_optional = _is_optional_type(param_type)
        has_default = param.default != inspect.Parameter.empty
        
        if not is_optional and not has_default:
            required.append(param_name)
    
    return {
        "type": "object",
        "properties": properties,
        "required": required
    }


def _is_optional_type(param_type: type) -> bool:
    """Check if a type annotation represents an optional parameter.
    
    Args:
        param_type: The type annotation to check.
        
    Returns:
        True if the type is Optional[T] or Union[T, None], False otherwise.
    """
    origin = get_origin(param_type)
    
    # Handle Union types (including Optional which is Union[T, None])
    if origin is Union:
        args = get_args(param_type)
        # Check if None is one of the union members
        return type(None) in args
    
    return False