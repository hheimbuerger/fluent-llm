#!/usr/bin/env python3
"""Command-line interface for fluent-llm.

This module allows running fluent-llm as a module with:
    python -m fluent_llm "llm.request('Who are you?').prompt()"
"""

import sys
import argparse
from typing import Any

from .builder import llm


def execute_llm_code(code: str) -> Any:
    """Execute the provided LLM code string and return the result.

    Args:
        code: Python code string to execute (e.g., "llm.request('Hello').prompt()")

    Returns:
        The result of executing the LLM code
    """
    # Create a safe execution environment with access to the llm object
    namespace = {
        'llm': llm,
        '__builtins__': __builtins__,
    }

    try:
        # Try to compile as an expression first (for simple cases like "llm.request(...).prompt()")
        compiled_code = compile(code, '<string>', 'eval')
        result = eval(compiled_code, namespace)
        return result
    except SyntaxError:
        # If it fails as an expression, try as a statement (for print, assignments, etc.)
        try:
            compiled_code = compile(code, '<string>', 'exec')
            exec(compiled_code, namespace)
            # exec doesn't return a value, so we return None
            return None
        except Exception as e:
            print(f"Error executing code: {e}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error executing code: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Execute fluent-llm requests from the command line",
        prog="python -m fluent_llm"
    )
    parser.add_argument(
        "code",
        help="Python code to execute (e.g., \"llm.request('Who are you?').prompt()\")"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"fluent-llm {getattr(__import__('fluent_llm'), '__version__', 'unknown')}"
    )

    args = parser.parse_args()

    # Run the execution
    result = execute_llm_code(args.code)
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()
