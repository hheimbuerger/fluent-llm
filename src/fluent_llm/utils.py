import asyncio
from functools import wraps

def asyncify(func):
    """
    Decorator that allows async functions to be called from both sync and async contexts.

    This decorator automatically detects whether it's being called from a synchronous
    context (like a Python REPL or regular function) or an asynchronous context
    (inside an async function with an active event loop). It then handles the
    execution appropriately:

    - In async contexts: Returns the coroutine directly for awaiting
    - In sync contexts: Automatically wraps the call in asyncio.run()

    The decorated function itself is always executed asynchronously - this decorator
    simply provides a bridge that makes async functions callable from sync code
    without requiring the caller to manage event loops manually.

    How it works:
    -----------
    1. Uses asyncio.get_running_loop() to detect if an event loop is active
    2. If an event loop exists: returns the coroutine (for await)
    3. If no event loop exists: creates one with asyncio.run() and executes

    The _is_coroutine attribute:
    ---------------------------
    Sets wrapper._is_coroutine = asyncio.coroutines._is_coroutine to ensure the
    wrapper is properly recognized as a coroutine function by asyncio's inspection
    utilities. This allows tools like asyncio.iscoroutinefunction() to correctly
    identify the decorated function as awaitable, enabling proper async/await syntax
    highlighting and type checking.

    Example:
    --------
    @asyncify
    async def fetch_data(url):
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            return response.json()

    # Works in sync context (REPL, regular functions):
    data = fetch_data("https://api.example.com")

    # Works in async context:
    async def main():
        data = await fetch_data("https://api.example.com")

    Notes:
    ------
    - The decorated function must be async (use async def)
    - All internal logic should assume async context (use await, async clients, etc.)
    - Cannot be used with functions that have required positional arguments after *args
    - Event loop creation via asyncio.run() means sync calls cannot be nested inside
        already-running event loops (this is an asyncio limitation)

    Use Cases:
    ----------
    - Making async libraries REPL-friendly for testing and exploration
    - Providing convenience APIs that work in both sync and async codebases
    - Bridging sync legacy code with new async implementations
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Are we already in an async context?
            asyncio.get_running_loop()
            # Yes - just return the coroutine
            return func(*args, **kwargs)
        except RuntimeError:
            # No - create one and run
            return asyncio.run(func(*args, **kwargs))

    wrapper._is_coroutine = asyncio.coroutines._is_coroutine
    return wrapper
