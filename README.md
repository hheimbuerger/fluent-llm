# Fluent LLM

Expressive, opinionated, and fluent Python interface for working with LLMs. Clean, intuitive, and production-ready—distributed on PyPI.

## Highlights

- **Expressive:** Write natural, readable, and chainable LLM interactions.
- **Opinionated:** Focuses on best practices and sensible defaults for LLM workflows.
- **Fluent API:** Compose prompts, context, and expectations in a single chain.
- **Supports multimodal (text, image, audio) inputs and outputs:** Automatically picks model based on modalities required.
- **Automatic coroutines** Can be used both in async and sync contexts.
- **Modern Python:** Type hints, async/await, and dataclasses throughout.

## Installation

```bash
uv sync --dev
```
- Installs all runtime and development dependencies (including pytest).
- Requires [uv](https://github.com/astral-sh/uv) for fast, modern Python dependency management.

## Setting API Keys

Set your OpenAI API key before running any code that calls the API:

```bash
# On Unix/macOS
export OPENAI_API_KEY=sk-...

# On Windows (cmd)
set OPENAI_API_KEY=sk-...

# On Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."
```

The `OPENAI_API_KEY` environment variable is required for all live API calls.

## Usage

### Basics

```python
from fluent_llm import llm

response = await llm \
    .agent("You are an art evaluator.") \
    .context("You received this painting and were tasked to evaluate whether it's museum-worthy.") \
    .image("painting.png") \
    .prompt()

assert isinstance(response, str)
print(response)
```

### Async/await

Just works. See if you can spot the difference to the example above.

```python
from fluent_llm import llm

response = await llm \
    .agent("You are an art evaluator.") \
    .context("You received this painting and were tasked to evaluate whether it's museum-worthy.") \
    .image("painting.png") \
    .prompt()

assert isinstance(response, str)
print(response)
```

### CLI usage

```bash
uvx fluent-llm llm.agent('foo').request('bar')
```

### Multimodality

```python
from fluent_llm import llm, ResponseType
import PIL

response = llm
    .agent("You are a 17th century classic painter.")
    .context("You were paid 10 francs for creating a portrait.")
    .request('Create a portrait of Louis XIV.')
    .prompt_for_image()

assert isinstance(response, PIL.Image)
response.show()
```

## Customization

If the defaults are not sufficient, you can customize the behavior of the builder by creating your own `LLMPromptBuilder`, instead of using the `llm` global instance provided for convenience.

However, note that you're probably quickly reaching the point at which you should ask yourself if you're not better off using the official OpenAI Python client library directly. This library is designed to be a simple and opinionated wrapper around the OpenAI API, and it's not intended to be a full-featured LLM client.

### Model Selection

Pass in a custom `ModelSelectionStrategy` to the `LLMPromptBuilder` constructor, to select provider and model based on your own criteria.

### Invocation

Instead of using the convenience methods `.prompt_*()`, you can use the `.call()` method to execute the prompt and return a response.

### Client

Pass in a custom `client` to the `.call()` method, to use a custom client for the API call.

## Running Tests

All tests are run with `uv`:

```bash
uv run pytest
```

### Mocked Tests
- Located in `tests/test_mocked.py`.
- Do **not** require a real OpenAI API key or network access.
- Fast and safe for CI or local development.

### Live API Tests
- Located in `tests/test_live_api.py`.
- **Require** a valid `OPENAI_API_KEY` and internet access.
- Run only when you want to test real OpenAI integration.
- To run only live tests:
  ```bash
  uv run pytest tests/test_live_api.py
  ```
- To run only mocked tests:
  ```bash
  uv run pytest tests/test_mocked.py
  ```

## License

Licensed under the MIT License.

## Disclaimer

Almost all code written by o3 and SWE-1, concept and design by @hheimbuerger.
