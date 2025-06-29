# Fluent LLM

Expressive, opinionated, and fluent Python interface for working with LLMs. Clean, intuitive, and production-ready—distributed on PyPI.

## Highlights

- **Expressive:** Write natural, readable, and chainable LLM interactions.
- **Opinionated:** Focuses on best practices and sensible defaults for LLM workflows.
- **Fluent API:** Compose prompts, context, and expectations in a single chain.
- **Supports multimodal (text, image, audio) inputs and outputs.**
- **Modern Python:** Type hints, async/await, and dataclasses throughout.
- **Easy testing:** Built-in support for both mocked and live API tests.
- **Fast setup:** Uses [uv](https://github.com/astral-sh/uv) for blazing-fast dependency management.

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

## Usage Examples

```python
from fluent_llm import llm, ResponseType

response = await llm
    .agent("You are an art evaluator.")
    .context("You received this painting and were tasked to evaluate whether it's museum-worthy.")
    .image("painting.png")
    .expect(ResponseType.TEXT)
    .call()

assert isinstance(response, str)
print(response)
```

```python
from fluent_llm import llm, ResponseType
import PIL

response = await llm
    .agent("You are a 17th century classic painter.")
    .context("You were paid 10 francs for creating a portrait.")
    .request('Create a portrait of Louis XIV.')
    .expect(ResponseType.IMAGE)
    .call()

assert isinstance(response, PIL.Image)
response.show()
```

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
