# Fluent LLM

Expressive, opinionated, and intuitive 'fluent interface' Python library for working with LLMs. 

## Mission statement

Express every LLM interaction in your app prototypes in a single statement, without having to reach for documentation, looking up model capabilities, or writing boilerplate code.

## Highlights

- **Expressive:** Write natural, readable, and chainable LLM interactions.
- **Opinionated:** Focuses on best practices and sensible defaults for LLM workflows.
- **Fluent API:** Compose prompts, context, and expectations in a single chain.
- **Supports multimodal (text, image, audio) inputs and outputs:** Automatically picks model based on modalities required.
- **Automatic coroutines** Can be used both in async and sync contexts.
- **Modern Python:** Type hints, async/await, and dataclasses throughout.

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

The `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` environment variables are required for all live API calls.

## Prompt Builder

The `llm` global instance is a `LLMPromptBuilder` instance, which can be used to build prompts.

The following prompt components can be used in an arbitrary order and multiple times:

* `.agent(str)`: Sets the agent description, defines system behavior.
* `.context(str)`: Passes textual context to the LLM.
* `.request(str)`: Passes the main request to the LLM.
* `.image(str)`: Passes an image to the LLM.
* `.audio(str)`: Passes an audio file to the LLM.

The prompt chain is terminated by the following methods:

* `.prompt(): str`: Sends the prompt to the LLM and expects a text response.
* `.prompt_for_image(): PIL.Image`: Sends the prompt to the LLM and expects an image response.
* `.prompt_for_audio(): soundfile.SoundFile`: Sends the prompt to the LLM and expects an audio response.
* `.prompt_for_structured_output(pydantic_model): BaseModel`: Sends the prompt to the LLM and expects a structured response.

They will either return the desired response if processing was successful, or raise an exception otherwise.

## Usage

### Callable module

You can use this library as a callable module to experiment with LLMs.

```bash
> pip install fluent-llm
> fluent-llm "llm.request('1+2=?').prompt()"
1 + 2 = 3.
```

Or even easier, without installing, as a tool with uvx:

```bash
uvx fluent-llm "llm.request('1+2=?').prompt()"
1 + 2 = 3.
```

### As a library

```python
response = llm \
    .agent("You are an art evaluator.") \
    .context("You received this painting and were tasked to evaluate whether it's museum-worthy.") \
    .image("painting.png") \
    .prompt()
print(response)
```

### Async/await

Just works. See if you can spot the difference to the example above.

```python
response = await llm \
    .agent("You are an art evaluator.") \
    .context("You received this painting and were tasked to evaluate whether it's museum-worthy.") \
    .image("painting.png") \
    .prompt()
print(response)
```

### Multimodality

```python
response = llm
    .agent("You are a 17th century classic painter.")
    .context("You were paid 10 francs for creating a portrait.")
    .request('Create a portrait of Louis XIV.')
    .prompt_for_image()

assert isinstance(response, PIL.Image)
response.show()
```

### Usage tracking

Usage tracking and price estimations for the last call are built-in.

```python
>>> llm.request('How are you?').prompt()
"I'm doing well, thank you! How about you?"

>>> print(llm.usage)
=== Last API Call Usage ===
Model: gpt-4o-mini-2024-07-18
input_tokens: 11 tokens
output_tokens: 12 tokens

💰 Cost Breakdown:
  input_tokens: 11 tokens → $0.000002
  output_tokens: 12 tokens → $0.000007

💵 Total Call Cost: $0.000009
==============================

>>> llm.usage.cost.total_call_cost_usd
0.000009

>>> llm.usage.cost.breakdown['input_tokens'].count
11
```

### Automatic Model Selection (recommended)

If choosing a provider or model per-invocation is not sufficient, you can define
a custom `ModelSelectionStrategy` and pass it to the `LLMPromptBuilder` constructor to select provider and model based on your own criteria.

### Provider and Model per-prompt override

You can specify preferred providers and models using the fluent chain API:

```python
# Use a specific provider (will select best available model)
response = await llm \
    .provider("anthropic") \
    .request("Hello, how are you?") \
    .prompt()

# Use a specific model
response = await llm \
    .model("claude-sonnet-4-20250514") \
    .request("Write a poem about coding") \
    .prompt()

# Combine provider and model preferences
response = await llm \
    .provider("openai") \
    .model("gpt-4.1-mini") \
    .request("Explain quantum computing") \
    .prompt()
```

## Customization

If the defaults are not sufficient, you can customize the behavior of the builder by creating your own `LLMPromptBuilder`, instead of using the `llm` global instance provided for convenience.

However, note that you're probably quickly reaching the point at which you should ask yourself if you're not better off using the official OpenAI Python client library directly. This library is designed to be a simple and opinionated wrapper around the OpenAI API, and it's not intended to be a full-featured LLM client.

### Invocation

Instead of using the convenience methods `.prompt_*()`, you can use the `.call()` method to execute the prompt and return a response.

### Client

Pass in a custom `client` to the `.call()` method, to use a custom client for the API call.

## Contribution

### Setup

```bash
uv sync --dev
```

- Installs all runtime and development dependencies (including pytest).
- Requires [uv](https://github.com/astral-sh/uv) for fast, modern Python dependency management.

### Running Tests

All tests are run with `uv`:

```bash
uv run pytest
```

### Mocked Tests
- Located in `tests/test_mocked.py`.
- Do **not** require a real OpenAI API key or network access.
- Fast and safe for CI or local development.

### Live API Tests
- Located in `tests/test_live_api_*.py`.
- **Require** a valid API KEY and internet access.
- Will consume credits!
- Run only when you want to test real OpenAI integration.

## License

Licensed under the MIT License.

## Disclaimer

Almost all code written by Claude, o3 and SWE-1, concept and design by @hheimbuerger.
