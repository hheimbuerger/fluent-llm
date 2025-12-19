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

## Examples

### Simple Text Interaction

```python
from fluent_llm import llm

# One-shot prompt
response = llm \
    .agent("You are a helpful math tutor.") \
    .request("What is the Pythagorean theorem?") \
    .prompt()

print(response)
```

### Multi-Turn Conversation with Continuation

```python
from fluent_llm import llm

# Start a conversation
conversation = llm \
    .agent("You are a Python expert.") \
    .request("What are list comprehensions?") \
    .prompt_conversation()

async for message in conversation:
    print(f"Assistant: {message.text}")

# Continue the conversation
follow_up = conversation.continuation \
    .request("Can you show me an example?") \
    .prompt_conversation()

async for message in follow_up:
    print(f"Assistant: {message.text}")
```

### Few-Shot Learning with Assistant Injection

```python
from fluent_llm import llm

# Use assistant messages to provide examples
response = llm \
    .agent("You are a sentiment analyzer.") \
    .request("I love this product!") \
    .assistant("Positive") \
    .request("This is terrible.") \
    .assistant("Negative") \
    .request("The weather is nice today.") \
    .prompt()

print(response)  # Expected: "Positive"
```

### Tool Calling

```python
from fluent_llm import llm

def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"Weather in {location}: Sunny, 72°F"

def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)

# Use tools in a conversation
conversation = llm \
    .agent("You are a helpful assistant with access to tools.") \
    .tools(get_weather, calculate) \
    .request("What's the weather in Paris and what's 15 * 23?") \
    .prompt_conversation()

async for message in conversation:
    if hasattr(message, 'tool_name'):
        print(f"Tool called: {message.tool_name}")
        print(f"Result: {message.result}")
    else:
        print(f"Assistant: {message.text}")
```

### Conversation Serialization and Restoration

```python
from fluent_llm import llm

# Create and save a conversation
conversation = llm \
    .agent("You are a creative writer.") \
    .request("Write the beginning of a story about a robot.") \
    .prompt_conversation()

async for message in conversation:
    print(message.text)

conversation.save("story_conversation.json")

# Later, load and continue with a different model
restored = llm.load_conversation("story_conversation.json")
continuation = restored.continuation \
    .provider("anthropic") \
    .request("Continue the story with a plot twist.") \
    .prompt_conversation()

async for message in continuation:
    print(message.text)

# Save the updated conversation
continuation.save("story_conversation_continued.json")
```

### Audio Processing

```python
from fluent_llm import llm

# Transcribe audio
transcription = llm \
    .audio("meeting_recording.mp3") \
    .request("Transcribe this audio and summarize the key points.") \
    .prompt()

print(transcription)
```

### Image Analysis

```python
from fluent_llm import llm

# Analyze an image
analysis = llm \
    .agent("You are an art critic.") \
    .image("painting.jpg") \
    .request("Analyze this painting's composition and style.") \
    .prompt()

print(analysis)
```

### Structured Output

```python
from fluent_llm import llm
from pydantic import BaseModel

class Recipe(BaseModel):
    name: str
    ingredients: list[str]
    instructions: list[str]
    prep_time_minutes: int

recipe = llm \
    .request("Give me a recipe for chocolate chip cookies.") \
    .prompt_for_type(Recipe)

print(f"Recipe: {recipe.name}")
print(f"Prep time: {recipe.prep_time_minutes} minutes")
print(f"Ingredients: {', '.join(recipe.ingredients)}")
```

## Overview

This library supports two related, but distinct prompt building paradigms:

1. One-shot prompts: you construct a prompt, send it, get a direct response in an immediately usable format (no `Response`-type class).
2. Multi-turn conversations: construct a prompt and use it to start a conversation, then request multiple responses from the LLM (potentially including tool calls), send a follow-up prompt, etc.

### Three-Class Architecture

Fluent LLM uses a clean three-class architecture that separates concerns:

1. **MessageList** (Mutable Data Container): Holds the conversation messages and handles serialization
2. **LLMConversation** (Mutable Execution Context): Owns the MessageList and manages API calls
3. **LLMPromptBuilder** (Immutable Composition Tool): Accumulates changes as deltas and applies them to conversations

**Key Principles:**
- **Single Source of Truth**: Messages exist only in the conversation's MessageList
- **Delta Pattern**: Builders accumulate changes and apply them on execution
- **Clear Mutability**: MessageList and Conversation are mutable for execution needs, Builder is immutable for composition safety
- **Reference-Based Continuation**: Continuation builders automatically reference their source conversation

## Constructing prompts

The `llm` global instance can be used to build prompts, using the following mutators:

* `.agent(str)`: Sets the agent description, defines system behavior.
* `.assistant(str)`: Injects an assistant message into the conversation (useful for priming or few-shot examples).
* `.context(str)`: Passes textual context to the LLM.
* `.request(str)`: Passes the main request to the LLM. (Identical to `.context()`, just used to clarify the intent.)
* `.image(filename | PIL.Image)`: Passes an image to the LLM.
* `.audio(filename | soundfile.SoundFile)`: Passes an audio file to the LLM.
* `.tool(tool_func)` or `.tools(tool_func1, tool_func2, ...)`: Registers functions as potential tool calls to offer to the LLM.

Other mutators change the behavior of the system, e.g. `.provider()`, `.model()` and `.call_limit()`. We'll discuss these later.

### Assistant Message Injection

The `.assistant()` method allows you to inject assistant messages into your conversation. This is useful for:

- **Few-shot learning**: Provide example responses to guide the model's behavior
- **Conversation priming**: Start with a specific assistant response
- **Conversation restoration**: Continue from a saved conversation state

```python
# Few-shot example
response = llm \
    .agent("You are a helpful translator.") \
    .request("Translate 'hello' to French") \
    .assistant("Bonjour") \
    .request("Translate 'goodbye' to French") \
    .prompt()
# Expected: "Au revoir"

# Priming a conversation
conversation = llm \
    .agent("You are a creative writer.") \
    .assistant("I'm ready to help you craft amazing stories!") \
    .request("Write a short story about a robot") \
    .prompt_conversation()
```

## Submitting prompts

When your prompt has been constructed, you submit it to the LLM in different ways, depending on the paradigm you require.

### One-shot prompts

To get a one-shot response, use one of the following methods:

* `.prompt(): str`: Sends the prompt to the LLM and returns a text response.
* `.prompt_for_image(): PIL.Image`: Sends the prompt to the LLM and returns an image response.
* *[to be implemented]* ~~`.prompt_for_audio(): soundfile.SoundFile`: Sends the prompt to the LLM and returns an audio response.~~
* `.prompt_for_structured_output(pydantic_model): BaseModel`: Sends the prompt to the LLM and returns a Python object instance.

They will either return the desired response if processing was successful, or raise an exception otherwise.

### Multi-turn conversations

Alternatively, begin a conversation:

* `.prompt_conversation()`: Starts a conversation with the LLM, and returns a `LLMConversation` instance.

This instance implements the async generator protocol, and can be used to iterate over the responses from the LLM.

```python
conversation = llm \
    .agent("You are a helpful assistant.") \
    .request("What is Python?") \
    .prompt_conversation()

async for message in conversation:
    print(f"Assistant: {message.text}")
```

Afterwards, you can retrieve a new builder from `conversation.continuation`, which you can use to follow-up with more prompts and keep the conversation going.

```python
# Continue the conversation
follow_up = conversation.continuation \
    .request("Tell me more about Python functions") \
    .prompt_conversation()

async for message in follow_up:
    print(f"Assistant: {message.text}")
```

### Conversation Continuation Patterns

The continuation system allows you to seamlessly continue conversations:

```python
# Start a conversation
conversation = llm \
    .agent("You are a math tutor.") \
    .request("What is 2 + 2?") \
    .prompt_conversation()

async for message in conversation:
    print(message.text)  # "2 + 2 equals 4."

# Continue with follow-up questions
continuation = conversation.continuation \
    .request("What about 3 + 3?") \
    .prompt_conversation()

async for message in continuation:
    print(message.text)  # "3 + 3 equals 6."

# Access continuation at any time during iteration
conversation = llm.request("Count to 5").prompt_conversation()
count = 0
async for message in conversation:
    count += 1
    if count == 2:
        # Stop early and continue with a different request
        break

# The conversation has the partial response
follow_up = conversation.continuation \
    .request("Now count backwards from 5") \
    .prompt_conversation()
```

## Getting Started

### Install with uv

TBD

### Setting API Keys

```bash
# On Unix/macOS
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-...

# On Windows (cmd)
set OPENAI_API_KEY=sk-...
set ANTHROPIC_API_KEY=sk-...

# On Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."
$env:ANTHROPIC_API_KEY="sk-..."
```

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

### Structured output

```python
from pydantic import BaseModel

class PaintingEvaluation(BaseModel):
    museum_worthy: bool
    reason: str

response = llm \
    .agent("You are an art evaluator.") \
    .context("You received this painting and were tasked to evaluate whether it's museum-worthy.") \
    .image("painting.png") \
    .prompt_for_type(PaintingEvaluation)
print(response)
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

## Conversation Serialization

Fluent LLM supports model-agnostic conversation serialization, allowing you to save and restore conversations across sessions or even switch between different LLM providers.

### Saving Conversations

Use the `.save()` method on a conversation to persist it:

```python
# Create and execute a conversation
conversation = llm \
    .agent("You are a helpful assistant.") \
    .request("What is Python?") \
    .prompt_conversation()

async for message in conversation:
    print(message.text)

# Save to file (string path)
conversation.save("my_conversation.json")

# Save to Path object
from pathlib import Path
conversation.save(Path("conversations/session1.json"))

# Save to stream
with open("conversation.json", "w") as f:
    conversation.save(f)
```

### Loading Conversations

Use the `.load_conversation()` method on a builder to restore a conversation:

```python
# Load from file (string path)
conversation = llm.load_conversation("my_conversation.json")

# Load from Path object
from pathlib import Path
conversation = llm.load_conversation(Path("conversations/session1.json"))

# Load from stream
with open("conversation.json", "r") as f:
    conversation = llm.load_conversation(f)

# Load from dictionary
import json
with open("conversation.json", "r") as f:
    data = json.load(f)
conversation = llm.load_conversation(data)
```

### Continuing Restored Conversations

Once loaded, you can continue conversations with any configuration:

```python
# Load a conversation
conversation = llm.load_conversation("my_conversation.json")

# Continue with a different provider or model
continuation = conversation.continuation \
    .provider("anthropic") \
    .model("claude-sonnet-4-20250514") \
    .request("Tell me more") \
    .prompt_conversation()

async for message in continuation:
    print(message.text)

# Save the updated conversation
continuation.save("my_conversation_continued.json")
```

### Model-Agnostic Serialization

The serialization format only includes message data, not configuration:

```python
# Start with OpenAI
conversation = llm \
    .provider("openai") \
    .model("gpt-4o") \
    .request("What is machine learning?") \
    .prompt_conversation()

async for message in conversation:
    print(message.text)

# Save the conversation
conversation.save("ml_conversation.json")

# Later, load and continue with Anthropic
restored = llm.load_conversation("ml_conversation.json")
continuation = restored.continuation \
    .provider("anthropic") \
    .model("claude-sonnet-4-20250514") \
    .request("Explain neural networks") \
    .prompt_conversation()

async for message in continuation:
    print(message.text)
```

### Serialization Format

Conversations are serialized as JSON with the following structure:

```json
{
  "messages": [
    {
      "type": "AgentMessage",
      "text": "You are a helpful assistant",
      "role": "system"
    },
    {
      "type": "TextMessage",
      "text": "What is Python?",
      "role": "user"
    },
    {
      "type": "TextMessage",
      "text": "Python is a high-level programming language...",
      "role": "assistant"
    }
  ],
  "version": "1.0"
}
```

## Tool Calls

TBD

## API Reference

### LLMPromptBuilder

The immutable builder class for composing prompts. All methods return new builder instances.

#### Message Composition Methods

- **`.agent(text: str) -> LLMPromptBuilder`**: Add a system message defining agent behavior
  ```python
  builder = llm.agent("You are a helpful coding assistant.")
  ```

- **`.assistant(text: str) -> LLMPromptBuilder`**: Inject an assistant message (for few-shot examples or priming)
  ```python
  builder = llm.assistant("I'm ready to help with your code!")
  ```

- **`.context(text: str) -> LLMPromptBuilder`**: Add user context
  ```python
  builder = llm.context("Here is some background information...")
  ```

- **`.request(text: str) -> LLMPromptBuilder`**: Add a user request (same as context, but clarifies intent)
  ```python
  builder = llm.request("Explain how async/await works")
  ```

- **`.image(source: str | Path | PIL.Image) -> LLMPromptBuilder`**: Add an image to the prompt
  ```python
  builder = llm.image("diagram.png")
  ```

- **`.audio(source: str | Path | SoundFile) -> LLMPromptBuilder`**: Add audio to the prompt
  ```python
  builder = llm.audio("recording.mp3")
  ```

#### Configuration Methods

- **`.provider(name: str) -> LLMPromptBuilder`**: Set preferred provider
  ```python
  builder = llm.provider("anthropic")
  ```

- **`.model(name: str) -> LLMPromptBuilder`**: Set preferred model
  ```python
  builder = llm.model("claude-sonnet-4-20250514")
  ```

- **`.tools(*functions) -> LLMPromptBuilder`**: Register tool functions
  ```python
  def get_weather(location: str) -> str:
      return f"Weather in {location}: Sunny, 72°F"
  
  builder = llm.tools(get_weather)
  ```

#### Execution Methods

- **`.prompt(**kwargs) -> str`**: Execute one-shot and return text response
  ```python
  response = llm.request("What is 2+2?").prompt()
  ```

- **`.prompt_for_image(**kwargs) -> PIL.Image`**: Execute one-shot and return image
  ```python
  image = llm.request("Draw a sunset").prompt_for_image()
  ```

- **`.prompt_for_type(model: Type[BaseModel], **kwargs) -> BaseModel`**: Execute one-shot and return structured output
  ```python
  from pydantic import BaseModel
  
  class Person(BaseModel):
      name: str
      age: int
  
  person = llm.request("Extract: John is 30 years old").prompt_for_type(Person)
  ```

- **`.prompt_conversation(**kwargs) -> LLMConversation`**: Start a multi-turn conversation
  ```python
  conversation = llm.request("Hello").prompt_conversation()
  ```

#### Serialization Methods

- **`.load_conversation(source: str | Path | IO | dict) -> LLMConversation`**: Load a conversation from file, stream, or dict
  ```python
  # From file path
  conversation = llm.load_conversation("conversation.json")
  
  # From Path object
  from pathlib import Path
  conversation = llm.load_conversation(Path("conversation.json"))
  
  # From stream
  with open("conversation.json", "r") as f:
      conversation = llm.load_conversation(f)
  
  # From dictionary
  conversation = llm.load_conversation({"messages": [...], "version": "1.0"})
  ```

### LLMConversation

The mutable execution context that owns the message history and handles API calls.

#### Properties

- **`.messages: MessageList`**: The conversation's message history (read-only access)
  ```python
  print(f"Conversation has {len(conversation.messages)} messages")
  ```

- **`.continuation: LLMPromptBuilder`**: Get a builder that references this conversation
  ```python
  follow_up = conversation.continuation.request("Tell me more")
  ```

#### Methods

- **`.save(destination: str | Path | IO) -> None`**: Save conversation to file or stream
  ```python
  # Save to file path
  conversation.save("conversation.json")
  
  # Save to Path object
  from pathlib import Path
  conversation.save(Path("conversation.json"))
  
  # Save to stream
  with open("conversation.json", "w") as f:
      conversation.save(f)
  ```

- **`.apply_config_deltas(delta_config: dict) -> None`**: Apply configuration changes (typically called internally)
  ```python
  conversation.apply_config_deltas({"preferred_provider": "anthropic"})
  ```

#### Async Iteration

Conversations implement the async iterator protocol:

```python
conversation = llm.request("Count to 3").prompt_conversation()

async for message in conversation:
    print(message.text)
    # Prints each response chunk or final message
```

### MessageList

The mutable data container that holds conversation messages and handles serialization.

#### Methods

- **`.append(message: Message) -> None`**: Add a message to the list
  ```python
  from fluent_llm import TextMessage, Role
  conversation.messages.append(TextMessage("Hello", Role.USER))
  ```

- **`.extend(messages: list[Message]) -> None`**: Add multiple messages
  ```python
  conversation.messages.extend([msg1, msg2, msg3])
  ```

- **`.copy() -> MessageList`**: Create a copy of the message list
  ```python
  backup = conversation.messages.copy()
  ```

- **`.to_dict() -> dict`**: Serialize to dictionary
  ```python
  data = conversation.messages.to_dict()
  import json
  json.dump(data, open("messages.json", "w"))
  ```

- **`.from_dict(data: dict) -> MessageList`**: Deserialize from dictionary (class method)
  ```python
  import json
  data = json.load(open("messages.json"))
  messages = MessageList.from_dict(data)
  ```

#### Iteration and Access

MessageList supports standard Python sequence operations:

```python
# Length
print(len(conversation.messages))

# Iteration
for message in conversation.messages:
    print(message.text)

# Indexing
first_message = conversation.messages[0]
last_message = conversation.messages[-1]
```

## Migration Guide

If you're upgrading from an earlier version of Fluent LLM, here are the key changes:

### Architecture Changes

**Old Architecture:**
- `ConversationGenerator` and `ConversationState` classes
- State duplication between builder and conversation
- Manual conversation state management

**New Architecture:**
- Three-class architecture: `MessageList`, `LLMConversation`, `LLMPromptBuilder`
- Single source of truth for messages (in `MessageList`)
- Delta pattern for immutable builders
- Automatic conversation reference in continuations

### Method Changes

#### Starting Conversations

**Before:**
```python
conversation = llm.request("Hello").start_conversation()
```

**After:**
```python
conversation = llm.request("Hello").prompt_conversation()
```

#### Accessing Continuations

**Before:**
```python
continuation_builder = conversation.llm_continuation
```

**After:**
```python
continuation_builder = conversation.continuation
```

#### Serialization

**Before:**
```python
# Manual JSON handling required
import json
data = {
    "messages": [msg.to_dict() for msg in conversation.messages],
    "config": conversation.config
}
json.dump(data, open("conv.json", "w"))
```

**After:**
```python
# Built-in convenience methods
conversation.save("conv.json")

# Or load
conversation = llm.load_conversation("conv.json")
```

### Assistant Message Injection

**New Feature:**
```python
# Now you can inject assistant messages for few-shot examples
response = llm \
    .agent("You are a translator.") \
    .request("Translate 'hello' to French") \
    .assistant("Bonjour") \
    .request("Translate 'goodbye' to French") \
    .prompt()
```

### Configuration Changes

**Before:**
```python
# Configuration was part of serialization
```

**After:**
```python
# Configuration is separate from serialization
# This allows model-agnostic conversation restoration
conversation = llm.load_conversation("conv.json")
continuation = conversation.continuation \
    .provider("anthropic") \
    .model("claude-sonnet-4-20250514") \
    .request("Continue") \
    .prompt_conversation()
```

### Breaking Changes

1. **Removed Classes:**
   - `ConversationGenerator` → Use `LLMConversation` with async iteration
   - `ConversationState` → State is now managed by `LLMConversation`

2. **Method Renames:**
   - `.start_conversation()` → `.prompt_conversation()`
   - `.llm_continuation` → `.continuation`

3. **Serialization Format:**
   - Configuration is no longer included in serialized data
   - Only message history is serialized (model-agnostic)

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
