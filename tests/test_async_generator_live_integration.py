"""Live integration test for async generator conversation functionality.

This test demonstrates the exact usage pattern from example.py and verifies
that the async generator approach works correctly with real tool calling.
"""

import pytest
import os
import tempfile
from pathlib import Path
from fluent_llm import llm
from fluent_llm.messages import TextMessage, ToolCallMessage
from fluent_llm.builder import LLMPromptBuilder


class TestAsyncGeneratorLiveIntegration:
    """Live integration tests for async generator conversations."""

    @pytest.mark.asyncio
    async def test_manual_iteration_pattern_from_example(self):
        """Test the exact usage pattern from example.py with manual iteration."""
        
        # Create temporary test files
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file1 = Path(temp_dir) / "test_notes.txt"
            test_file2 = Path(temp_dir) / "characters.txt"
            
            test_file1.write_text("This is a story about adventure. See also: characters.txt")
            test_file2.write_text("Main character: Hero\nVillain: Dark Lord")
            
            def read_file(filename: str) -> str:
                """Reads a file and returns its contents as a string."""
                try:
                    with open(filename, 'r') as f:
                        return f.read()
                except FileNotFoundError:
                    return f"File not found: {filename}"
                except Exception as e:
                    return f"Error reading file: {str(e)}"
            
            # Test the async generator pattern
            conversation = llm \
                .agent('You are a helpful assistant with access to the local filesystem.') \
                .tool(read_file) \
                .request(f'Read the file "{test_file1}" and analyze its contents.') \
                .prompt_conversation()
            
            messages = []
            continuation_builder = None
            
            # Manual iteration using __anext__()
            try:
                while True:
                    message = await conversation.__anext__()
                    messages.append(message)
                    
                    # Stop after getting a few messages to avoid infinite loops in test
                    if len(messages) >= 3:
                        break
            except StopAsyncIteration:
                # Generator completed naturally
                pass
            
            # Access continuation builder from the conversation generator
            continuation_builder = conversation.llm_continuation
            
            # Verify we got expected message types
            assert len(messages) >= 1
            
            # Should have at least one tool call message
            tool_messages = [msg for msg in messages if isinstance(msg, ToolCallMessage)]
            assert len(tool_messages) >= 1
            
            # Verify tool call structure
            first_tool_call = tool_messages[0]
            assert first_tool_call.tool_name == "read_file"
            assert str(test_file1) in str(first_tool_call.arguments.get("filename", ""))
            assert first_tool_call.result is not None or first_tool_call.error is not None
            
            # Continuation builder should be available after completion
            if continuation_builder:
                assert isinstance(continuation_builder, LLMPromptBuilder)

    @pytest.mark.asyncio
    async def test_async_for_loop_pattern(self):
        """Test async for loop iteration pattern."""
        
        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return f"Weather in {location}: Sunny, 72°F"
        
        def get_time() -> str:
            """Get current time."""
            return "Current time: 2:30 PM"
        
        conversation = llm \
            .agent('You are a helpful assistant.') \
            .tools([get_weather, get_time]) \
            .request('What is the weather in Paris and what time is it?') \
            .prompt_conversation()
        
        messages = []
        async for message in conversation:
            messages.append(message)
            # Limit iterations to avoid infinite loops in test
            if len(messages) >= 5:
                break
        
        # Verify we got expected message types
        assert len(messages) >= 1
        
        # Should have tool call messages
        tool_messages = [msg for msg in messages if isinstance(msg, ToolCallMessage)]
        assert len(tool_messages) >= 1
        
        # Verify continuation builder is available after async for completion
        continuation_builder = conversation.llm_continuation
        if continuation_builder:
            assert isinstance(continuation_builder, LLMPromptBuilder)

    @pytest.mark.asyncio
    async def test_prompt_agentically_compatibility_with_continuation(self):
        """Test prompt_agentically compatibility and continuation builder functionality."""
        
        def calculate(operation: str, x: float, y: float) -> float:
            """Perform basic mathematical operations."""
            if operation == "add":
                return x + y
            elif operation == "multiply":
                return x * y
            elif operation == "divide":
                return x / y if y != 0 else float('inf')
            else:
                raise ValueError(f"Unknown operation: {operation}")
        
        # First conversation using prompt_agentically
        messages, continuation = await llm \
            .agent('You are a calculator assistant.') \
            .tool(calculate) \
            .request('Calculate 5 + 3 and then multiply the result by 2') \
            .prompt_agentically(max_calls=5)
        
        # Verify we got messages and continuation
        assert len(messages) >= 1
        assert isinstance(continuation, LLMPromptBuilder)
        
        # Should have tool call messages
        tool_messages = [msg for msg in messages if isinstance(msg, ToolCallMessage)]
        assert len(tool_messages) >= 1
        
        # Test continuation with follow-up
        follow_up_messages, final_continuation = await continuation \
            .request("Now divide that result by 4") \
            .prompt_agentically(max_calls=3)
        
        assert len(follow_up_messages) >= 1
        assert isinstance(final_continuation, LLMPromptBuilder)

    @pytest.mark.asyncio
    async def test_tool_error_handling_in_async_generator(self):
        """Test that tool errors are properly handled in the async generator."""
        
        def failing_tool(should_fail: bool = True) -> str:
            """A tool that can be made to fail."""
            if should_fail:
                raise ValueError("This tool was designed to fail")
            return "Success!"
        
        conversation = llm \
            .agent('You are a test assistant.') \
            .tool(failing_tool) \
            .request('Use the failing tool with should_fail=True') \
            .prompt_conversation()
        
        messages = []
        async for message in conversation:
            messages.append(message)
            # Limit to avoid infinite loops
            if len(messages) >= 3:
                break
        
        # Should have at least one message
        assert len(messages) >= 1
        
        # Find tool call message with error
        tool_messages = [msg for msg in messages if isinstance(msg, ToolCallMessage)]
        assert len(tool_messages) >= 1
        
        # At least one should have an error
        error_messages = [msg for msg in tool_messages if msg.error is not None]
        assert len(error_messages) >= 1
        
        error_msg = error_messages[0]
        assert isinstance(error_msg.error, ValueError)
        assert "designed to fail" in str(error_msg.error)



    def test_conversation_method_signature_validation(self):
        """Test that prompt_conversation method has correct signature (no message parameter)."""
        
        import inspect
        
        # Get the method signature
        sig = inspect.signature(llm.prompt_conversation)
        
        # Should not have a 'message' parameter
        assert 'message' not in sig.parameters
        
        # Should have **kwargs for additional arguments
        param_names = list(sig.parameters.keys())
        assert 'kwargs' in param_names or any(
            param.kind == param.VAR_KEYWORD for param in sig.parameters.values()
        )
        
        # Return type should be AsyncGenerator (this is harder to test at runtime)
        # but we can at least verify the method exists and is callable
        assert callable(llm.prompt_conversation)