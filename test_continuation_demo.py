"""Demo script showing that continuation now works correctly."""
import asyncio
from fluent_llm import llm

async def main():
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
    
    print("=== First conversation ===")
    conversation = llm \
        .agent('You are a calculator assistant.') \
        .tool(calculate) \
        .request('Calculate 5 + 3') \
        .prompt_conversation()
    
    messages = []
    async for message in conversation:
        print(f"Message: {message}")
        messages.append(message)
        if len(messages) >= 5:
            break
    
    print(f"\n=== Got {len(messages)} messages in first conversation ===\n")
    
    # Get continuation and follow up
    print("=== Continuation conversation ===")
    continuation = conversation.continuation
    follow_up = continuation.request("Now multiply that by 2").prompt_conversation()
    
    follow_up_messages = []
    async for message in follow_up:
        print(f"Follow-up message: {message}")
        follow_up_messages.append(message)
        if len(follow_up_messages) >= 3:
            break
    
    print(f"\n=== Got {len(follow_up_messages)} messages in continuation ===")
    
    if len(follow_up_messages) > 0:
        print("\n✅ SUCCESS: Continuation works!")
    else:
        print("\n❌ FAILURE: Continuation returned no messages")

if __name__ == "__main__":
    asyncio.run(main())
