# str-message

Any Message Could be a String, for LLM Usage.

A universal message format that seamlessly works with OpenAI's Chat Completion API, Response API, and Agents SDK. Supports multi-turn conversations, tools, and MCP (Model Context Protocol).

## Compatibility Matrix

| Feature                     | OpenAI | Gemini | Anthropic | Groq |
|-----------------------------|--------|--------|-----------|------|
| ChatCompletion/Basic        | ✅      | ✅      | ✅         | ✅    |
| ChatCompletion/Basic Stream | ✅      | ✅      | ✅         | ✅    |
| ChatCompletion/Tool         | ✅      | ✅      | ✅         | ✅    |
| ChatCompletion/Tool Stream  | ✅      | ✅      | ✅         | ✅    |
| ChatCompletion/Audio        | ✅      | ❌      | ❌         | ❌    |
| ChatCompletion/Image        | ✅      | ❌      | ❌         | ❌    |
| Response/Basic              | ✅      | ❌      | ❌         | ✅    |
| Response/Basic Stream       | ✅      | ❌      | ❌         | ✅    |
| Response/Tool               | ✅      | ✅      | ✅         | ❌    |
| Response/Tool Stream        | ✅      | ✅      | ✅         | ❌    |
| Response/MCP                | ✅      | ❌      | ❌         | ❌    |
| Response/MCP Stream         | ✅      | ❌      | ❌         | ❌    |
| Response/Image              | ✅      | ❌      | ❌         | ❌    |
| Agents/Basic                | ✅      | ✅      | ✅         | ✅    |
| Agents/Basic Stream         | ✅      | ✅      | ✅         | ✅    |
| Agents/Tool                 | ✅      | ✅      | ✅         | ✅    |
| Agents/Tool Stream          | ✅      | ✅      | ✅         | ✅    |
| Agents/MCP                  | ✅      | ❌      | ❌         | ✅    |
| Agents/MCP Stream           | ✅      | ❌      | ❌         | ✅    |
| Agents/Image                | ✅      | ❌      | ❌         | ❌    |

## Installation

```bash
pip install str-message
```

## Quick Start

```python
import agents
import openai
from str_message import Conversation, Message, UserMessage
from str_message.extra.func_defs import func_def_get_current_time
from str_message.extra.mcps import aws_knowledge_mcp_tool
from str_message.utils.might_reasoning import might_reasoning
from str_message.utils.might_temperature import might_temperature

MODEL = "gpt-5-mini"

# Create agent with custom tool and MCP
agent = agents.Agent(
    "assistant",
    model=agents.OpenAIResponsesModel(
        model=MODEL,
        openai_client=openai.AsyncOpenAI(),
    ),
    model_settings=agents.ModelSettings(
        temperature=might_temperature(MODEL, 0.0, default=None),
        reasoning=might_reasoning(MODEL, "low", default=None),
    ),
    instructions="You are a helpful assistant.",
    tools=[
        func_def_get_current_time().agents_tool,  # Custom tool
        aws_knowledge_mcp_tool,  # MCP tool
    ],
)

conv = Conversation()

# 4-turn conversation demonstrating reasoning, tool, MCP, and regular chat
user_says = [
    "Solve this: If I have 3 apples and buy 2 more, then give 1 away, how many do I have?",  # Reasoning
    "What time is it in Tokyo?",  # Custom tool
    "What is AWS S3?",  # MCP
    "Thank you!",  # Regular chat
]

for user_say in user_says:
    conv.add_message(UserMessage(content=user_say))

    input_messages = Message.to_response_input_param(conv.messages)

    run_result = await agents.run.Runner().run(agent, input_messages, context={})

    # Update conversation with agent's response
    conv.messages[:] = [
        Message.from_any(item) for item in run_result.to_input_list()
    ]

    # Track usage and cost
    if usage := run_result.context_wrapper.usage:
        conv.add_usage(usage, model=MODEL)

    conv.clean_messages()

print(f"Total cost: ${conv.total_cost:.4f}")
```

## Features

* **Universal Format**: Single message format works across all OpenAI APIs
* **Multi-turn Conversations**: Built-in conversation management with `Conversation` class
* **Tool Support**: Native support for function calling and tool execution
* **MCP Compatible**: Built-in MCP (Model Context Protocol) integration
* **Usage Tracking**: Automatic token usage and cost tracking
* **Type Safe**: Full type annotations with Pydantic models

## Key Classes

* `Message`: Universal message container
* `Conversation`: Manages multi-turn conversations and usage tracking
* `UserMessage`, `AssistantMessage`, `ToolCallOutputMessage`: Typed message builders
* `FuncDef`: Type-safe function/tool definitions

## Links

* **GitHub**: <https://github.com/allen2c/str-message>
* **PyPI**: <https://pypi.org/project/str-message/>

## License

MIT
