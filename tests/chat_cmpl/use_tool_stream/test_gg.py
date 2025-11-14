import os
import typing

import openai
import pytest
from openai.types.chat.parsed_function_tool_call import ParsedFunctionToolCall
from rich.console import Console

from str_message import (
    Conversation,
    Message,
    SystemMessage,
    ToolCallOutputMessage,
    UserMessage,
)
from str_message.types.func_def import FuncDef
from str_message.utils.might_reasoning import might_reasoning_effort
from str_message.utils.might_temperature import might_temperature
from str_message.utils.safe_pop import safe_pop

MODEL = "gemini-2.5-flash"

user_says: list[str] = [
    "what time in Tokyo now?",
    "why grass is green?",
]


@pytest.mark.asyncio
async def test_gg(console: Console, func_defs: typing.Dict[str, FuncDef]):
    client = openai.AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    conv = Conversation()
    conv.add_message(SystemMessage(content="You are a taciturn assistant."))
    tools = [f.chat_cmpl_tool_param for f in func_defs.values()]
    console.print("[_] tools:")
    console.print(tools)
    console.print("")

    for idx, user_say in enumerate[str](user_says):
        conv.add_message(UserMessage(content=user_say))

        is_user_last_message: bool = True
        tool_calls: typing.List[ParsedFunctionToolCall] = []

        while is_user_last_message or len(tool_calls) > 0:
            while tool_call := safe_pop(tool_calls):
                assert tool_call.type == "function"
                assert tool_call.function.name in func_defs
                result = await func_defs[tool_call.function.name].callable(
                    tool_call.function.arguments
                )
                conv.add_message(
                    ToolCallOutputMessage(
                        content=result,
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.function.name,
                        tool_call_arguments=tool_call.function.arguments,
                    )
                )

            input_messages = Message.to_chat_cmpl_input_messages(conv.messages)
            console.print(f"[{idx}] input_messages:")
            console.print(input_messages)
            console.print("")

            async with client.chat.completions.stream(
                model=MODEL,
                messages=input_messages,
                tools=tools,
                temperature=might_temperature(MODEL, 0.0),
                reasoning_effort=might_reasoning_effort(MODEL, "low"),
                timeout=10.0,
            ) as stream:
                counter: int = 0
                async for chunk in stream:
                    if chunk.type == "content.delta":
                        console.print(chunk.delta, style="hot_pink", end="")
                    elif chunk.type == "tool_calls.function.arguments.delta":
                        console.print(chunk.arguments_delta, style="hot_pink", end="")
                    counter += 1
            console.print(f"\n[{idx}] total chunks: {counter}")

            response = await stream.get_final_completion()
            console.print(f"[{idx}] response:")
            console.print(response)
            console.print("")

            if response.usage:
                conv.add_usage(
                    response.usage, model=MODEL, annotations=f"test_gg.{idx}"
                )

            conv.add_message(Message.from_any(response))

            # Dialogue control
            tool_calls = response.choices[0].message.tool_calls or []
            is_user_last_message = False

    console.print(f"[{idx}] conversation:")
    console.print(conv)
    console.print("")
    console.print(f"total cost: {conv.total_cost}")

    return None  # test done
