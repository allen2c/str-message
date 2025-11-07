import typing

import openai
import pytest
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCallUnion,
)
from rich.console import Console

from str_message import Conversation, Message, ToolCallOutputMessage, UserMessage
from str_message.types.func_def import FuncDef
from str_message.utils.might_reasoning import might_reasoning_effort
from str_message.utils.might_temperature import might_temperature
from str_message.utils.safe_pop import safe_pop

MODEL = "gpt-5-nano"

user_says: list[str] = [
    "what time in Tokyo now?",
    "why grass is green?",
]


@pytest.mark.asyncio
async def test_oai(console: Console, func_defs: typing.Dict[str, FuncDef]):
    client = openai.AsyncOpenAI()

    conv = Conversation()

    for idx, user_say in enumerate[str](user_says):
        conv.add_message(UserMessage(content=user_say))

        is_user_last_message: bool = True
        tool_calls: typing.List[ChatCompletionMessageToolCallUnion] = []

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
            console.print()

            response = await client.chat.completions.create(
                model=MODEL,
                messages=input_messages,
                tools=[f.chat_cmpl_tool_param for f in func_defs.values()],
                temperature=might_temperature(MODEL, 0.0),
                reasoning_effort=might_reasoning_effort(MODEL, "low"),
                timeout=10.0,
            )
            console.print(f"[{idx}] response:")
            console.print(response)
            console.print()

            if response.usage:
                conv.add_usage(
                    response.usage, model=MODEL, annotations=f"test_oai.{idx}"
                )

            conv.add_message(Message.from_any(response))

            # Dialogue control
            tool_calls = response.choices[0].message.tool_calls or []
            is_user_last_message = False

    console.print(f"[{idx}] conversation:")
    console.print(conv)
    console.print()
    console.print(f"total cost: {conv.total_cost}")

    return None  # test done
