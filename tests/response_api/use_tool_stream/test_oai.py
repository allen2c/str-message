import typing

import openai
import pytest
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from rich.console import Console

from str_message import Conversation, Message, ToolCallOutputMessage, UserMessage
from str_message.types.func_def import FuncDef
from str_message.utils.might_reasoning import might_reasoning
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
        tool_calls: typing.List[ResponseFunctionToolCall] = []

        while is_user_last_message or len(tool_calls) > 0:

            # Tool handling
            while tool_call := safe_pop(tool_calls):
                assert tool_call.type == "function_call"
                assert tool_call.name in func_defs
                result = await func_defs[tool_call.name].callable(tool_call.arguments)
                conv.add_message(
                    ToolCallOutputMessage(
                        content=result,
                        tool_call_id=tool_call.call_id,
                        tool_name=tool_call.name,
                        tool_call_arguments=tool_call.arguments,
                    )
                )

            input_messages = Message.to_response_input_param(conv.messages)
            console.print(f"[{idx}] input_messages:")
            console.print(input_messages)
            console.print("")

            stream_manager = client.responses.stream(
                input=input_messages,
                model=MODEL,
                tools=[f.response_tool_param for f in func_defs.values()],
                temperature=might_temperature(MODEL, 0.0),
                reasoning=might_reasoning(MODEL, "low"),
                timeout=10.0,
            )
            async with stream_manager as stream:
                async for chunk in stream:
                    if chunk.type == "response.output_text.delta":
                        console.print(chunk.delta, style="hot_pink", end="")

            response = await stream.get_final_response()
            console.print(f"[{idx}] response:")
            console.print(response)
            console.print("")

            if response.usage:
                conv.add_usage(
                    response.usage, model=MODEL, annotations=f"test_oai.{idx}"
                )

            for item in response.output:
                conv.add_message(Message.from_any(item))

                if item.type == "function_call":
                    tool_calls.append(item)

            is_user_last_message = False

            conv.clean_messages()

    console.print(f"[{idx}] conversation:")
    console.print(conv)
    console.print("")
    console.print(f"total cost: {conv.total_cost}")

    return None  # test done
