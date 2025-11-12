import typing

import openai
import pytest
from openai.types.responses.tool_param import ToolParam
from rich.console import Console

from str_message import Conversation, Message, UserMessage
from str_message.extra.mcps import aws_knowledge_mcp_param
from str_message.types.func_def import FuncDef
from str_message.utils.might_reasoning import might_reasoning
from str_message.utils.might_temperature import might_temperature

MODEL = "gpt-5-nano"

user_says: list[str] = [
    "Can you explain what is AWS S3 for me shortly?",
    "why grass is green?",
]


@pytest.mark.asyncio
async def test_oai(console: Console, func_defs: typing.Dict[str, FuncDef]):
    client = openai.AsyncOpenAI()

    conv = Conversation()

    for idx, user_say in enumerate[str](user_says):
        conv.add_message(UserMessage(content=user_say))

        input_messages = Message.to_response_input_param(conv.messages)
        console.print(f"[{idx}] input_messages:")
        console.print(input_messages)
        console.print("")

        tools: list[ToolParam] = [aws_knowledge_mcp_param]

        response = await client.responses.create(
            input=input_messages,
            model=MODEL,
            tools=tools,
            temperature=might_temperature(MODEL, 0.0),
            reasoning=might_reasoning(MODEL, "low"),
            timeout=10.0,
        )
        console.print(f"[{idx}] response:")
        console.print(response)
        console.print("")

        if response.usage:
            conv.add_usage(response.usage, model=MODEL, annotations=f"test_oai.{idx}")

        for item in response.output:
            conv.add_message(Message.from_any(item))

        conv.clean_messages()

    console.print(f"[{idx}] conversation:")
    console.print(conv)
    console.print("")
    console.print(f"total cost: {conv.total_cost}")

    return None  # test done
