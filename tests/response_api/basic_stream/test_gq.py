import os

import openai
import pytest
from rich.console import Console

from str_message import Conversation, Message, SystemMessage, UserMessage
from str_message.utils.might_reasoning import might_reasoning_param
from str_message.utils.might_temperature import might_temperature

MODEL = "openai/gpt-oss-120b"

user_says: list[str] = [
    "why grass is green?",
    "hi",
]


@pytest.mark.asyncio
async def test_gq(console: Console):
    client = openai.AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    conv = Conversation()
    conv.add_message(SystemMessage(content="You are a taciturn assistant."))

    for idx, user_say in enumerate[str](user_says):
        conv.add_message(UserMessage(content=user_say))

        input_messages = conv.response_input_param
        console.print(f"[{idx}] input_messages:")
        console.print(input_messages)
        console.print("")

        stream_manager = client.responses.stream(
            input=input_messages,
            model=MODEL,
            temperature=might_temperature(MODEL, 0.0),
            reasoning=might_reasoning_param(MODEL, "low"),
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

        for item in response.output:
            conv.add_message(Message.from_any(item))

        if response.usage:
            conv.add_usage(response.usage, model=MODEL, annotations=f"test_oai.{idx}")

        console.print(f"[{idx}] conversation:")
        console.print(conv)
        console.print("")

    console.print(f"total cost: {conv.total_cost}")

    return None  # test done
