import os

import openai
from rich.console import Console

from str_message import Conversation, Message, UserMessage
from str_message.utils.might_reasoning import might_reasoning_effort
from str_message.utils.might_temperature import might_temperature

MODEL = "gemini-2.5-flash"

user_says: list[str] = [
    "why grass is green?",
    "thank you",
]


def test_gg(console: Console):
    client = openai.OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    conv = Conversation()

    for idx, user_say in enumerate[str](user_says):
        conv.add_message(UserMessage(content=user_say))

        input_messages = Message.to_chat_cmpl_input_messages(conv.messages)
        console.print(f"[{idx}] input_messages:")
        console.print(input_messages)
        console.print("")

        response = client.chat.completions.create(
            model=MODEL,
            messages=input_messages,
            temperature=might_temperature(MODEL, 0.0),
            reasoning_effort=might_reasoning_effort(MODEL, "low"),
            timeout=10.0,
        )
        console.print(f"[{idx}] response:")
        console.print(response)
        console.print("")

        conv.add_message(Message.from_any(response))
        if response.usage:
            conv.add_usage(response.usage, model=MODEL, annotations=f"test_gg.{idx}")

        console.print(f"[{idx}] conversation:")
        console.print(conv)
        console.print("")

    console.print(f"total cost: {conv.total_cost}")

    return None  # test done
