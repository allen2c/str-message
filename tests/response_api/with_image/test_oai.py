import pathlib

import durl
import openai
from rich.console import Console
from rich.pretty import pretty_repr

from str_message import Conversation, Message, UserMessage
from str_message.utils.might_reasoning import might_reasoning_param
from str_message.utils.might_temperature import might_temperature

MODEL = "gpt-5-nano"

user_says: list[str] = [
    "<|image|>",
    "hi",
]


def test_oai(console: Console, sample_image: pathlib.Path):
    client = openai.OpenAI()

    conv = Conversation()

    for idx, user_say in enumerate[str](user_says):
        if user_say == "<|image|>":
            user_message = UserMessage(content="What is this image?").add_image(
                sample_image.read_bytes(), durl.MIMEType.JPEG_IMAGES
            )
        else:
            user_message = UserMessage(content=user_say)

        conv.add_message(user_message)

        input_messages = conv.response_input_param
        console.print(f"[{idx}] input_messages:")
        console.print(pretty_repr(input_messages, max_string=300))
        console.print("")

        response = client.responses.create(
            input=input_messages,
            model=MODEL,
            temperature=might_temperature(MODEL, 0.0),
            reasoning=might_reasoning_param(MODEL, "low"),
            timeout=10.0,
        )
        console.print(f"[{idx}] response:")
        console.print(pretty_repr(response, max_string=300))
        console.print("")

        for m in response.output:
            conv.add_message(Message.from_any(m))

        if response.usage:
            conv.add_usage(response.usage, model=MODEL, annotations=f"test_oai.{idx}")

        console.print(f"[{idx}] conversation:")
        console.print(pretty_repr(conv, max_string=300))
        console.print("")

    console.print(f"total cost: {conv.total_cost}")

    return None  # test done
