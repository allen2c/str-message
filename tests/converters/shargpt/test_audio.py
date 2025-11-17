import pathlib

import durl
import openai
from rich.console import Console
from rich.pretty import pretty_repr
from rich.text import Text as RichText

from str_message import Conversation, Message, UserMessage
from str_message.utils.might_reasoning import might_reasoning_effort
from str_message.utils.might_temperature import might_temperature

MODEL = "gpt-4o-mini-audio-preview"

user_says: list[str] = [
    "<|audio|>",
    "What did I say?",
]


def test_audio(console: Console, sample_audio: pathlib.Path):
    client = openai.OpenAI()

    conv = Conversation()

    for idx, user_say in enumerate[str](user_says):
        if user_say == "<|audio|>":
            user_message = UserMessage(content=" ").add_audio(
                sample_audio.read_bytes(), durl.MIMEType.WAVEFORM_AUDIO_FORMAT
            )
        else:
            user_message = UserMessage(content=user_say)

        conv.add_message(user_message)

        input_messages = Message.to_chat_cmpl_input_messages(conv.messages)
        console.print(f"[{idx}] input_messages:")
        console.print(pretty_repr(input_messages, max_string=300))
        console.print("")

        response = client.chat.completions.create(
            model=MODEL,
            messages=input_messages,
            temperature=might_temperature(MODEL, 0.0),
            reasoning_effort=might_reasoning_effort(MODEL, "low"),
            timeout=10.0,
        )
        console.print(f"[{idx}] response:")
        console.print(pretty_repr(response, max_string=300))
        console.print("")

        conv.add_message(Message.from_any(response))
        if response.usage:
            conv.add_usage(response.usage, model=MODEL, annotations=f"test_oai.{idx}")

        console.print(f"[{idx}] conversation:")
        console.print(pretty_repr(conv, max_string=300))
        console.print("")

        conv.clean_messages()

    console.print(f"total cost: {conv.total_cost}\n")

    console.print(RichText("[final] sharegpt:"))
    console.print(Message.to_sharegpt(conv.messages))
    console.print("")

    return None  # test done
