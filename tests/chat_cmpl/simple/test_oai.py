import openai
from rich.console import Console

from str_message import Message, MessageTypes, UserMessage
from str_message.utils.might_reasoning import might_reasoning_effort
from str_message.utils.might_temperature import might_temperature

MODEL = "gpt-5-mini"

user_says: list[str] = [
    "hi",
    "why grass is green?",
]


def test_oai(console: Console):
    client = openai.OpenAI()

    messages: list[MessageTypes] = []

    for idx, user_say in enumerate(user_says):
        messages.append(UserMessage(content=user_say))

        input_messages = Message.to_chat_cmpl_input_messages(messages)
        console.print(f"[{idx}] input_messages: {input_messages}")

        response = client.chat.completions.create(
            model=MODEL,
            messages=input_messages,
            temperature=might_temperature(MODEL, 0.0),
            reasoning_effort=might_reasoning_effort(MODEL, "low"),
        )
        console.print(f"[{idx}] response: {response}")

        messages.append(Message.from_any(response))

    return None  # test done
