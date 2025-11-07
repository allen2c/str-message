import openai

from str_message import Message, MessageTypes, UserMessage
from str_message.utils.might_reasoning import might_reasoning_effort
from str_message.utils.might_temperature import might_temperature

MODEL = "gpt-5-mini"

user_says: list[str] = [
    "hi",
    "why grass is green?",
]


def test_oai():
    client = openai.OpenAI()

    messages: list[MessageTypes] = []

    for user_say in user_says:
        messages.append(UserMessage(content=user_say))

        response = client.chat.completions.create(
            model=MODEL,
            messages=Message.to_chat_cmpl_input_messages(messages),
            temperature=might_temperature(MODEL, 0.0),
            reasoning_effort=might_reasoning_effort(MODEL, "low"),
        )

        messages.append(Message.from_any(response))

    return None  # test done
