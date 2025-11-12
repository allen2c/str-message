import openai
from rich.console import Console

from str_message import Conversation, Message, UserMessage
from str_message.utils.might_reasoning import might_reasoning
from str_message.utils.might_temperature import might_temperature

MODEL = "gpt-5-nano"

user_says: list[str] = [
    "why grass is green?",
    "hi",
]


def test_oai(console: Console):
    client = openai.OpenAI()

    conv = Conversation()

    for idx, user_say in enumerate[str](user_says):
        conv.add_message(UserMessage(content=user_say))

        input_messages = conv.response_input_param
        console.print(f"[{idx}] input_messages:")
        console.print(input_messages)
        console.print("")

        response = client.responses.create(
            input=input_messages,
            model=MODEL,
            temperature=might_temperature(MODEL, 0.0),
            reasoning=might_reasoning(MODEL, "low"),
            timeout=10.0,
        )
        console.print(f"[{idx}] response:")
        console.print(response)
        console.print("")

        for m in response.output:
            conv.add_message(Message.from_any(m))

        if response.usage:
            conv.add_usage(response.usage, model=MODEL, annotations=f"test_oai.{idx}")

        console.print(f"[{idx}] conversation:")
        console.print(conv)
        console.print("")

    console.print(f"total cost: {conv.total_cost}")

    return None  # test done
