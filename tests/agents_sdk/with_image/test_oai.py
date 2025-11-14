import pathlib

import agents
import durl
import openai
import pytest
from rich.console import Console
from rich.pretty import pretty_repr

from str_message import Conversation, Message, UserMessage
from str_message.utils.might_reasoning import might_reasoning
from str_message.utils.might_temperature import might_temperature

MODEL = "gpt-5-nano"

user_says: list[str] = [
    "<|image|>",
    "hi",
]


@pytest.mark.asyncio
async def test_oai(console: Console, sample_image: pathlib.Path):
    agent = agents.Agent[dict](
        "test_oai",
        model=agents.OpenAIResponsesModel(
            model=MODEL,
            openai_client=openai.AsyncOpenAI(),
        ),
        model_settings=agents.ModelSettings(
            temperature=might_temperature(MODEL, 0.0, default=None),
            reasoning=might_reasoning(MODEL, "low", default=None),
        ),
        instructions="You are a taciturn assistant.",
    )

    conv = Conversation()

    for idx, user_say in enumerate[str](user_says):
        if user_say == "<|image|>":
            user_message = UserMessage(content="What is this image?").add_image(
                sample_image.read_bytes(), durl.MIMEType.JPEG_IMAGES
            )
        else:
            user_message = UserMessage(content=user_say)

        conv.add_message(user_message)

        input_messages = Message.to_response_input_param(conv.messages)
        console.print(f"[{idx}] input_messages:")
        console.print(pretty_repr(input_messages, max_string=300))
        console.print("")

        run_result = await agents.run.Runner().run(
            agent,
            input_messages,
            context={},
            run_config=agents.RunConfig(tracing_disabled=True),
        )

        response = run_result.final_output
        console.print(f"[{idx}] response:")
        console.print(pretty_repr(response, max_string=300))
        console.print("")

        console.print(f"[{idx}] run_result.to_input_list():")
        console.print(run_result.to_input_list())
        console.print("")

        conv.messages[:] = [
            m for item in run_result.to_input_list() for m in Message.from_any(item)
        ]

        if usage := run_result.context_wrapper.usage:
            conv.add_usage(usage, model=MODEL, annotations=f"test_oai.{idx}")

        console.print(f"[{idx}] conversation:")
        console.print(pretty_repr(conv, max_string=300))
        console.print("")

    console.print(f"total cost: {conv.total_cost}")

    return None  # test done
