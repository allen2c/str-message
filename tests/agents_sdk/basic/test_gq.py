import os

import agents
import openai
import pytest
from rich.console import Console

from str_message import Conversation, Message, UserMessage
from str_message.utils.might_reasoning import might_reasoning
from str_message.utils.might_temperature import might_temperature

MODEL = "openai/gpt-oss-120b"

user_says: list[str] = [
    "why grass is green?",
    "hi",
]


@pytest.mark.asyncio
async def test_gq(console: Console):
    openai_client = openai.AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    model_object = agents.OpenAIChatCompletionsModel(
        model=MODEL,
        openai_client=openai_client,
    )

    agent = agents.Agent[dict](
        "test_oai",
        model=model_object,
        model_settings=agents.ModelSettings(
            temperature=might_temperature(MODEL, 0.0, default=None),
            reasoning=might_reasoning(MODEL, "low", default=None),
        ),
        instructions="You are a taciturn assistant.",
    )

    conv = Conversation()

    for idx, user_say in enumerate[str](user_says):
        conv.add_message(UserMessage(content=user_say))

        input_messages = Message.to_response_input_param(conv.messages)
        console.print(f"[{idx}] input_messages:")
        console.print(input_messages)
        console.print("")

        run_result = await agents.run.Runner().run(
            agent,
            input_messages,
            context={},
            run_config=agents.RunConfig(tracing_disabled=True),
        )

        response = run_result.final_output
        console.print(f"[{idx}] response:")
        console.print(response)
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
        console.print(conv)
        console.print("")

    console.print(f"total cost: {conv.total_cost}")

    return None  # test done
