import typing

import agents
import openai
import pytest
from rich.console import Console

from str_message import Conversation, Message, UserMessage
from str_message.types.func_def import FuncDef
from str_message.utils.might_reasoning import might_reasoning
from str_message.utils.might_temperature import might_temperature

MODEL = "gpt-5-nano"

user_says: list[str] = [
    "what time in Tokyo now?",
    "why grass is green?",
]


@pytest.mark.asyncio
async def test_oai(console: Console, func_defs: typing.Dict[str, FuncDef]):
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
        tools=[f.agents_tool for f in func_defs.values()],
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

        conv.messages[:] = [
            Message.from_any(item) for item in run_result.to_input_list()
        ]

        if usage := run_result.context_wrapper.usage:
            conv.add_usage(usage, model=MODEL, annotations=f"test_oai.{idx}")

        console.print(f"[{idx}] conversation:")
        console.print(conv)
        console.print("")

    console.print(f"total cost: {conv.total_cost}")

    return None  # test done
