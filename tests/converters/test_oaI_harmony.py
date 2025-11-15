import typing

import agents
import openai
import pytest
from rich.console import Console
from rich.text import Text as RichText

from str_message import Conversation, Message, UserMessage
from str_message.extra.mcps import aws_knowledge_mcp_tool
from str_message.types.func_def import FuncDef
from str_message.utils.might_reasoning import might_reasoning
from str_message.utils.might_temperature import might_temperature

MODEL = "gpt-5-nano"

user_says: list[str] = [
    "why grass is green?",
    "what time in Tokyo now?",
    "Can you explain what is AWS S3 for me shortly?",
    "bye",
]


@pytest.mark.asyncio
async def test_oai_harmony(console: Console, func_defs: typing.Dict[str, FuncDef]):
    openai_client = openai.AsyncOpenAI()
    model_obj = agents.OpenAIResponsesModel(
        model=MODEL,
        openai_client=openai_client,
    )
    tools: typing.List[agents.Tool] = [f.agents_tool for f in func_defs.values()] + [
        aws_knowledge_mcp_tool
    ]  # type: ignore
    agent = agents.Agent[dict](
        "test_oai",
        model=model_obj,
        model_settings=agents.ModelSettings(
            temperature=might_temperature(MODEL, 0.0, default=None),
            reasoning=might_reasoning(MODEL, "low", default=None),
        ),
        instructions="You are a taciturn assistant.",
        tools=tools,
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
            conv.add_usage(usage, model=MODEL, annotations=f"test_oai_harmony.{idx}")

        console.print(f"[{idx}] conversation:")
        console.print(conv)
        console.print("")

        conv.clean_messages()

    console.print(f"total cost: {conv.total_cost}\n")

    console.print(RichText("[final] harmony:"))
    console.print(
        RichText(
            Message.to_harmony(
                conv.messages,
                [f.func_def for f in func_defs.values()],
            )
        )
    )
    console.print("")

    return None  # test done
