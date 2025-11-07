# tests/conftest.py
import datetime
import typing
import zoneinfo

import pydantic
import pytest
import rich.console
from openai.types.shared.function_definition import FunctionDefinition
from str_or_none import str_or_none

import str_message.patches.patch_openai
from str_message.types.func_def import FuncDef

str_message.patches.patch_openai.patch_openai()


@pytest.fixture(scope="module")
def console():
    return rich.console.Console()


@pytest.fixture(scope="module")
def func_def_get_current_time() -> FuncDef:
    class Arguments(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="forbid")
        timezone: str = pydantic.Field(
            default="",
            description=(
                "The timezone to get the current time in. If not provided, "
                + "the current time in Asia/Taipei timezone will be returned."
            ),
        )

    async def get_current_time(arguments: Arguments | str) -> str:
        """Get the current time"""
        arguments = (
            Arguments.model_validate_json(arguments)
            if not isinstance(arguments, Arguments)
            else arguments
        )
        dt = datetime.datetime.now(
            zoneinfo.ZoneInfo(str_or_none(arguments.timezone) or "Asia/Taipei")
        )
        dt = dt.replace(microsecond=0)
        return dt.isoformat()

    func_def = FunctionDefinition(
        name="get_current_time",
        description="Get the current time of optional timezone.",
        parameters=Arguments.model_json_schema(),
    )

    return FuncDef(func_def, get_current_time, Arguments, context=None)


@pytest.fixture(scope="module")
def func_def_get_current_weather() -> FuncDef:
    class Arguments(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="forbid")
        city: str = pydantic.Field(
            description="The city to get the weather of.",
        )

    async def get_current_weather(arguments: Arguments | str) -> str:
        """Get the weather of the city"""
        arguments = (
            Arguments.model_validate_json(arguments)
            if not isinstance(arguments, Arguments)
            else arguments
        )

        return "The weather of the city is sunny now."

    func_def = FunctionDefinition(
        name="get_current_weather",
        description="Get the current weather of the city.",
        parameters=Arguments.model_json_schema(),
    )

    return FuncDef(func_def, get_current_weather, Arguments, context=None)


@pytest.fixture(scope="module")
def func_defs(
    func_def_get_current_time: FuncDef, func_def_get_current_weather: FuncDef
) -> typing.Dict[str, FuncDef]:
    return {
        "get_current_time": func_def_get_current_time,
        "get_current_weather": func_def_get_current_weather,
    }
