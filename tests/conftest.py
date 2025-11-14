# tests/conftest.py
import logging
import typing

import logging_bullet_train as lbt
import pytest
import rich.console

import str_message.patches.patch_openai
from str_message.types.func_def import FuncDef

lbt.set_logger("openai_usage", level=logging.DEBUG)
lbt.set_logger("str_message", level=logging.DEBUG)

str_message.patches.patch_openai.patch_openai()


@pytest.fixture(scope="module")
def console():
    return rich.console.Console()


@pytest.fixture(scope="module")
def func_defs() -> typing.Dict[str, FuncDef]:
    from str_message.extra.func_defs import (
        func_def_get_current_time,
        func_def_get_current_weather,
    )

    return {
        "get_current_time": func_def_get_current_time(),
        "get_current_weather": func_def_get_current_weather(),
    }
