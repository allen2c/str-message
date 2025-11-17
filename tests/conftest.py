# tests/conftest.py
import logging
import pathlib
import textwrap
import typing

import logging_bullet_train as lbt
import openai
import pytest
import requests
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


@pytest.fixture(scope="module")
def sample_audio() -> pathlib.Path:
    sample_audio_path = pathlib.Path("tests/data/sample_audio.wav")

    if sample_audio_path.is_file():
        return sample_audio_path

    sample_audio_path.parent.mkdir(parents=True, exist_ok=True)

    openai_client = openai.OpenAI()
    response = openai_client.audio.speech.create(
        model="gpt-4o-mini-tts",
        input="Hello, my name is Coral.",
        voice="coral",
        instructions=textwrap.dedent(
            """
            Affect/personality: A cheerful guide

            Tone: Friendly, clear, and reassuring, creating a calm atmosphere and making the listener feel confident and comfortable.

            Pronunciation: Clear, articulate, and steady, ensuring each instruction is easily understood while maintaining a natural, conversational flow.

            Pause: Brief, purposeful pauses after key instructions (e.g., "cross the street" and "turn right") to allow time for the listener to process the information and follow along.

            Emotion: Warm and supportive, conveying empathy and care, ensuring the listener feels guided and safe throughout the journey.
            """  # noqa: E501
        ).strip(),
        response_format="wav",
    )
    sample_audio_path.write_bytes(response.content)
    return sample_audio_path


@pytest.fixture(scope="module")
def sample_image() -> pathlib.Path:
    sample_image_path = pathlib.Path("tests/data/sample_image.jpg")

    if sample_image_path.is_file():
        return sample_image_path

    sample_image_path.parent.mkdir(parents=True, exist_ok=True)

    url = "https://picsum.photos/200/300"
    response = requests.get(url)
    response.raise_for_status()
    sample_image_path.write_bytes(response.content)
    return sample_image_path
