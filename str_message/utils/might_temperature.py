import typing

from openai._types import omit

T = typing.TypeVar("T")


def might_temperature(
    model: str,
    temperature: float = 0.7,
    *,
    default: typing.Union[float, T] = omit,
) -> float | T:
    from openai_usage.extra.open_router import get_model

    might_model = get_model(model)
    if might_model is None:
        return default

    if "temperature" in might_model.supported_parameters:
        return temperature
    else:
        return default
