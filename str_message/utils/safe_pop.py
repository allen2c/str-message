import typing

T = typing.TypeVar("T")


def safe_pop(lst: typing.List[T] | None) -> typing.Optional[T]:
    if lst is None:
        return None
    try:
        return lst.pop()
    except IndexError:
        return None
