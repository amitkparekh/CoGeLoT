from collections.abc import Mapping, Sequence
from string import Formatter
from typing import Any


class DefaultFormatter(Formatter):
    """Default to the key if there is no value."""

    def get_value(self, key: str | int, args: Sequence[Any], kwds: Mapping[str, Any]) -> str:
        """Get the value of the key."""
        if isinstance(key, str):
            try:
                return kwds[key]
            except KeyError:
                return f"{{{key}}}"

        return super().get_value(key, args, kwds)


class TemplateFormatter(DefaultFormatter):
    """Format a prompt template."""

    def format(self, format_string: str, *args: Any, **kwds: Any) -> str:
        """Format the string."""
        formatted_string = super().format(format_string, *args, **kwds)
        formatted_string = formatted_string.strip()
        formatted_string = formatted_string.capitalize()
        if not formatted_string.endswith("."):
            formatted_string += "."  # noqa: WPS336
        return formatted_string
