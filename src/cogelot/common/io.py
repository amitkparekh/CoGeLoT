from typing import Any

import orjson


def orjson_dumps(v: Any, *, default: Any) -> str:
    """Convert Model to JSON string.

    orjson.dumps returns bytes, to match standard json.dumps we need to decode.
    """
    return orjson.dumps(
        v,
        default=default,
        option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_APPEND_NEWLINE,
    ).decode()
