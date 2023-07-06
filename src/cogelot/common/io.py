import gzip
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import dill as pickle
import orjson


ORJSON_OPTIONS = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_APPEND_NEWLINE


def orjson_dumps(v: Any, *, default: Any) -> str:
    """Convert Model to JSON string.

    orjson.dumps returns bytes, to match standard json.dumps we need to decode.
    """
    return orjson.dumps(
        v,
        default=default,
        option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_APPEND_NEWLINE,
    ).decode()


def _save(
    save_func: Callable[[Any], bytes],
    data: Any,  # noqa: WPS110
    path: Path,
    *,
    compress: bool = False,
) -> Path:
    """Generic function for saving data."""
    data_bytes = save_func(data)
    if compress:
        data_bytes = gzip.compress(data_bytes)
        # Add .gz suffix if not present
        if not path.suffix.endswith("gz"):
            path = path.with_suffix(f"{path.suffix}.gz")

    path.write_bytes(data_bytes)
    return path


def _load(
    load_func: Callable[[bytes], Any],
    path: Path,
) -> Any:
    """Generic function for loading data."""
    data_bytes = path.read_bytes()
    if path.suffix.endswith("gz"):
        data_bytes = gzip.decompress(data_bytes)
    return load_func(data_bytes)


def save_json(data: Any, path: Path, *, compress: bool = False) -> Path:  # noqa: WPS110
    """Save the json to the path."""
    dump_fn = partial(orjson.dumps, option=ORJSON_OPTIONS)
    return _save(dump_fn, data, path, compress=compress)


def load_json(path: Path) -> Any:
    """Load the data from the path."""
    return _load(orjson.loads, path)


def save_pickle(data: Any, path: Path, *, compress: bool = False) -> Path:  # noqa: WPS110
    """Save the pickle to the path."""
    return _save(pickle.dumps, data, path, compress=compress)


def load_pickle(path: Path) -> Any:
    """Load the data from the path."""
    return _load(pickle.loads, path)
