import os
from pathlib import Path

import pytest

if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):  # type: ignore[no-untyped-def] # noqa: ANN201, ANN001
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):  # type: ignore[no-untyped-def] # noqa: ANN201, ANN001
        raise excinfo.value


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "-E",
        action="store",
        metavar="NAME",
        help="only run tests matching the environment NAME.",
    )


def pytest_configure(config: pytest.Config) -> None:
    # register an additional marker
    config.addinivalue_line("markers", "env(name): mark test to run only on named environment")


def pytest_runtest_setup(item: pytest.Item) -> None:
    envnames = [mark.args[0] for mark in item.iter_markers(name="env")]
    if envnames and item.config.getoption("-E") not in envnames:
        pytest.skip(f"test requires env in {envnames!r}")


# Import all the fixtures from every file in the tests/fixtures dir.
pytest_plugins = [
    fixture_file.as_posix().replace("/", ".").replace(".py", "")
    for fixture_file in Path().rglob("tests/fixtures/[!__]*.py")
]
