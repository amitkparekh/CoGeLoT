"""
Taken from https://github.com/vimalabs/VIMABench/blob/main/vimasim/__init__.py

Due to historical reason, the dataset was generated with package name "vimasim". To avoid package
not found error when loading pickled data, we add an alias here.
"""
# unimport: skip_file
# ruff: noqa

import sys

import vima_bench
from vima_bench import *  # noqa: F403


sys.modules["vimasim"] = vima_bench
