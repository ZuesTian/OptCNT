"""Compatibility alias for the core analyzer module now located under src.core."""

import importlib as _importlib
import sys as _sys

_impl = _importlib.import_module("src.core.analyzer_core")
_sys.modules[__name__] = _impl
