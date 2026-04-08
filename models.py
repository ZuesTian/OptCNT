"""Compatibility alias for the data model module now located under src.core."""

import importlib as _importlib
import sys as _sys

_impl = _importlib.import_module("src.core.models")
_sys.modules[__name__] = _impl
