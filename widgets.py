"""Compatibility alias for the GUI widgets module now located under src.gui."""

import importlib as _importlib
import sys as _sys

_impl = _importlib.import_module("src.gui.widgets")
_sys.modules[__name__] = _impl
