"""Compatibility alias for the GUI panels module now located under src.gui."""

import importlib as _importlib
import sys as _sys

_impl = _importlib.import_module("src.gui.panels")
_sys.modules[__name__] = _impl
