"""
Lean Windows packaging script for OptCNT.

Key goals:
- build a single-file GUI executable
- shrink size with UPX when available
- work best from a dedicated virtual environment
- drop optional heavyweight dependencies that already have safe fallbacks
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import PyInstaller.__main__


ROOT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = ROOT_DIR / ".tools"
UPX_DIR = TOOLS_DIR / "upx"
LOCAL_UPX_GLOB = "upx*"

EXCLUDE_MODULES = [
    "Cython",
    "IPython",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "Pygments",
    "_pytest",
    "cython",
    "idlelib",
    "jupyter",
    "lib2to3",
    "llvmlite",
    "matplotlib.backends.backend_gtk3",
    "matplotlib.backends.backend_gtk3agg",
    "matplotlib.backends.backend_gtk4",
    "matplotlib.backends.backend_gtk4agg",
    "matplotlib.backends.backend_macosx",
    "matplotlib.backends.backend_nbagg",
    "matplotlib.backends.backend_qt",
    "matplotlib.backends.backend_qt5",
    "matplotlib.backends.backend_qt5agg",
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_webagg",
    "matplotlib.backends.backend_webagg_core",
    "matplotlib.tests",
    "mpl_toolkits.tests",
    "notebook",
    "numba",
    "numpy.random._examples",
    "pandas",
    "pip",
    "pytest",
    "scipy",
    "setuptools",
    "sklearn",
    "skimage",
    "sphinx",
    "test",
    "threadpoolctl",
    "tkinter.test",
    "tkinter.tix",
    "tkinter.ttk.test",
    "turtle",
    "turtledemo",
    "venv",
    "wheel",
]

HIDDEN_IMPORTS = [
    "matplotlib.backends._backend_tk",
    "matplotlib.backends.backend_tkagg",
    "PIL._imagingtk",
    "PIL._tkinter_finder",
]

UPX_EXCLUDES = [
    "_uuid.pyd",
    "python3.dll",
]


def _print(message: str) -> None:
    print(message)


def running_in_virtualenv() -> bool:
    """Whether the current interpreter belongs to a virtual environment."""
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def clean_build_dirs() -> None:
    """Remove old packaging output."""
    for name in ("build", "dist", "__pycache__"):
        path = ROOT_DIR / name
        if path.exists():
            shutil.rmtree(path)
            _print(f"Removed {path}")


def _iter_upx_candidates() -> Iterable[Path]:
    env_upx_dir = os.environ.get("UPX_DIR")
    if env_upx_dir:
        yield Path(env_upx_dir) / "upx.exe"
        yield Path(env_upx_dir) / "upx"

    path_upx = shutil.which("upx")
    if path_upx:
        yield Path(path_upx)

    if UPX_DIR.exists():
        yield from UPX_DIR.rglob("upx.exe")
        yield from UPX_DIR.rglob("upx")

    for local_dir in ROOT_DIR.glob(LOCAL_UPX_GLOB):
        if local_dir.is_dir():
            yield from local_dir.rglob("upx.exe")
            yield from local_dir.rglob("upx")


def find_upx_executable() -> Optional[Path]:
    """Find an existing UPX executable."""
    for candidate in _iter_upx_candidates():
        if candidate.exists():
            return candidate.resolve()
    return None


def _download_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "OptCNT-build-script",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def download_upx() -> Path:
    """Download the latest UPX Windows x64 release into .tools/upx."""
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    UPX_DIR.mkdir(parents=True, exist_ok=True)

    api_url = "https://api.github.com/repos/upx/upx/releases/latest"
    release = json.loads(_download_bytes(api_url).decode("utf-8"))
    assets = release.get("assets", [])

    selected_asset = None
    for asset in assets:
        name = str(asset.get("name", "")).lower()
        if name.endswith("win64.zip"):
            selected_asset = asset
            break

    if not selected_asset:
        raise RuntimeError("Could not find a Windows x64 UPX asset in the latest release.")

    asset_url = selected_asset.get("browser_download_url")
    asset_name = selected_asset.get("name", "upx-win64.zip")
    if not asset_url:
        raise RuntimeError("UPX release asset is missing a download URL.")

    archive_path = UPX_DIR / asset_name
    extract_root = UPX_DIR / "extracted"

    _print(f"Downloading UPX from {asset_url}")
    archive_path.write_bytes(_download_bytes(asset_url))

    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_root)

    upx_exe = find_upx_executable()
    if upx_exe is None:
        raise RuntimeError("UPX download completed, but upx.exe was not found after extraction.")

    _print(f"UPX ready at {upx_exe}")
    return upx_exe


def ensure_upx() -> Path:
    """Return a usable UPX executable, downloading it if needed."""
    upx_exe = find_upx_executable()
    if upx_exe is not None:
        _print(f"Using existing UPX: {upx_exe}")
        return upx_exe
    return download_upx()


def build_minimal() -> Path:
    """Build the smallest practical OptCNT GUI executable."""
    if not running_in_virtualenv():
        _print("Warning: build_minimal.py is designed to run inside a dedicated virtual environment.")

    clean_build_dirs()
    upx_exe = ensure_upx()

    args = [
        "main.py",
        "--name=OptCNT",
        "--onefile",
        "--windowed",
        f"--upx-dir={upx_exe.parent}",
        "--optimize=2",
        "--clean",
        "--log-level=WARN",
        "--collect-data=matplotlib",
    ]

    if os.name != "nt":
        args.append("--strip")

    for module_name in EXCLUDE_MODULES:
        args.append(f"--exclude-module={module_name}")

    for module_name in HIDDEN_IMPORTS:
        args.append(f"--hidden-import={module_name}")

    for filename in UPX_EXCLUDES:
        args.append(f"--upx-exclude={filename}")

    _print("Starting PyInstaller build...")
    _print(" ".join(args))
    PyInstaller.__main__.run(args)

    exe_path = ROOT_DIR / "dist" / "OptCNT.exe"
    if not exe_path.exists():
        raise FileNotFoundError(f"Build finished without producing {exe_path}")

    size_mb = exe_path.stat().st_size / (1024 * 1024)
    _print(f"Build completed: {exe_path}")
    _print(f"Executable size: {size_mb:.2f} MB")

    build_dir = ROOT_DIR / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
        _print(f"Removed {build_dir}")

    return exe_path


if __name__ == "__main__":
    build_minimal()
