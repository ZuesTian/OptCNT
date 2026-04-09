# OptCNT

OptCNT is a desktop tool for CNT SEM image analysis built with Python, Tkinter, and OpenCV.

It focuses on practical single-image and ROI-based workflows:

- automatic or manual scale calibration
- adjustable preprocessing with live preview
- CNT detection and length / width / slenderness measurement
- result visualization with contour and skeleton overlays
- spatial distribution and hotspot analysis
- lean Windows packaging with UPX compression

## Quick Start

Install the runtime dependencies and launch the app:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

The repository root `main.py` is a thin launcher that delegates to `src.main`.

## Project Layout

```text
OptCNT/
|-- main.py
|-- src/
|   |-- main.py
|   |-- core/
|   |   |-- analyzer_core.py
|   |   |-- models.py
|   |   |-- stats_compat.py
|   |   `-- utils.py
|   `-- gui/
|       |-- gui.py
|       |-- gui_styles.py
|       |-- panels.py
|       `-- widgets.py
|-- tests/
|-- benchmark.py
|-- benchmark_data.py
|-- build_minimal.py
|-- package_windows.ps1
|-- PACKAGING.md
|-- requirements.txt
|-- requirements-dev.txt
|-- requirements-build.txt
`-- requirements-optional.txt
```

## Dependencies

- `requirements.txt`: runtime dependencies for the desktop app
- `requirements-dev.txt`: runtime dependencies plus `pytest`
- `requirements-build.txt`: runtime dependencies plus PyInstaller
- `requirements-optional.txt`: optional extras such as `scikit-learn`

The packaged build uses `opencv-contrib-python-headless` so the fast `cv2.ximgproc.thinning` path is available in the executable as well.

## Tests

```powershell
pip install -r requirements-dev.txt
pytest -q
```

## Windows Packaging

Build the smallest practical Windows executable with UPX:

```powershell
powershell -ExecutionPolicy Bypass -File .\package_windows.ps1
```

This script will:

1. recreate `.venv-build`
2. install `requirements-build.txt`
3. reuse or download UPX when needed
4. build `dist\OptCNT.exe`

The packaging flow is tuned to keep the executable small without sacrificing the OpenCV thinning path used by CNT skeleton processing.

More details are documented in [PACKAGING.md](PACKAGING.md).
