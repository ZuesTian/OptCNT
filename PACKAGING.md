# Windows packaging

Use a dedicated virtual environment so PyInstaller only sees the minimal runtime
dependencies required by the app.

## One-command build

```powershell
powershell -ExecutionPolicy Bypass -File .\package_windows.ps1
```

The script will:

1. Create `.venv-build`
2. Install `requirements-build.txt`
3. Download UPX automatically when needed
4. Build `dist\OptCNT.exe`

## Minimal runtime dependencies

`requirements.txt` intentionally excludes optional heavy packages:

- `opencv-python-headless` is used instead of the full OpenCV desktop build
- `scikit-learn` is optional and listed in `requirements-optional.txt`
- `scipy` and `scikit-image` are no longer required for the packaged GUI build

If you want the optional clustering implementation, install:

```powershell
.venv-build\Scripts\python.exe -m pip install -r .\requirements-optional.txt
```
