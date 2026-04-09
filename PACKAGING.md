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

`requirements.txt` keeps the packaged GUI lean while preserving fast thinning:

- `opencv-contrib-python-headless` is used so `cv2.ximgproc.thinning` is available in packaged builds
- `scikit-learn` is optional and listed in `requirements-optional.txt`
- `scipy` and `scikit-image` are no longer required for the packaged GUI build

If you want the optional clustering implementation, install:

```powershell
.venv-build\Scripts\python.exe -m pip install -r .\requirements-optional.txt
```
