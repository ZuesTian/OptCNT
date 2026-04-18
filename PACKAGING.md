# Windows packaging

Use a dedicated virtual environment so PyInstaller only sees the minimal runtime
dependencies required by the app.

## One-command build

```powershell
powershell -ExecutionPolicy Bypass -File .\package_windows.ps1
```

This defaults to the `slim` packaging profile. To keep `numba` / `llvmlite`
for maximum runtime acceleration, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\package_windows.ps1 -Profile performance
```

The script will:

1. Create `.venv-build`
2. Install `requirements-build.txt`
3. Download UPX automatically when needed
4. Build `dist\OptCNT.exe`
5. Run a final `upx --best` pass on the packaged executable

## Minimal runtime dependencies

`requirements.txt` keeps the packaged GUI lean while preserving fast thinning:

- `opencv-contrib-python-headless` is used so `cv2.ximgproc.thinning` is available in packaged builds
- `numba` is kept in the packaged build so vectorized / JIT-accelerated paths remain available at runtime
- `scikit-learn` is optional and listed in `requirements-optional.txt`
- `scipy` and `scikit-image` are no longer required for the packaged GUI build

## Profiles

- `slim`: smallest executable size, excludes `numba` / `llvmlite`
- `performance`: larger executable, keeps `numba` / `llvmlite` and trims only their test modules
- both profiles use UPX `--best` plus a final safe UPX pass on the onefile executable

If you want the optional clustering implementation, install:

```powershell
.venv-build\Scripts\python.exe -m pip install -r .\requirements-optional.txt
```
