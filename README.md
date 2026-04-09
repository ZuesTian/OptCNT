# OptCNT

OptCNT is a CNT SEM image analysis tool built with Python, Tkinter, and OpenCV.

## Run

```bash
python main.py
```

## Layout

```text
OptCNT/
├── main.py                  # launcher entrypoint
├── src/
│   ├── main.py              # app entrypoint
│   ├── core/
│   │   ├── analyzer_core.py
│   │   ├── models.py
│   │   ├── stats_compat.py
│   │   └── utils.py
│   └── gui/
│       ├── gui.py
│       ├── gui_styles.py
│       ├── panels.py
│       └── widgets.py
├── tests/
├── build_minimal.py
├── package_windows.ps1
├── OptCNT.spec
└── PACKAGING.md
```

## Install

```bash
pip install -r requirements.txt
```

For tests:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Packaging

Windows packaging notes live in `PACKAGING.md`.
