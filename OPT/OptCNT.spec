# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = []
datas += collect_data_files('skimage')
datas += collect_data_files('matplotlib')


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['cv2', 'numpy', 'PIL', 'PIL._imagingtk', 'PIL._tkinter_finder', 'skimage', 'skimage.filters', 'skimage.morphology', 'matplotlib', 'matplotlib.backends.backend_tkagg', 'matplotlib.pyplot'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib.tests', 'numpy.random._examples', 'scipy', 'pandas', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'tkinter.test', 'unittest', 'pytest', 'IPython', 'jupyter', 'notebook', 'sphinx', 'pydoc', 'email', 'http.server', 'xmlrpc', 'multiprocessing', 'concurrent.futures', 'ctypes.test', 'pip', 'wheel', 'Cython', 'cython', 'tkinter.tix', 'tkinter.ttk.test', 'idlelib', 'test', 'lib2to3', 'ensurepip', 'venv', 'turtledemo', 'turtle'],
    noarchive=False,
    optimize=2,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [('O', None, 'OPTION'), ('O', None, 'OPTION')],
    name='OptCNT',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
