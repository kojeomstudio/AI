# -*- mode: python ; coding: utf-8 -*-
import os
import sys

block_cipher = None

PROJECT_DIR = os.path.dirname(os.path.abspath(SPECPATH))

a = Analysis(
    [os.path.join(PROJECT_DIR, 'app.py')],
    pathex=[PROJECT_DIR],
    binaries=[],
    datas=[
        (os.path.join(PROJECT_DIR, 'config'), 'config'),
        (os.path.join(PROJECT_DIR, 'ml', 'training_output'), os.path.join('ml', 'training_output')),
        (os.path.join(PROJECT_DIR, 'assets'), 'assets'),
    ],
    hiddenimports=[
        'ultralytics',
        'cv2',
        'numpy',
        'PIL',
        'pyautogui',
        'win32gui',
        'win32con',
        'win32api',
        'win32process',
        'ctypes',
        'ctypes.wintypes',
        'psutil',
        'jsonschema',
        'typer',
        'torch',
        'torchvision',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'seaborn',
        'pandas',
        'scipy',
        'sympy',
        'networkx',
        'IPython',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MabinogiMobileMacro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MabinogiMobileMacro',
)
