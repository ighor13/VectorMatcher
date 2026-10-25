# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

# --- Собираем PyTorch и его CUDA-зависимости ---
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')
torch_hiddenimports += [
    'torch._C',
    'torch.cuda',
    'torch.backends.cudnn',
    'torch.backends.cuda',
    'sentence_transformers',
    'transformers',
]

# --- Собираем PySide6 ---
pyside_datas, pyside_binaries, pyside_hiddenimports = collect_all('PySide6')

# --- Указываем, какие файлы положить рядом с exe ---
# Мы убрали отсюда old.tsv и new.tsv. Положим их руками.
datas = []

# --- Собираем всё вместе ---
a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=torch_binaries + pyside_binaries,
    datas=torch_datas + pyside_datas + datas,
    hiddenimports=torch_hiddenimports + pyside_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'IPython',
        'pytest',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VectorMatcher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VectorMatcher',
)