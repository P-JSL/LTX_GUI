$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
python -m pip install -e ".[build]"
if ($LASTEXITCODE -ne 0) { throw "Dependency install failed" }
python -m PyInstaller --noconfirm --clean --noconsole --onedir --name LTX_GUI --collect-all cv2 --collect-submodules PySide6 ltx_gui/__main__.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
Write-Host "GUI executable: dist/LTX_GUI/LTX_GUI.exe"
Write-Host "Note: Install FFmpeg separately (or place ffmpeg.exe in PATH)."
