$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
python -m pip install -e ".[monitoring,build]"
python -m PyInstaller --noconfirm --clean --noconsole --onedir --name LTX_GUI --collect-submodules PySide6 ltx_gui/__main__.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
Write-Host "Completed: dist/LTX_GUI/LTX_GUI.exe"
