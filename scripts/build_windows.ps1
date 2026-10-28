param([switch]$IncludeNative)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($IncludeNative) {
    python -m pip install -e ".[build,native]"
} else {
    python -m pip install -e ".[build]"
}
if ($LASTEXITCODE -ne 0) { throw "Dependency install failed" }
$packArgs = @("--noconfirm", "--clean", "--noconsole", "--onedir", "--name", "LTX_GUI",
              "--collect-all", "cv2", "--collect-submodules", "PySide6")
if ($IncludeNative) {
    $packArgs += @("--collect-all", "torch", "--hidden-import", "ltx_gui.native_engine",
                   "--collect-submodules", "ltx_gui.neural")
}
python -m PyInstaller @packArgs ltx_gui/__main__.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
Write-Host "GUI executable: dist/LTX_GUI/LTX_GUI.exe"
Write-Host "Note: Install FFmpeg separately (or place ffmpeg.exe in PATH)."

if (-not $IncludeNative) {
    Write-Host "Neural mode requires a separate native-enabled build: ./scripts/build_windows.ps1 -IncludeNative"
} else {
    Write-Host "Native PyTorch included; trained checkpoints and FFmpeg are NOT bundled."
}
