$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }

& $PythonBin -m pip install -r requirements-local-tools.txt
& $PythonBin scripts/setup_local_classifiers.py --model all

if ($env:WITH_RAG_MODELS -eq "1") {
    & $PythonBin -m pip install -r requirements-rag.txt
    & $PythonBin scripts/setup_rag.py
}
if ($env:WITH_OCR -eq "1") {
    & $PythonBin -m pip install -r requirements-ocr.txt
    & $PythonBin scripts/setup_ocr.py
}

Write-Host "Local models are ready. No API key was read or written."
