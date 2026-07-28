# Supervisor del daemon krita-autoselect: lo relanza si muere.
# torch puede morir con un crash nativo (sin traceback) bajo presion de
# memoria — p.ej. cuando ComfyUI carga un modelo grande en paralelo — y un
# daemon caido rompe `kri select sam` y el docker hasta que alguien lo note.
# Uso:  powershell -File run_daemon.ps1
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $repo ".venv\Scripts\python.exe"
$server = Join-Path $repo "server.py"
$weights = Join-Path $repo "models\sam3"
if (Test-Path $weights) { $env:AUTOSELECT_WEIGHTS_PATH = $weights }
while ($true) {
    Write-Host "[supervisor] iniciando daemon..."
    & $python -u $server
    Write-Host "[supervisor] el daemon murio (exit $LASTEXITCODE); reinicio en 2s"
    Start-Sleep -Seconds 2
}
