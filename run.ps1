# Run SpeechtoText (activate audio_env if present)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (Test-Path "audio_env\Scripts\Activate.ps1") {
    Write-Host "Activating audio_env..."
    & .\audio_env\Scripts\Activate.ps1
}
python main.py @args
