# setup_venv.ps1 - PowerShell script to set up virtual environment

Write-Host "Setting up Python virtual environment..." -ForegroundColor Cyan

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPath = Join-Path $projectDir "venv"

# Check if venv already exists
if (Test-Path $venvPath) {
    Write-Host "[INFO] Virtual environment already exists at: $venvPath" -ForegroundColor Yellow
    Write-Host "To activate it, run:" -ForegroundColor Yellow
    Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Green
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv venv
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Virtual environment created successfully!" -ForegroundColor Green
        Write-Host "To activate it, run:" -ForegroundColor Yellow
        Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`nTo install dependencies after activation:" -ForegroundColor Yellow
Write-Host "  pip install -r requirements.txt" -ForegroundColor Green

