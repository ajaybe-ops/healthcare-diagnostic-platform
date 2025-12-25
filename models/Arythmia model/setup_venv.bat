@echo off
REM setup_venv.bat - Batch script to set up virtual environment

echo Setting up Python virtual environment...

if exist venv (
    echo [INFO] Virtual environment already exists.
    echo To activate it, run:
    echo   venv\Scripts\activate.bat
) else (
    echo Creating virtual environment...
    python -m venv venv
    
    if %ERRORLEVEL% EQU 0 (
        echo [OK] Virtual environment created successfully!
        echo To activate it, run:
        echo   venv\Scripts\activate.bat
    ) else (
        echo [ERROR] Failed to create virtual environment
        exit /b 1
    )
)

echo.
echo To install dependencies after activation:
echo   pip install -r requirements.txt

pause

