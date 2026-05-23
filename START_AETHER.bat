@echo off
setlocal
echo ========================================================
echo  NCA AETHER AI IDE — AUTOMATED LAUNCHER
echo ========================================================
echo.

:: Get the directory of the batch file
set "ROOT_DIR=%~dp0"

:: Use Node.js for auto-compilation and dispatching
node "%ROOT_DIR%launch.js"

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Automated launch sequence failed.
    pause
)
