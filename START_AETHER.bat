@echo off
setlocal
echo ========================================================
echo  NCA AETHER AI IDE — ONE-CLICK LAUNCHER
echo ========================================================
echo.

:: Get the directory of the batch file
set "ROOT_DIR=%~dp0"

:: Navigate to the Release folder
pushd "%ROOT_DIR%build\deployment\Release"

if exist "Aether_AI_IDE.exe" (
    echo [LAUNCH] Starting Silicon Engine ^& Dashboard...
    start "" "Aether_AI_IDE.exe"
) else (
    echo [ERROR] Binaries not found! Please build the project first.
    echo Expected path: %ROOT_DIR%build\deployment\Release\Aether_AI_IDE.exe
    pause
)

popd
echo [DONE] Bootstrapper dispatched.
exit
