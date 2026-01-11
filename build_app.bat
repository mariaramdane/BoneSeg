@echo off
echo ====================================================
echo Building BoneSeg EXE - Isolation Environment
echo ====================================================

:: 1. Activate virtual environment
if not exist "env" (
    echo [ERROR] Virtual environment 'env' not found. Run setup_me.bat first.
    pause
    exit /b
)
call env\Scripts\activate.bat

:: 2. Clean up old build files
echo [+] Cleaning up old files...
if exist "build" rd /s /q build
if exist "dist" rd /s /q dist
if exist "BoneSeg.spec" del BoneSeg.spec

:: 3. Launch build (FIXED FOR ND2READER AND SKLEARN)
echo [+] Compiling (this may take a few minutes)...
python -m PyInstaller --noconfirm --onedir --windowed ^
    --icon="icon.ico" ^
    --add-data "sam2/sam2/configs;sam2/sam2/configs" ^
    --add-data "sam2/checkpoints;sam2/checkpoints" ^
    --add-data "logo.png;." ^
    --collect-all "hydra" ^
    --collect-all "sam2" ^
    --collect-all "customtkinter" ^
    --collect-all "cellpose" ^
    --hidden-import "nd2reader" ^
    --hidden-import "pims" ^
    --hidden-import "sklearn.utils._typedefs" ^
    --hidden-import "sklearn.neighbors._partition_nodes" ^
    "BoneSeg.py"

:: 4. Verification and Deployment
if not exist "dist\BoneSeg\BoneSeg.exe" (
    echo.
    echo [ERROR] PyInstaller failed! Check the console for errors.
    pause
    exit /b
)

echo [+] Deploying files...
if exist "BoneSeg.exe" del /f /q "BoneSeg.exe"
if exist "_internal" rd /s /q "_internal"

move /y "dist\BoneSeg\BoneSeg.exe" "."
move /y "dist\BoneSeg\_internal" "."

:: 5. Final cleanup
rd /s /q dist
rd /s /q build
if exist "BoneSeg.spec" del BoneSeg.spec

echo.
echo ====================================================
echo SUCCESS! BoneSeg.exe is ready.
echo ====================================================
pause