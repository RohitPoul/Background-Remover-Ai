@echo off
echo ========================================
echo Video Background Remover - Starting App
echo ========================================

REM Check Node.js and npm
where node >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    pause
    exit /b 1
)

REM Check Python
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Install Node dependencies if needed
if not exist node_modules (
    echo Installing Node.js dependencies...
    npm install
)

REM Ensure Python dependencies are installed
if exist api_requirements.txt (
    echo Checking Python dependencies...
    python -m pip install --upgrade pip >nul 2>&1
    python -m pip install -r api_requirements.txt >nul 2>&1
    echo Python dependencies ready.
)

REM Check if FFMPEG exists
if not exist bin\ffmpeg.exe (
    echo FFMPEG not found. Downloading...
    call npm run download-ffmpeg
    if not exist bin\ffmpeg.exe (
        echo WARNING: FFMPEG download may have failed.
        echo You can download it manually from https://www.gyan.dev/ffmpeg/builds/
        echo and place ffmpeg.exe in the bin directory.
        echo.
        echo Continuing anyway...
    )
)

REM Build React app if needed
if not exist frontend\build (
    echo Building React frontend...
    cd frontend
    npm run build
    cd ..
)

REM Start the app
echo.
echo Starting application...
echo Python API will run on http://localhost:5000
echo Electron app will open shortly...
echo.
set REACT_APP_API_BASE=http://localhost:5000
npm start

echo.
echo If the application didn't start, please check:
echo 1. Python is installed and in your PATH
echo 2. FFMPEG is in the bin directory
echo 3. Node.js is installed and in your PATH
echo.
pause
