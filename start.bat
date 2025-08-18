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
    echo.
    echo WARNING: FFMPEG not found in bin\ffmpeg.exe
    echo.
    echo Please download FFMPEG manually:
    echo 1. Go to: https://www.gyan.dev/ffmpeg/builds/
    echo 2. Download the "essentials" build
    echo 3. Extract ffmpeg.exe to the bin\ directory
    echo.
    echo The application will continue but video processing won't work without FFMPEG.
    echo.
    pause
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

echo Starting Electron application...
npm start

REM If we reach here, the app has closed
echo.
echo Application closed.
echo.
echo If there were any issues, check:
echo 1. Python is installed and in your PATH
echo 2. FFMPEG is in the bin directory  
echo 3. Node.js is installed and in your PATH
echo 4. Check logs folder for Python error details
echo.
pause
