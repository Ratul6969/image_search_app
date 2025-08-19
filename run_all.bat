@echo off
setlocal ENABLEDELAYEDEXPANSION

:: Set the project directory (where this script is located)
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

:: --- 1. Check for Conda installation ---
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo ----------------------------------------------------
    echo ERROR: Conda is not found in your PATH.
    echo Please install Miniconda/Anaconda: https://docs.conda.io/en/latest/miniconda.html
    echo Or open "Anaconda Prompt" and run this script there.
    echo ----------------------------------------------------
    pause
    exit /b 1
)

:: --- 2. Check for existing environment and create if not found ---
echo.
echo Checking for 'image_search' Conda environment...
conda env list | findstr /B /C:"image_search " >nul 2>nul
if %errorlevel% neq 0 (
    echo Environment 'image_search' not found. Creating it...
    conda create -n image_search python=3.9 -y || (
        echo.
        echo ----------------------------------------------------
        echo ERROR: Failed to create Conda environment.
        echo Please ensure Conda is properly installed and try again.
        echo ----------------------------------------------------
        pause
        exit /b 1
    )
) else (
    echo Environment 'image_search' found.
)

:: --- 3. Activate Conda environment ---
echo Activating Conda environment...
call conda activate image_search || (
    echo.
    echo ----------------------------------------------------
    echo ERROR: Failed to activate Conda environment 'image_search'.
    echo Please try running "conda activate image_search" manually in your terminal.
    echo ----------------------------------------------------
    pause
    exit /b 1
)

:: --- 4. Install dependencies ---
echo.
echo Installing Python dependencies from requirements.txt...
pip install -r requirements.txt || (
    echo.
    echo ----------------------------------------------------
    echo ERROR: Failed to install Python dependencies.
    echo Check the error messages above for details.
    echo ----------------------------------------------------
    pause
    exit /b 1
)

:: --- 5. Run the data cleaning script (NEW STEP) ---
echo.
echo Running clean_product_data.py to infer and update product metadata...
echo (Check clean_data_log.txt for detailed progress and errors)
call python clean_product_data.py > clean_data_log.txt 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ----------------------------------------------------
    echo ERROR: Data cleaning script (clean_product_data.py) failed.
    echo Please review clean_data_log.txt for the exact error.
    echo Ensure your products.json exists and Gemini API key in config.py is correct.
    echo ----------------------------------------------------
    pause
    exit /b 1
)
del clean_data_log.txt >nul 2>nul
echo.
echo Data cleaning script completed.

:: --- 6. Run the setup script to download images (if needed) and build Annoy index ---
echo.
echo Running setup.py to build Annoy index...
call python setup.py || (
    echo.
    echo ----------------------------------------------------
    echo ERROR: Setup script (setup.py) failed.
    echo Check previous messages for specific errors.
    echo ----------------------------------------------------
    pause
    exit /b 1
)

:: --- 7. Start the Flask app ---
echo.
echo Starting Flask app...
echo Access the API at http://localhost:5000
echo Open index.html in your browser: %PROJECT_DIR%index.html
echo Press Ctrl+C in this window to stop the server.
echo.
python app.py

endlocal
pause
