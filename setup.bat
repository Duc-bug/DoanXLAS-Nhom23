@echo off
echo ========================================
echo   AI Signature Verification App Setup
echo ========================================
echo.

REM Kiểm tra Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+ first.
    echo Download from: https://python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Tạo virtual environment
echo 🔧 Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

echo ✅ Virtual environment created
echo.

REM Kích hoạt virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Nâng cấp pip
echo 🔧 Upgrading pip...
python -m pip install --upgrade pip

REM Cài đặt dependencies
echo 🔧 Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    echo Try running: pip install -r requirements.txt
    pause
    exit /b 1
)

echo ✅ Dependencies installed
echo.

REM Tạo thư mục cần thiết
echo 🔧 Creating necessary directories...
if not exist "data" mkdir data
if not exist "data\signatures" mkdir data\signatures
if not exist "data\test" mkdir data\test
if not exist "model" mkdir model

echo ✅ Directories created
echo.

REM Chạy demo test
echo 🧪 Running demo tests...
python demo_test.py
if errorlevel 1 (
    echo ⚠️ Some tests failed, but you can still try running the app
) else (
    echo ✅ All tests passed!
)

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo To start the application:
echo   1. Make sure virtual environment is activated: venv\Scripts\activate
echo   2. Run: streamlit run app.py
echo.
echo The app will open in your browser at: http://localhost:8501
echo.

REM Hỏi có muốn chạy ngay không
set /p choice="Do you want to start the app now? (y/n): "
if /i "%choice%"=="y" (
    echo.
    echo 🚀 Starting the application...
    streamlit run app.py
) else (
    echo.
    echo To start later, run: streamlit run app.py
)

pause
