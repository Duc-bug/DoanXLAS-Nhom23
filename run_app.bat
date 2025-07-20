@echo off
echo ========================================
echo   AI Signature Verification App
echo ========================================
echo.

REM Kiểm tra virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found!
    echo Please run setup.bat first to install the application.
    pause
    exit /b 1
)

REM Kích hoạt virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Kiểm tra Streamlit
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ❌ Streamlit not found! Please run setup.bat first.
    pause
    exit /b 1
)

echo ✅ Environment ready
echo.

REM Tạo thư mục nếu chưa có
if not exist "data" mkdir data
if not exist "data\signatures" mkdir data\signatures
if not exist "data\test" mkdir data\test

echo 🚀 Starting AI Signature Verification App...
echo.
echo The app will open in your browser at: http://localhost:8501
echo To stop the app, press Ctrl+C in this window.
echo.

REM Chạy ứng dụng
streamlit run app.py

pause
