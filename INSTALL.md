# 🚀 Hướng Dẫn Cài Đặt Nhanh

## Yêu cầu hệ thống
- Windows 10/11
- Python 3.8 hoặc mới hơn
- RAM: Tối thiểu 4GB
- Dung lượng: 2GB trống

## Cài đặt nhanh (3 bước)
streamlit run app.py chạy cái này sau khi vào thư viện ảo
### Bước 1: Cài đặt Python
- Tải Python từ: https://python.org/downloads/
- Chọn "Add Python to PATH" khi cài đặt
- Restart máy sau khi cài đặt

### Bước 2: Cài đặt ứng dụng
```bash
# Mở Command Prompt (cmd) tại thư mục ứng dụng
# Chạy lệnh:
setup.bat
```

### Bước 3: Chạy ứng dụng
```bash
# Chạy file:
run_app.bat
```

## Cài đặt thủ công (nếu cần)

```bash
# 1. Tạo virtual environment
python -m venv venv

# 2. Kích hoạt virtual environment
venv\Scripts\activate

# 3. Cài đặt packages
pip install -r requirements.txt

# 4. Chạy ứng dụng
streamlit run app.py
```

## Kiểm tra cài đặt
- Chạy `demo_test.py` để kiểm tra hệ thống
- Mở http://localhost:8501 trên trình duyệt
- Thử tạo người dùng và đăng ký chữ ký mẫu

## Sửa lỗi thường gặp

**Lỗi: "Python not found"**
```bash
# Kiểm tra Python đã cài đặt
python --version

# Nếu chưa có, tải và cài đặt Python từ python.org
```

**Lỗi: "pip install failed"**
```bash
# Nâng cấp pip
python -m pip install --upgrade pip

# Cài đặt lại
pip install -r requirements.txt
```

**Lỗi: "Streamlit command not found"**
```bash
# Kích hoạt lại virtual environment
venv\Scripts\activate

# Cài đặt lại Streamlit
pip install streamlit
```

**Lỗi: "Module not found"**
```bash
# Đảm bảo đang ở đúng thư mục và virtual environment
cd signature_ai_app
venv\Scripts\activate
python app.py
```

## Liên hệ hỗ trợ
- Tạo issue trên GitHub
- Email: support@signature-ai.com
- Đọc README.md để biết thêm chi tiết
