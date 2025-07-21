# 🖋️ Ứng Dụng Nhận Diện Chữ Ký

Ứng dụng  nhận diện và xác minh chữ ký sử dụng Siamese Network và Streamlit.

## 🚀 Tính năng

-  Đăng ký và quản lý chữ ký mẫu
-  Vẽ chữ ký trực tiếp trên web



## ⚡ Cài đặt và chạy

### Cài đặt nhanh (Windows)
```bash
setup.bat
run_app.bat
```

### Cài đặt thủ công
```bash
# Clone repository
git clone https://github.com/username/app_nhan_dien_chu_ky.git
cd app_nhan_dien_chu_ky

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy ứng dụng
streamlit run app.py
```

Ứng dụng sẽ mở tại: `http://localhost:8501`

## 🛠️ Yêu cầu

- Python 3.12
- RAM: 4GB+ 
- CPU hỗ trợ AVX

## 📁 Cấu trúc

```
app_nhan_dien_chu_ky/
├── model/               
├── data/               # Database và ảnh
├── utils/              # Tiện ích xử lý
├── app.py              # Ứng dụng chính
├── requirements.txt    # Dependencies
└── README.md          # Tài liệu
```

## 📖 Hướng dẫn sử dụng

1. **Tạo người dùng**: Menu "👤 Quản Lý Người Dùng"
2. **Đăng ký mẫu**: Upload 3-5 ảnh chữ ký mẫu
3. **Xác minh**: Upload ảnh để kiểm tra độ tương tự
4. **Vẽ chữ ký**: Vẽ trực tiếp trên canvas






## 📝 License

MIT License - Xem file LICENSE để biết chi tiết.
