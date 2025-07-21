# 🖋️ Ứng Dụng Nhận Diện Chữ Ký AI

Ứng dụng AI nhận diện và xác minh chữ ký sử dụng Siamese Network và Streamlit.

## 🚀 Tính năng

- ✅ Đăng ký và quản lý chữ ký mẫu
- ✅ Xác minh chữ ký với AI
- ✅ Vẽ chữ ký trực tiếp trên web
- ✅ Thống kê và lịch sử xác minh
- ✅ Huấn luyện mô hình tùy chỉnh

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

- Python 3.8+
- RAM: 4GB+ 
- CPU hỗ trợ AVX

## 📁 Cấu trúc

```
app_nhan_dien_chu_ky/
├── model/               # Mô hình AI
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
5. **Xem thống kê**: Dashboard và lịch sử chi tiết

## 🤖 Mô hình AI

- **Kiến trúc**: Siamese Network với CNN
- **Input**: Ảnh 128x128 grayscale  
- **Output**: Độ tương tự 0-100%
- **Ngưỡng mặc định**: 80% (max) + 75% (avg) - NGHIÊM NGẶT

## 📝 License

MIT License - Xem file LICENSE để biết chi tiết.