# 🖋️ Ứng Dụng Nhận Diện Chữ Ký

Ứng dụng web đơn giản để kiểm tra chữ ký có giống với chữ ký mẫu hay không.

## ✨ Chức năng

- 📝 Lưu chữ ký mẫu
- 🔍 Kiểm tra chữ ký mới  
- 🎨 Vẽ chữ ký trên web
- 👤 Quản lý người dùng

## 🚀 Cách chạy

### Cách 1: Dễ nhất (Windows)
1. Tải Python từ: https://python.org
2. Chạy file `setup.bat`
3. Chạy file `run_app.bat`
4. Mở: http://localhost:8501

### Cách 2: Thủ công
```bash
pip install streamlit opencv-python scikit-learn pandas numpy pillow
streamlit run app.py
```

## 📖 Cách sử dụng

1. **Tạo người dùng**: Vào mục "Quản Lý Người Dùng"
2. **Thêm chữ ký mẫu**: Upload 2-3 ảnh chữ ký 
3. **Kiểm tra**: Upload ảnh chữ ký cần kiểm tra
4. **Xem kết quả**: Hợp lệ ✅ hoặc Không hợp lệ ❌

## 💡 Lưu ý

- Ảnh nên có nền trắng, chữ đen
- Chụp rõ nét, không bị mờ
- Đây là đồ án học tập, chỉ để demo

## Cấu trúc dự án
-app_nhan_dien_chu_ky/             
├── data/               # Database và ảnh
├── utils/              # Tiện ích xử lý
├── app.py              # Ứng dụng chính
├── requirements.txt    # Dependencies
└── README.md          # Tài liệu

## 🛠️ Công nghệ

- Python + Streamlit
- OpenCV (xử lý ảnh)
- SQLite (lưu dữ liệu)

##Liên hệ

**Email**: duc.2373401010100@vanlanguni.vn  
**GitHub**: [Nhóm 23](https://github.com/Duc-bug/DoanXLAS-Nhom23)

---
**Nhóm 23 - Đồ án 2025** 
