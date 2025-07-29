import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

def select_image():
    """Chọn file ảnh từ máy tính"""
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh chữ ký",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.JPG *.PNG *.BMP")]
        )
        
        root.destroy()
        return file_path
    except Exception as e:
        print(f"Lỗi chọn file: {e}")
        return None

def load_image_unicode(file_path):
    """Load ảnh có tên tiếng Việt"""
    try:
        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        return img
    except:
        return None

def preprocess_image(img):
    """Xử lý ảnh với chi tiết từng bước"""
    # Bước 1: Cải thiện contrast
    enhanced = cv2.equalizeHist(img)
    
    # Bước 2: Phân ngưỡng OTSU
    threshold_val, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Bước 3: Resize để trích xuất đặc trưng
    resized = cv2.resize(binary, (64, 64))
    
    return enhanced, binary, resized, threshold_val

def extract_features(img):
    """Trích xuất đặc trưng"""
    pixels = img.flatten() / 255.0
    mean_val = np.mean(pixels)
    std_val = np.std(pixels)
    features = np.concatenate([pixels, [mean_val, std_val]])
    return features

def calculate_similarity(f1, f2):
    """Tính độ tương tự cosine"""
    dot_product = np.dot(f1, f2)
    norm1 = np.linalg.norm(f1)
    norm2 = np.linalg.norm(f2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot_product / (norm1 * norm2)

def compare_signatures():
    """So sánh 2 chữ ký với báo cáo chi tiết"""
    # Chọn ảnh
    print("🔍 Chọn ảnh chữ ký thứ 1...")
    img1_path = select_image()
    if not img1_path:
        print("❌ Không chọn được ảnh 1!")
        return
    
    print("🔍 Chọn ảnh chữ ký thứ 2...")
    img2_path = select_image()
    if not img2_path:
        print("❌ Không chọn được ảnh 2!")
        return
    
    # Load ảnh
    print("📂 Đang load và xử lý ảnh...")
    img1 = load_image_unicode(img1_path)
    img2 = load_image_unicode(img2_path)
    
    if img1 is None or img2 is None:
        print("❌ Không đọc được ảnh!")
        return
    
    # Xử lý ảnh
    enhanced1, binary1, processed1, thresh1 = preprocess_image(img1)
    enhanced2, binary2, processed2, thresh2 = preprocess_image(img2)
    
    # Trích xuất đặc trưng
    features1 = extract_features(processed1)
    features2 = extract_features(processed2)
    
    # Tính độ tương tự
    similarity = calculate_similarity(features1, features2)
    threshold = 0.5
    
    # === HIỂN THỊ KẾT QUẢ ===
    
    plt.figure(figsize=(15, 10))
    
    # 1. Ảnh gốc
    plt.subplot(3, 4, 1)
    plt.imshow(img1, cmap='gray')
    plt.title(f'Ảnh 1 gốc\n{img1.shape}')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(img2, cmap='gray')
    plt.title(f'Ảnh 2 gốc\n{img2.shape}')
    plt.axis('off')
    
    # 2. Ảnh sau cải thiện contrast
    plt.subplot(3, 4, 3)
    plt.imshow(enhanced1, cmap='gray')
    plt.title('Ảnh 1 cải thiện')
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(enhanced2, cmap='gray')
    plt.title('Ảnh 2 cải thiện')
    plt.axis('off')
    
    # 3. Ảnh sau phân ngưỡng
    plt.subplot(3, 4, 5)
    plt.imshow(binary1, cmap='gray')
    plt.title(f'Ảnh 1 phân ngưỡng\n(Ngưỡng: {thresh1:.0f})')
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    plt.imshow(binary2, cmap='gray')
    plt.title(f'Ảnh 2 phân ngưỡng\n(Ngưỡng: {thresh2:.0f})')
    plt.axis('off')
    
    # 4. Ảnh cuối cùng (64x64)
    plt.subplot(3, 4, 7)
    plt.imshow(processed1, cmap='gray')
    plt.title('Ảnh 1 trích xuất\n(64x64)')
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.imshow(processed2, cmap='gray')
    plt.title('Ảnh 2 trích xuất\n(64x64)')
    plt.axis('off')
    
    # 5. So sánh đặc trưng (200 đầu tiên)
    plt.subplot(3, 4, 9)
    plt.plot(features1[:200], 'blue', label='Ảnh 1', alpha=0.7)
    plt.plot(features2[:200], 'red', label='Ảnh 2', alpha=0.7)
    plt.title('So sánh đặc trưng')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Histogram đặc trưng
    plt.subplot(3, 4, 10)
    plt.hist(features1[:-2], bins=30, alpha=0.5, label='Ảnh 1', color='blue')
    plt.hist(features2[:-2], bins=30, alpha=0.5, label='Ảnh 2', color='red')
    plt.title('Phân bố đặc trưng')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Thống kê đặc trưng (SỬA LỖI Ở ĐÂY)
    plt.subplot(3, 4, 11)
    stats1 = [np.mean(features1[:-2]), np.std(features1[:-2])]
    stats2 = [np.mean(features2[:-2]), np.std(features2[:-2])]
    x = np.arange(2)
    width = 0.35
    plt.bar(x - width/2, stats1, width, label='Ảnh 1', alpha=0.7)
    plt.bar(x + width/2, stats2, width, label='Ảnh 2', alpha=0.7)
    plt.xticks(x, ['Mean', 'Std'])  # SỬA: plt.xticks thay vì plt.set_xticks
    plt.title('Thống kê so sánh')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Kết quả cuối cùng
    plt.subplot(3, 4, 12)
    color = 'green' if similarity >= threshold else 'red'
    plt.bar(['Độ tương tự'], [similarity], color=color, alpha=0.8)
    plt.axhline(y=threshold, color='black', linestyle='--', label=f'Ngưỡng = {threshold}')
    plt.ylabel('Điểm số')
    plt.title('KẾT QUẢ CUỐI')
    plt.ylim(0, 1)
    plt.text(0, similarity + 0.02, f'{similarity:.3f}', ha='center', va='bottom', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('BÁO CÁO CHI TIẾT SO SÁNH CHỮ KÝ', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # === BÁO CÁO CHI TIẾT ===
    print(f"\n📋 BÁO CÁO CHI TIẾT:")
    print("=" * 50)
    print(f"📄 File 1: {os.path.basename(img1_path)}")
    print(f"📄 File 2: {os.path.basename(img2_path)}")
    print(f"📐 Kích thước gốc: {img1.shape} vs {img2.shape}")
    print(f"🎯 Ngưỡng OTSU: {thresh1:.0f} vs {thresh2:.0f}")
    print(f"🧬 Số đặc trưng: {len(features1)} cho mỗi ảnh")
    print(f"📊 Độ tương tự: {similarity:.6f}")
    print(f"📈 Tỷ lệ khớp: {similarity * 100:.2f}%")
    print(f"⚖️ Ngưỡng quyết định: {threshold}")
    
    if similarity >= threshold:
        confidence = ((similarity - threshold) / (1 - threshold)) * 100
        print(f"\n✅ KẾT LUẬN: HAI CHỮ KÝ GIỐNG NHAU")
        print(f"✅ Mức độ tin cậy: {confidence:.1f}%")
    else:
        difference = ((threshold - similarity) / threshold) * 100
        print(f"\n❌ KẾT LUẬN: HAI CHỮ KÝ KHÁC NHAU")
        print(f"❌ Mức độ sai khác: {difference:.1f}%")
    
    return similarity

if __name__ == "__main__":
    compare_signatures()