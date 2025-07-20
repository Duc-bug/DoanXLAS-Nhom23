"""
Phiên bản đơn giản của mô hình nhận diện chữ ký
Không cần TensorFlow, chỉ dùng OpenCV và sklearn
"""

import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pickle
import os

class SimpleSignatureModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_features(self, image):
        """
        Trích xuất đặc trưng từ ảnh chữ ký
        """
        # Đảm bảo ảnh là grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize về kích thước chuẩn
        image = cv2.resize(image, (128, 128))
        
        # 1. Raw pixel features (giảm xuống)
        resized_img = cv2.resize(image, (32, 32))  # Giảm kích thước
        pixel_features = resized_img.flatten() / 255.0
        
        # 2. Gradient features
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x)
        
        # Histogram của gradient
        mag_hist, _ = np.histogram(magnitude.flatten(), bins=16, range=(0, 255))
        angle_hist, _ = np.histogram(angle.flatten(), bins=16, range=(-np.pi, np.pi))
        
        # 3. Contour features
        contours, _ = cv2.findContours(
            (image < 128).astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        contour_features = []
        if contours:
            # Largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Moments
            moments = cv2.moments(largest_contour)
            if moments['m00'] != 0:
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
            else:
                cx = cy = 0
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h != 0 else 0
            
            contour_features = [area, perimeter, cx, cy, aspect_ratio, len(contours)]
        else:
            contour_features = [0, 0, 0, 0, 0, 0]
        
        # 4. Statistical features
        stats = [
            np.mean(image),
            np.std(image),
            np.min(image),
            np.max(image),
            np.sum(image < 128) / image.size,  # Tỷ lệ pixel đen
            np.sum(image > 200) / image.size   # Tỷ lệ pixel trắng
        ]
        
        # Kết hợp tất cả features
        all_features = np.concatenate([
            pixel_features,      # 32*32 = 1024 features
            mag_hist / np.sum(mag_hist) if np.sum(mag_hist) > 0 else mag_hist,  # 16 features
            angle_hist / np.sum(angle_hist) if np.sum(angle_hist) > 0 else angle_hist,  # 16 features
            contour_features,    # 6 features
            stats               # 6 features
        ])
        
        return all_features
    
    def calculate_similarity(self, features1, features2):
        """
        Tính độ tương đồng giữa hai tập đặc trưng
        """
        # Reshape để dùng cosine similarity
        features1 = features1.reshape(1, -1)
        features2 = features2.reshape(1, -1)
        
        # Cosine similarity
        cosine_sim = cosine_similarity(features1, features2)[0][0]
        
        # Euclidean similarity
        euclidean_dist = np.linalg.norm(features1 - features2)
        max_dist = np.sqrt(len(features1[0]))
        euclidean_sim = 1 - (euclidean_dist / max_dist)
        
        # Kết hợp
        similarity = (cosine_sim + euclidean_sim) / 2
        
        return max(0, min(1, similarity))  # Clamp về [0, 1]
    
    def predict_similarity(self, image1, image2):
        """
        Dự đoán độ tương đồng giữa hai ảnh
        """
        features1 = self.extract_features(image1)
        features2 = self.extract_features(image2)
        
        return self.calculate_similarity(features1, features2)
    
    def save_model(self, filepath):
        """
        Lưu mô hình
        """
        model_data = {
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Đã lưu mô hình tại: {filepath}")
    
    def load_model(self, filepath):
        """
        Load mô hình đã lưu
        """
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.scaler = model_data['scaler']
                self.is_trained = model_data['is_trained']
                
                print(f"Đã load mô hình từ: {filepath}")
                return True
            except Exception as e:
                print(f"Lỗi khi load mô hình: {e}")
                return False
        else:
            print(f"Không tìm thấy mô hình tại: {filepath}")
            return False
    
    def summary(self):
        """
        Hiển thị thông tin mô hình
        """
        print("=== SIMPLE SIGNATURE MODEL SUMMARY ===")
        print("Model type: Traditional ML + Computer Vision")
        print("Features: Pixels + Gradients + Contours + Statistics")
        print(f"Status: {'Trained' if self.is_trained else 'Not trained'}")
        print("Total features: ~1068 (1024 + 16 + 16 + 6 + 6)")

# Adapter class để tương thích với code cũ
class SiameseNetwork:
    def __init__(self, input_shape=(128, 128, 1)):
        self.model = SimpleSignatureModel()
        self.input_shape = input_shape
        self.base_network = None
        
    def create_siamese_network(self):
        """Tương thích với API cũ"""
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Tương thích với API cũ"""
        pass
    
    def predict_similarity(self, image1, image2):
        """Tương thích với API cũ"""
        return self.model.predict_similarity(image1, image2)
    
    def save_model(self, filepath):
        """Tương thích với API cũ"""
        # Đổi đuôi file
        filepath = filepath.replace('.h5', '.pkl')
        self.model.save_model(filepath)
    
    def load_model(self, filepath):
        """Tương thích với API cũ"""
        # Thử cả hai đuôi file
        pkl_path = filepath.replace('.h5', '.pkl')
        if os.path.exists(pkl_path):
            return self.model.load_model(pkl_path)
        else:
            return self.model.load_model(filepath)
    
    def summary(self):
        """Tương thích với API cũ"""
        self.model.summary()
