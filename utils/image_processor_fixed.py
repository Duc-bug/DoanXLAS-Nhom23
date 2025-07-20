import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class SignatureProcessor:
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size
        self.scaler = StandardScaler()
    
    def preprocess_image(self, image_input):
        """
        Xử lý ảnh chữ ký: chuyển sang grayscale, resize, normalize
        """
        try:
            # Xử lý input - có thể là đường dẫn file hoặc numpy array
            if isinstance(image_input, str):
                # Đọc từ file
                image = cv2.imread(image_input)
                if image is None:
                    raise ValueError(f"Không thể đọc ảnh từ {image_input}")
            elif hasattr(image_input, 'read'):
                # File object từ Streamlit
                import io
                from PIL import Image
                pil_image = Image.open(image_input)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            else:
                # Numpy array
                image = image_input.copy()
            
            # Đảm bảo ảnh không None
            if image is None:
                raise ValueError("Ảnh đầu vào không hợp lệ")
            
            # Chuyển sang grayscale
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                elif image.shape[2] == 3:  # RGB/BGR
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image[:, :, 0]  # Lấy channel đầu tiên
            else:
                gray = image.copy()
            
            # Đảm bảo ảnh có kích thước hợp lệ
            if gray.shape[0] == 0 or gray.shape[1] == 0:
                raise ValueError("Ảnh có kích thước không hợp lệ")
            
            # Cải thiện contrast
            gray = cv2.equalizeHist(gray)
            
            # Áp dụng threshold để tách nền (adaptive threshold tốt hơn)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Tìm contours để crop ảnh
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Lọc contours quá nhỏ
                filtered_contours = [c for c in contours if cv2.contourArea(c) > 50]
                
                if filtered_contours:
                    # Tìm bounding box của tất cả contours
                    all_contours = np.vstack(filtered_contours)
                    x, y, w, h = cv2.boundingRect(all_contours)
                    
                    # Crop ảnh với padding
                    padding = 20
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    x2 = min(gray.shape[1], x + w + 2 * padding)
                    y2 = min(gray.shape[0], y + h + 2 * padding)
                    
                    cropped = gray[y:y2, x:x2]
                else:
                    cropped = gray
            else:
                cropped = gray
            
            # Đảm bảo cropped có kích thước hợp lý
            if cropped.shape[0] < 10 or cropped.shape[1] < 10:
                cropped = gray
            
            # Resize về kích thước chuẩn
            resized = cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Normalize về [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            print(f"Lỗi xử lý ảnh: {str(e)}")
            # Trả về ảnh đen như fallback
            return np.zeros(self.target_size, dtype=np.float32)
    
    def extract_features(self, image):
        """
        Trích xuất đặc trưng từ ảnh chữ ký
        """
        try:
            # Đảm bảo ảnh đúng định dạng
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            
            # Nếu ảnh đã được normalize về [0,1], chuyển về [0,255] cho việc tính gradient
            if np.max(image) <= 1.0:
                image_255 = (image * 255).astype(np.uint8)
            else:
                image_255 = image.astype(np.uint8)
            
            # Histogram của gradient
            grad_x = cv2.Sobel(image_255, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image_255, cv2.CV_64F, 0, 1, ksize=3)
            
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            angle = np.arctan2(grad_y, grad_x)
            
            # Histogram của magnitude
            mag_hist, _ = np.histogram(magnitude.flatten(), bins=32, range=(0, 255))
            
            # Histogram của góc
            angle_hist, _ = np.histogram(angle.flatten(), bins=32, range=(-np.pi, np.pi))
            
            # Các đặc trưng thống kê
            stats = [
                np.mean(image),
                np.std(image),
                np.min(image),
                np.max(image),
                np.sum(image > 0.5) / image.size if np.max(image) <= 1.0 else np.sum(image > 127) / image.size  # Tỷ lệ pixel sáng
            ]
            
            # Kết hợp tất cả đặc trưng
            features = np.concatenate([
                image.flatten(),  # Raw pixels
                mag_hist / (np.sum(mag_hist) + 1e-8),  # Normalized gradient magnitude histogram
                angle_hist / (np.sum(angle_hist) + 1e-8),  # Normalized gradient angle histogram
                stats            # Statistical features
            ])
            
            return features
            
        except Exception as e:
            print(f"Lỗi trích xuất đặc trưng: {str(e)}")
            # Trả về vector zero như fallback
            return np.zeros(self.target_size[0] * self.target_size[1] + 64 + 5)
    
    def calculate_similarity(self, features1, features2):
        """
        Tính độ tương đồng giữa hai tập đặc trưng
        """
        try:
            # Đảm bảo features có cùng kích thước
            min_length = min(len(features1), len(features2))
            features1 = features1[:min_length]
            features2 = features2[:min_length]
            
            # Cosine similarity
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_sim = dot_product / (norm1 * norm2)
            
            # Euclidean distance (normalized)
            euclidean_dist = np.linalg.norm(features1 - features2)
            max_possible_dist = np.sqrt(len(features1) * 2)  # Max distance for normalized features
            euclidean_sim = 1 - (euclidean_dist / max_possible_dist)
            
            # Kết hợp cả hai metric
            similarity = (cosine_sim + euclidean_sim) / 2
            
            return max(0, min(1, similarity))  # Clamp về [0, 1]
            
        except Exception as e:
            print(f"Lỗi tính similarity: {str(e)}")
            return 0.0
    
    def visualize_signature(self, image, title="Chữ ký"):
        """
        Hiển thị ảnh chữ ký
        """
        try:
            plt.figure(figsize=(8, 6))
            if len(image.shape) == 2:
                plt.imshow(image, cmap='gray')
            else:
                plt.imshow(image)
            plt.title(title)
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Lỗi hiển thị ảnh: {str(e)}")
