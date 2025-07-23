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
        Trích xuất đặc trưng từ ảnh chữ ký - FIXED VERSION
        """
        try:
            print(f"🔍 Bắt đầu trích xuất features từ ảnh shape: {image.shape}")
            
            # Validate input
            if image is None:
                raise ValueError("Ảnh đầu vào là None")
            
            if len(image.shape) != 2:
                raise ValueError(f"Ảnh phải là grayscale 2D, nhận được shape: {image.shape}")
            
            # Kiểm tra kích thước ảnh
            if image.shape != self.target_size:
                print(f"⚠️ Ảnh không đúng kích thước: {image.shape}, resize về {self.target_size}")
                image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Đảm bảo ảnh đúng định dạng
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            
            # Chuyển về [0,255] cho việc tính gradient
            if np.max(image) <= 1.0:
                image_255 = (image * 255).astype(np.uint8)
            else:
                image_255 = image.astype(np.uint8)
            
            print(f"📊 Ảnh sau preprocessing: shape={image.shape}, dtype={image.dtype}, min={np.min(image):.3f}, max={np.max(image):.3f}")
            
            # 1. RAW PIXEL FEATURES (128x128 = 16,384 features)
            raw_pixels = image.flatten()
            print(f"✅ Raw pixels: {len(raw_pixels)} features")
            
            # 2. GRADIENT FEATURES
            try:
                # Tính gradient bằng Sobel
                grad_x = cv2.Sobel(image_255, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(image_255, cv2.CV_64F, 0, 1, ksize=3)
                
                # Magnitude và direction
                magnitude = np.sqrt(grad_x**2 + grad_y**2)
                angle = np.arctan2(grad_y, grad_x)
                
                print(f"✅ Gradient: magnitude shape={magnitude.shape}, angle shape={angle.shape}")
                
                # Histogram của magnitude (32 bins)
                mag_hist, _ = np.histogram(magnitude.flatten(), bins=32, range=(0, 300))  # Tăng range
                mag_hist = mag_hist.astype(np.float32)
                if np.sum(mag_hist) > 0:
                    mag_hist = mag_hist / np.sum(mag_hist)  # Normalize
                print(f"✅ Magnitude histogram: {len(mag_hist)} bins, sum={np.sum(mag_hist):.3f}")
                
                # Histogram của góc (32 bins) 
                angle_hist, _ = np.histogram(angle.flatten(), bins=32, range=(-np.pi, np.pi))
                angle_hist = angle_hist.astype(np.float32)
                if np.sum(angle_hist) > 0:
                    angle_hist = angle_hist / np.sum(angle_hist)  # Normalize
                print(f"✅ Angle histogram: {len(angle_hist)} bins, sum={np.sum(angle_hist):.3f}")
                
            except Exception as e:
                print(f"⚠️ Lỗi gradient features: {e}")
                mag_hist = np.zeros(32, dtype=np.float32)
                angle_hist = np.zeros(32, dtype=np.float32)
        
        # 3. STATISTICAL FEATURES (5 features)
        try:
            mean_val = float(np.mean(image))
            std_val = float(np.std(image))
            min_val = float(np.min(image))
            max_val = float(np.max(image))
            
            # Tỷ lệ pixel sáng (threshold = 0.5 cho normalized image)
            if np.max(image) <= 1.0:
                bright_ratio = float(np.sum(image > 0.5) / image.size)
            else:
                bright_ratio = float(np.sum(image > 127) / image.size)
            
            stats = np.array([mean_val, std_val, min_val, max_val, bright_ratio], dtype=np.float32)
            print(f"✅ Statistical features: {stats}")
            
        except Exception as e:
            print(f"⚠️ Lỗi statistical features: {e}")
            stats = np.zeros(5, dtype=np.float32)
        
        # 4. KẾT HỢP TẤT CẢ FEATURES
        try:
            features = np.concatenate([
                raw_pixels,    # 16,384 features (128x128)
                mag_hist,      # 32 features  
                angle_hist,    # 32 features
                stats         # 5 features
            ])
            print(f"✅ Kết hợp features: {len(features)} total")
            # Total: 16,384 + 32 + 32 + 5 = 16,453 features
            
        except Exception as e:
            print(f"❌ Lỗi kết hợp features: {e}")
            expected_size = self.target_size[0] * self.target_size[1] + 32 + 32 + 5
            return np.zeros(expected_size, dtype=np.float32)
        
        # Validate output size
        expected_size = self.target_size[0] * self.target_size[1] + 32 + 32 + 5
        if len(features) != expected_size:
            print(f"⚠️ Kích thước features không đúng: {len(features)}, mong đợi: {expected_size}")
            # Resize nếu cần
            if len(features) < expected_size:
                features = np.pad(features, (0, expected_size - len(features)), 'constant')
            else:
                features = features[:expected_size]
        
        # Validate output
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            print("⚠️ Features chứa NaN hoặc Inf, đang sửa...")
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        
        print(f"✅ Trích xuất thành công {len(features)} features")
        return features.astype(np.float32)
        
    except Exception as e:
        print(f"❌ Lỗi trích xuất đặc trưng: {str(e)}")
        import traceback
        traceback.print_exc()
        # Trả về vector zero với kích thước chính xác
        fallback_size = self.target_size[0] * self.target_size[1] + 32 + 32 + 5  # 16,453
        return np.zeros(fallback_size, dtype=np.float32)
    
    def calculate_similarity(self, features1, features2):
        """
        So sánh độ tương đồng - IMPROVED VERSION
        """
        try:
            # Validate inputs
            if features1 is None or features2 is None:
                print("❌ Features là None")
                return 0.0
            
            f1 = np.array(features1, dtype=np.float32).flatten()
            f2 = np.array(features2, dtype=np.float32).flatten()
            
            if len(f1) == 0 or len(f2) == 0:
                print("❌ Features rỗng")
                return 0.0
            
            if len(f1) != len(f2):
                print(f"❌ Features khác chiều: {len(f1)} vs {len(f2)}")
                return 0.0
            
            # Clean data
            f1 = np.nan_to_num(f1, nan=0.0, posinf=1.0, neginf=0.0)
            f2 = np.nan_to_num(f2, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Method 1: Cosine Similarity
            dot_product = np.dot(f1, f2)
            norm1 = np.linalg.norm(f1)
            norm2 = np.linalg.norm(f2)
            
            if norm1 == 0 or norm2 == 0:
                cosine_sim = 0.0
            else:
                cosine_sim = dot_product / (norm1 * norm2)
                cosine_sim = max(0, min(1, cosine_sim))
            
            # Method 2: Euclidean Similarity với standardization
            try:
                # Combine và standardize
                combined = np.vstack([f1.reshape(1, -1), f2.reshape(1, -1)])
                scaler = StandardScaler()
                scaled = scaler.fit_transform(combined)
                
                f1_scaled = scaled[0]
                f2_scaled = scaled[1]
                
                # Euclidean distance
                distance = np.linalg.norm(f1_scaled - f2_scaled)
                max_distance = np.sqrt(len(f1_scaled))
                
                euclidean_sim = 1 - (distance / max_distance)
                euclidean_sim = max(0, min(1, euclidean_sim))
                
            except Exception:
                euclidean_sim = cosine_sim
            
            # Weighted combination
            final_similarity = 0.6 * cosine_sim + 0.4 * euclidean_sim
            
            print(f"🔍 Cosine: {cosine_sim:.4f}, Euclidean: {euclidean_sim:.4f}, Final: {final_similarity:.4f}")
            
            return float(final_similarity)
            
        except Exception as e:
            print(f"❌ Lỗi tính similarity: {str(e)}")
            return 0.0
    
    def compare_signatures(self, image1, image2):
        """
        So sánh 2 ảnh chữ ký hoàn chỉnh
        """
        try:
            print("🔄 Bắt đầu so sánh chữ ký...")
            
            # Preprocess
            processed1 = self.preprocess_image(image1)
            processed2 = self.preprocess_image(image2)
            
            # Extract features
            features1 = self.extract_features(processed1)
            features2 = self.extract_features(processed2)
            
            # Calculate similarity
            similarity = self.calculate_similarity(features1, features2)
            
            print(f"✅ Hoàn thành so sánh. Similarity: {similarity:.4f}")
            
            return {
                'similarity': similarity,
                'processed_image1': processed1,
                'processed_image2': processed2,
                'features1': features1,
                'features2': features2
            }
            
        except Exception as e:
            print(f"❌ Lỗi so sánh chữ ký: {str(e)}")
            return {
                'similarity': 0.0,
                'processed_image1': np.zeros(self.target_size, dtype=np.float32),
                'processed_image2': np.zeros(self.target_size, dtype=np.float32), 
                'features1': np.zeros(16453, dtype=np.float32),
                'features2': np.zeros(16453, dtype=np.float32)
            }
    
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



if __name__ == "__main__":
    test_processor()
