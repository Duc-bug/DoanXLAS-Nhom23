import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from model.siamese_network import SiameseNetwork
from utils.image_processor import SignatureProcessor

class SignatureTrainer:
    def __init__(self, data_dir="data", model_save_path="model/signature_model.h5"):
        self.data_dir = data_dir
        self.model_save_path = model_save_path
        self.processor = SignatureProcessor()
        self.siamese_net = SiameseNetwork()
        
    def load_dataset(self, genuine_dir="genuine", forged_dir="forged"):
        """
        Load dataset từ thư mục
        Cấu trúc thư mục:
        data/
        ├── genuine/
        │   ├── person1/
        │   │   ├── sig1.png
        │   │   ├── sig2.png
        │   └── person2/
        └── forged/
            ├── person1/
            └── person2/
        """
        genuine_path = os.path.join(self.data_dir, genuine_dir)
        forged_path = os.path.join(self.data_dir, forged_dir)
        
        genuine_signatures = {}
        forged_signatures = {}
        
        # Load genuine signatures
        if os.path.exists(genuine_path):
            for person in os.listdir(genuine_path):
                person_path = os.path.join(genuine_path, person)
                if os.path.isdir(person_path):
                    genuine_signatures[person] = []
                    for img_file in os.listdir(person_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(person_path, img_file)
                            try:
                                processed_img = self.processor.preprocess_image(img_path)
                                genuine_signatures[person].append(processed_img)
                            except Exception as e:
                                print(f"Lỗi khi xử lý {img_path}: {e}")
        
        # Load forged signatures
        if os.path.exists(forged_path):
            for person in os.listdir(forged_path):
                person_path = os.path.join(forged_path, person)
                if os.path.isdir(person_path):
                    forged_signatures[person] = []
                    for img_file in os.listdir(person_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(person_path, img_file)
                            try:
                                processed_img = self.processor.preprocess_image(img_path)
                                forged_signatures[person].append(processed_img)
                            except Exception as e:
                                print(f"Lỗi khi xử lý {img_path}: {e}")
        
        return genuine_signatures, forged_signatures
    
    def create_pairs(self, genuine_signatures, forged_signatures, pairs_per_person=100):
        """
        Tạo các cặp ảnh để huấn luyện
        """
        pairs = []
        labels = []
        
        for person in genuine_signatures:
            person_genuine = genuine_signatures[person]
            person_forged = forged_signatures.get(person, [])
            
            # Tạo positive pairs (cặp chữ ký thật)
            if len(person_genuine) >= 2:
                for _ in range(pairs_per_person // 2):
                    idx1, idx2 = random.sample(range(len(person_genuine)), 2)
                    pairs.append([person_genuine[idx1], person_genuine[idx2]])
                    labels.append(1)  # Genuine pair
            
            # Tạo negative pairs (cặp chữ ký thật vs giả)
            if len(person_forged) > 0:
                for _ in range(pairs_per_person // 4):
                    genuine_idx = random.randint(0, len(person_genuine) - 1)
                    forged_idx = random.randint(0, len(person_forged) - 1)
                    pairs.append([person_genuine[genuine_idx], person_forged[forged_idx]])
                    labels.append(0)  # Forged pair
            
            # Tạo negative pairs (cặp chữ ký của người khác)
            other_people = [p for p in genuine_signatures if p != person]
            if other_people:
                for _ in range(pairs_per_person // 4):
                    other_person = random.choice(other_people)
                    other_signatures = genuine_signatures[other_person]
                    if other_signatures:
                        genuine_idx = random.randint(0, len(person_genuine) - 1)
                        other_idx = random.randint(0, len(other_signatures) - 1)
                        pairs.append([person_genuine[genuine_idx], other_signatures[other_idx]])
                        labels.append(0)  # Different person
        
        # Chuyển thành numpy arrays
        pairs = np.array(pairs)
        labels = np.array(labels)
        
        # Reshape để phù hợp với input của Siamese network
        if len(pairs) > 0:
            pairs = pairs.reshape(pairs.shape[0], 2, pairs.shape[2], pairs.shape[3], 1)
        
        print(f"Đã tạo {len(pairs)} cặp ảnh")
        print(f"Positive pairs: {np.sum(labels)}")
        print(f"Negative pairs: {len(labels) - np.sum(labels)}")
        
        return pairs, labels
    
    def create_demo_dataset(self):
        """
        Tạo dataset demo với ảnh mẫu
        """
        print("Tạo dataset demo...")
        
        # Tạo thư mục demo
        demo_genuine_dir = os.path.join(self.data_dir, "demo_genuine")
        demo_forged_dir = os.path.join(self.data_dir, "demo_forged")
        
        os.makedirs(demo_genuine_dir, exist_ok=True)
        os.makedirs(demo_forged_dir, exist_ok=True)
        
        # Tạo một số chữ ký demo bằng cách vẽ
        def create_signature_sample(name, signature_type, num_samples=5):
            person_dir = os.path.join(demo_genuine_dir if signature_type == "genuine" else demo_forged_dir, name)
            os.makedirs(person_dir, exist_ok=True)
            
            for i in range(num_samples):
                # Tạo ảnh trắng
                img = np.ones((200, 400), dtype=np.uint8) * 255
                
                # Vẽ chữ ký giả lập
                if signature_type == "genuine":
                    # Chữ ký thật - có pattern nhất định
                    cv2.putText(img, f"{name}", (50, 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, 0, 3)
                    # Thêm một số đường cong
                    cv2.ellipse(img, (200, 150), (100, 20), 0, 0, 180, 0, 2)
                else:
                    # Chữ ký giả - khác biệt một chút
                    cv2.putText(img, f"{name}", (60, 110), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.8, 0, 3)
                    # Đường cong khác
                    cv2.ellipse(img, (180, 140), (80, 25), 15, 0, 160, 0, 2)
                
                # Thêm nhiễu
                noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
                img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
                
                filename = f"{name}_{signature_type}_{i+1}.png"
                filepath = os.path.join(person_dir, filename)
                cv2.imwrite(filepath, img)
        
        # Tạo mẫu cho 3 người
        for person in ["John", "Alice", "Bob"]:
            create_signature_sample(person, "genuine", 8)
            create_signature_sample(person, "forged", 5)
        
        print("Đã tạo dataset demo!")
        
        return self.load_dataset("demo_genuine", "demo_forged")
    
    def train_model(self, genuine_signatures=None, forged_signatures=None, 
                   epochs=50, batch_size=32, validation_split=0.2):
        """
        Huấn luyện mô hình
        """
        # Nếu không có dữ liệu, tạo demo dataset
        if genuine_signatures is None or forged_signatures is None:
            print("Không có dữ liệu training, tạo demo dataset...")
            genuine_signatures, forged_signatures = self.create_demo_dataset()
        
        if not genuine_signatures:
            raise ValueError("Không có dữ liệu chữ ký thật để huấn luyện!")
        
        # Tạo pairs
        pairs, labels = self.create_pairs(genuine_signatures, forged_signatures)
        
        if len(pairs) == 0:
            raise ValueError("Không thể tạo cặp dữ liệu để huấn luyện!")
        
        # Chia train/validation
        train_pairs, val_pairs, train_labels, val_labels = train_test_split(
            pairs, labels, test_size=validation_split, random_state=42, stratify=labels
        )
        
        print(f"Training pairs: {len(train_pairs)}")
        print(f"Validation pairs: {len(val_pairs)}")
        
        # Tạo và compile mô hình
        self.siamese_net.create_siamese_network()
        self.siamese_net.compile_model()
        
        # Chuẩn bị dữ liệu validation
        val_data = None
        if len(val_pairs) > 0:
            val_data = ([val_pairs[:, 0], val_pairs[:, 1]], val_labels)
        
        # Huấn luyện
        history = self.siamese_net.train(
            train_pairs, train_labels,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Lưu mô hình
        self.save_model()
        
        # Vẽ biểu đồ training
        self.plot_training_history(history)
        
        return history
    
    def save_model(self):
        """
        Lưu mô hình đã huấn luyện
        """
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        self.siamese_net.save_model(self.model_save_path)
    
    def load_model(self):
        """
        Load mô hình đã lưu
        """
        return self.siamese_net.load_model(self.model_save_path)
    
    def plot_training_history(self, history):
        """
        Vẽ biểu đồ quá trình huấn luyện
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(self.model_save_path), 'training_history.png'))
        plt.show()
    
    def evaluate_model(self, test_pairs, test_labels):
        """
        Đánh giá mô hình
        """
        if self.siamese_net.model is None:
            if not self.load_model():
                raise ValueError("Không thể load mô hình!")
        
        # Dự đoán
        predictions = self.siamese_net.model.predict([test_pairs[:, 0], test_pairs[:, 1]])
        binary_predictions = (predictions > 0.5).astype(int).flatten()
        
        # Tính accuracy
        accuracy = np.mean(binary_predictions == test_labels)
        
        # Tính precision, recall
        tp = np.sum((binary_predictions == 1) & (test_labels == 1))
        fp = np.sum((binary_predictions == 1) & (test_labels == 0))
        fn = np.sum((binary_predictions == 0) & (test_labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

if __name__ == "__main__":
    # Chạy training
    trainer = SignatureTrainer()
    trainer.train_model(epochs=30, batch_size=16)
