
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os

class SiameseNetwork:
    def __init__(self, input_shape=(128, 128, 1)):
        self.input_shape = input_shape
        self.model = None
        self.base_network = None
        
    def create_base_network(self):
        """
        Tạo mạng cơ sở để trích xuất đặc trưng
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Convolutional layers
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(128, activation='relu')(x)
        
        return Model(inputs, outputs, name='base_network')
    
    def euclidean_distance(self, vectors):
        """
        Tính khoảng cách Euclidean giữa hai vector
        """
        x, y = vectors
        sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))
    
    def create_siamese_network(self):
        """
        Tạo mạng Siamese hoàn chỉnh
        """
        # Tạo base network
        self.base_network = self.create_base_network()
        
        # Input cho hai ảnh
        input_a = layers.Input(shape=self.input_shape, name='input_a')
        input_b = layers.Input(shape=self.input_shape, name='input_b')
        
        # Trích xuất đặc trưng từ cả hai ảnh
        processed_a = self.base_network(input_a)
        processed_b = self.base_network(input_b)
        
        # Tính khoảng cách
        distance = layers.Lambda(self.euclidean_distance, name='distance')([processed_a, processed_b])
        
        # Output: xác suất hai ảnh giống nhau
        outputs = layers.Dense(1, activation='sigmoid', name='similarity')(distance)
        
        self.model = Model(inputs=[input_a, input_b], outputs=outputs, name='siamese_network')
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile mô hình
        """
        if self.model is None:
            self.create_siamese_network()
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, pairs, labels, validation_data=None, epochs=50, batch_size=32):
        """
        Huấn luyện mô hình
        """
        if self.model is None:
            raise ValueError("Mô hình chưa được tạo. Gọi create_siamese_network() trước.")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Chia dữ liệu train
        train_data = [pairs[:, 0], pairs[:, 1]]
        
        history = self.model.fit(
            train_data,
            labels,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict_similarity(self, image1, image2):
        """
        Dự đoán độ tương đồng giữa hai ảnh
        """
        if self.model is None:
            raise ValueError("Mô hình chưa được load hoặc huấn luyện.")
        
        # Đảm bảo shape đúng
        if len(image1.shape) == 2:
            image1 = np.expand_dims(image1, axis=-1)
        if len(image2.shape) == 2:
            image2 = np.expand_dims(image2, axis=-1)
        
        # Thêm batch dimension
        image1 = np.expand_dims(image1, axis=0)
        image2 = np.expand_dims(image2, axis=0)
        
        # Dự đoán
        similarity = self.model.predict([image1, image2], verbose=0)[0][0]
        
        return similarity
    
    def save_model(self, filepath):
        """
        Lưu mô hình
        """
        if self.model is None:
            raise ValueError("Không có mô hình để lưu.")
        
        # Lưu toàn bộ mô hình
        self.model.save(filepath)
        print(f"Đã lưu mô hình tại: {filepath}")
    
    def load_model(self, filepath):
        """
        Load mô hình đã lưu
        """
        if os.path.exists(filepath):
            self.model = tf.keras.models.load_model(filepath)
            print(f"Đã load mô hình từ: {filepath}")
            return True
        else:
            print(f"Không tìm thấy mô hình tại: {filepath}")
            return False
    
    def summary(self):
        """
        Hiển thị thông tin mô hình
        """
        if self.model is not None:
            print("=== SIAMESE NETWORK SUMMARY ===")
            self.model.summary()
            
            if self.base_network is not None:
                print("\n=== BASE NETWORK SUMMARY ===")
                self.base_network.summary()
        else:
            print("Mô hình chưa được tạo.")
