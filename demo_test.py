"""
Demo script để kiểm tra các chức năng cơ bản của ứng dụng
Chạy script này để đảm bảo mọi thứ hoạt động đúng
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.image_processor import SignatureProcessor
from utils.database import SignatureDatabase
from model.siamese_network import SiameseNetwork
from model.trainer import SignatureTrainer

def test_image_processor():
    """Test xử lý ảnh"""
    print("🖼️ Testing Image Processor...")
    
    processor = SignatureProcessor()
    
    # Tạo ảnh mẫu
    test_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "John Doe", (50, 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, 0, 3)
    cv2.ellipse(test_image, (200, 150), (100, 20), 0, 0, 180, 0, 2)
    
    try:
        # Test preprocessing
        processed = processor.preprocess_image(test_image)
        print(f"✅ Preprocessed image shape: {processed.shape}")
        
        # Test feature extraction
        features = processor.extract_features(processed)
        print(f"✅ Extracted {len(features)} features")
        
        # Test similarity calculation
        similarity = processor.calculate_similarity(features, features)
        print(f"✅ Self-similarity: {similarity:.4f} (should be ~1.0)")
        
        return True
    except Exception as e:
        print(f"❌ Image processor test failed: {e}")
        return False

def test_database():
    """Test database operations"""
    print("\n🗄️ Testing Database...")
    
    # Sử dụng test database
    db = SignatureDatabase("test_database.db")
    
    try:
        # Test user operations
        user_id = db.add_user("Test User", "test@example.com")
        print(f"✅ Added user with ID: {user_id}")
        
        user = db.get_user("Test User")
        print(f"✅ Retrieved user: {user['name']}")
        
        # Test signature operations
        dummy_features = np.random.rand(100)
        sig_id = db.add_signature(user_id, "test_path.png", dummy_features, is_template=True)
        print(f"✅ Added signature with ID: {sig_id}")
        
        templates = db.get_template_signatures(user_id)
        print(f"✅ Found {len(templates)} template signatures")
        
        # Test verification
        verification_id = db.save_verification(user_id, sig_id, sig_id, 0.95, True)
        print(f"✅ Saved verification with ID: {verification_id}")
        
        # Test stats
        stats = db.get_stats()
        print(f"✅ Database stats: {stats}")
        
        # Cleanup
        os.remove("test_database.db")
        print("✅ Cleaned up test database")
        
        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        if os.path.exists("test_database.db"):
            os.remove("test_database.db")
        return False

def test_siamese_network():
    """Test Siamese Network"""
    print("\n🧠 Testing Siamese Network...")
    
    try:
        # Tạo mô hình
        siamese = SiameseNetwork(input_shape=(64, 64, 1))  # Smaller for testing
        model = siamese.create_siamese_network()
        print("✅ Created Siamese Network")
        
        # Compile
        siamese.compile_model()
        print("✅ Compiled model")
        
        # Test prediction với dummy data
        dummy_img1 = np.random.rand(64, 64)
        dummy_img2 = np.random.rand(64, 64)
        
        similarity = siamese.predict_similarity(dummy_img1, dummy_img2)
        print(f"✅ Prediction works, similarity: {similarity:.4f}")
        
        # Test model summary
        siamese.summary()
        
        return True
    except Exception as e:
        print(f"❌ Siamese Network test failed: {e}")
        return False

def test_trainer():
    """Test trainer functionality"""
    print("\n🏋️ Testing Trainer...")
    
    try:
        trainer = SignatureTrainer(data_dir="test_data", model_save_path="test_model.h5")
        
        # Test demo dataset creation
        genuine_sigs, forged_sigs = trainer.create_demo_dataset()
        print(f"✅ Created demo dataset: {len(genuine_sigs)} genuine, {len(forged_sigs)} forged")
        
        # Test pair creation
        pairs, labels = trainer.create_pairs(genuine_sigs, forged_sigs, pairs_per_person=20)
        print(f"✅ Created {len(pairs)} training pairs")
        
        # Cleanup
        if os.path.exists("test_data"):
            import shutil
            shutil.rmtree("test_data")
        if os.path.exists("test_model.h5"):
            os.remove("test_model.h5")
        print("✅ Cleaned up test files")
        
        return True
    except Exception as e:
        print(f"❌ Trainer test failed: {e}")
        return False

def create_demo_signatures():
    """Tạo một số ảnh chữ ký mẫu"""
    print("\n🎨 Creating demo signatures...")
    
    demo_dir = "data/demo_signatures"
    os.makedirs(demo_dir, exist_ok=True)
    
    names = ["John_Doe", "Alice_Smith", "Bob_Johnson"]
    
    for name in names:
        for i in range(3):
            # Tạo ảnh chữ ký
            img = np.ones((150, 300), dtype=np.uint8) * 255
            
            # Viết tên
            cv2.putText(img, name.replace("_", " "), (20, 80), 
                       cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5, 0, 2)
            
            # Thêm đường gạch dưới
            cv2.line(img, (20, 100), (280, 105), 0, 2)
            
            # Thêm một chút nhiễu để tạo sự khác biệt
            noise = np.random.normal(0, 5, img.shape)
            img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
            
            filename = f"{name}_signature_{i+1}.png"
            filepath = os.path.join(demo_dir, filename)
            cv2.imwrite(filepath, img)
    
    print(f"✅ Created demo signatures in {demo_dir}")
    return demo_dir

def main():
    """Chạy tất cả tests"""
    print("🚀 Starting Demo Tests...\n")
    
    tests = [
        ("Image Processor", test_image_processor),
        ("Database", test_database),
        ("Siamese Network", test_siamese_network),
        ("Trainer", test_trainer),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Tạo demo signatures
    try:
        demo_dir = create_demo_signatures()
        results.append(("Demo Signatures", True))
    except Exception as e:
        print(f"❌ Demo signatures creation failed: {e}")
        results.append(("Demo Signatures", False))
    
    # Tổng kết
    print("\n" + "="*50)
    print("📊 TEST RESULTS:")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nSummary: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! The application is ready to use.")
        print("\nTo start the application, run:")
        print("streamlit run app.py")
    else:
        print(f"\n⚠️ {len(results) - passed} tests failed. Please check the issues above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
