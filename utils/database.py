import sqlite3
import json
import os
from datetime import datetime
import numpy as np

class SignatureDatabase:
    def __init__(self, db_path="database.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """
        Khởi tạo cơ sở dữ liệu
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Bảng người dùng
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Bảng chữ ký
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signatures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                image_path TEXT NOT NULL,
                features TEXT,  -- JSON string của features
                is_template BOOLEAN DEFAULT 0,  -- 1 nếu là mẫu, 0 nếu là test
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Bảng kết quả xác minh
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS verifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                template_signature_id INTEGER,
                test_signature_id INTEGER,
                similarity_score REAL,
                is_genuine BOOLEAN,
                verification_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (template_signature_id) REFERENCES signatures (id),
                FOREIGN KEY (test_signature_id) REFERENCES signatures (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"Đã khởi tạo cơ sở dữ liệu: {self.db_path}")
    
    def add_user(self, name, email=None):
        """
        Thêm người dùng mới
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO users (name, email) VALUES (?, ?)",
                (name, email)
            )
            user_id = cursor.lastrowid
            conn.commit()
            print(f"Đã thêm người dùng: {name} (ID: {user_id})")
            return user_id
        except sqlite3.IntegrityError:
            print(f"Người dùng {name} đã tồn tại")
            return None
        finally:
            conn.close()
    
    def get_user(self, name):
        """
        Lấy thông tin người dùng
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE name = ?", (name,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user[0],
                'name': user[1],
                'email': user[2],
                'created_at': user[3]
            }
        return None
    
    def list_users(self):
        """
        Liệt kê tất cả người dùng
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users ORDER BY name")
        users = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': user[0],
                'name': user[1],
                'email': user[2],
                'created_at': user[3]
            }
            for user in users
        ]
    
    def add_signature(self, user_id, image_path, features=None, is_template=False):
        """
        Thêm chữ ký vào cơ sở dữ liệu
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Chuyển features thành JSON string
        features_json = json.dumps(features.tolist()) if features is not None else None
        
        cursor.execute('''
            INSERT INTO signatures (user_id, image_path, features, is_template)
            VALUES (?, ?, ?, ?)
        ''', (user_id, image_path, features_json, is_template))
        
        signature_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"Đã thêm chữ ký (ID: {signature_id}) cho user {user_id}")
        return signature_id
    
    def get_template_signatures(self, user_id):
        """
        Lấy tất cả chữ ký mẫu của một người dùng
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM signatures 
            WHERE user_id = ? AND is_template = 1
            ORDER BY created_at DESC
        ''', (user_id,))
        
        signatures = cursor.fetchall()
        conn.close()
        
        result = []
        for sig in signatures:
            features = None
            if sig[3]:  # features column
                features = np.array(json.loads(sig[3]))
            
            result.append({
                'id': sig[0],
                'user_id': sig[1],
                'image_path': sig[2],
                'features': features,
                'is_template': bool(sig[4]),
                'created_at': sig[5]
            })
        
        return result
    
    def get_signature(self, signature_id):
        """
        Lấy thông tin một chữ ký cụ thể
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM signatures WHERE id = ?", (signature_id,))
        sig = cursor.fetchone()
        conn.close()
        
        if sig:
            features = None
            if sig[3]:  # features column
                features = np.array(json.loads(sig[3]))
            
            return {
                'id': sig[0],
                'user_id': sig[1],
                'image_path': sig[2],
                'features': features,
                'is_template': bool(sig[4]),
                'created_at': sig[5]
            }
        return None
    
    def save_verification(self, user_id, template_id, test_id, similarity_score, is_genuine):
        """
        Lưu kết quả xác minh
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO verifications 
            (user_id, template_signature_id, test_signature_id, similarity_score, is_genuine)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, template_id, test_id, similarity_score, is_genuine))
        
        verification_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return verification_id
    
    def get_verification_history(self, user_id=None, limit=100):
        """
        Lấy lịch sử xác minh
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute('''
                SELECT v.*, u.name as user_name 
                FROM verifications v
                JOIN users u ON v.user_id = u.id
                WHERE v.user_id = ?
                ORDER BY v.verification_time DESC
                LIMIT ?
            ''', (user_id, limit))
        else:
            cursor.execute('''
                SELECT v.*, u.name as user_name 
                FROM verifications v
                JOIN users u ON v.user_id = u.id
                ORDER BY v.verification_time DESC
                LIMIT ?
            ''', (limit,))
        
        verifications = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': v[0],
                'user_id': v[1],
                'template_signature_id': v[2],
                'test_signature_id': v[3],
                'similarity_score': v[4],
                'is_genuine': bool(v[5]),
                'verification_time': v[6],
                'user_name': v[7]
            }
            for v in verifications
        ]
    
    def delete_signature(self, signature_id):
        """
        Xóa chữ ký
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Lấy thông tin ảnh trước khi xóa
        signature = self.get_signature(signature_id)
        if signature and os.path.exists(signature['image_path']):
            os.remove(signature['image_path'])
        
        cursor.execute("DELETE FROM signatures WHERE id = ?", (signature_id,))
        conn.commit()
        conn.close()
        
        print(f"Đã xóa chữ ký ID: {signature_id}")
    
    def get_stats(self):
        """
        Lấy thống kê cơ sở dữ liệu
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Số người dùng
        cursor.execute("SELECT COUNT(*) FROM users")
        users_count = cursor.fetchone()[0]
        
        # Số chữ ký mẫu
        cursor.execute("SELECT COUNT(*) FROM signatures WHERE is_template = 1")
        templates_count = cursor.fetchone()[0]
        
        # Số lần xác minh
        cursor.execute("SELECT COUNT(*) FROM verifications")
        verifications_count = cursor.fetchone()[0]
        
        # Tỷ lệ chữ ký thật
        cursor.execute("SELECT COUNT(*) FROM verifications WHERE is_genuine = 1")
        genuine_count = cursor.fetchone()[0]
        
        conn.close()
        
        genuine_rate = (genuine_count / verifications_count * 100) if verifications_count > 0 else 0
        
        return {
            'users_count': users_count,
            'templates_count': templates_count,
            'verifications_count': verifications_count,
            'genuine_rate': genuine_rate
        }
    
    def analyze_template_quality(self, user_id, processor):
        """
        Phân tích chất lượng các mẫu chữ ký của một người dùng
        """
        templates = self.get_template_signatures(user_id)
        if len(templates) < 2:
            return None
        
        similarities = []
        quality_scores = []
        
        for i, template1 in enumerate(templates):
            template_similarities = []
            
            for j, template2 in enumerate(templates):
                if i != j and template1['features'] is not None and template2['features'] is not None:
                    try:
                        similarity = processor.calculate_similarity(
                            template1['features'], 
                            template2['features']
                        )
                        template_similarities.append(similarity)
                        
                        if i < j:  # Avoid duplicate pairs
                            similarities.append({
                                'template1_id': template1['id'],
                                'template2_id': template2['id'],
                                'similarity': similarity
                            })
                    except:
                        continue
            
            # Điểm chất lượng của mẫu = độ tương đồng trung bình với các mẫu khác
            if template_similarities:
                avg_similarity = np.mean(template_similarities)
                quality_scores.append({
                    'template_id': template1['id'],
                    'quality_score': avg_similarity,
                    'consistency': np.std(template_similarities)  # Độ nhất quán
                })
        
        return {
            'pairwise_similarities': similarities,
            'quality_scores': quality_scores,
            'overall_consistency': np.mean([s['similarity'] for s in similarities]) if similarities else 0,
            'recommendation': self._get_quality_recommendation(quality_scores, similarities)
        }
    
    def _get_quality_recommendation(self, quality_scores, similarities):
        """
        Đưa ra khuyến nghị về chất lượng mẫu
        """
        if not quality_scores or not similarities:
            return "Không đủ dữ liệu để phân tích"
        
        avg_quality = np.mean([q['quality_score'] for q in quality_scores])
        min_quality = min([q['quality_score'] for q in quality_scores])
        
        if avg_quality > 0.8:
            return "✅ Chất lượng mẫu rất tốt - các mẫu nhất quán cao"
        elif avg_quality > 0.6:
            if min_quality < 0.5:
                return "⚠️ Có mẫu chất lượng kém - nên xem xét loại bỏ mẫu có điểm thấp"
            else:
                return "✅ Chất lượng mẫu tốt"
        else:
            return "❌ Chất lượng mẫu kém - nên đăng ký lại các mẫu chữ ký"
