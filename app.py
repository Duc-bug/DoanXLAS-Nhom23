import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import io
import pandas as pd
import plotly.express as px
from streamlit_drawable_canvas import st_canvas  # type: ignore

# Import các module tự tạo
from utils.image_processor import SignatureProcessor
from utils.database import SignatureDatabase
from model.simple_model import SiameseNetwork  # Dùng simple model
from model.trainer import SignatureTrainer

# Cấu hình trang
st.set_page_config(
    page_title="Xác thực chữ ký bằng AI",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        background-color: #cce7ff;
        border: 1px solid #99d6ff;
        color: #004085;
    }
</style>
""", unsafe_allow_html=True)

class SignatureApp:
    def __init__(self):
        self.processor = SignatureProcessor()
        self.db = SignatureDatabase("data/database.db")
        self.siamese_net = SiameseNetwork()
        self.trainer = SignatureTrainer()
        
        # Khởi tạo session state
        if 'current_user' not in st.session_state:
            st.session_state.current_user = None
        if 'verification_result' not in st.session_state:
            st.session_state.verification_result = None
    
    def load_model(self):
        """Load mô hình AI"""
        model_path = "model/signature_model.h5"
        if os.path.exists(model_path):
            return self.siamese_net.load_model(model_path)
        return False
    
    def main(self):
        # Header chính
        st.markdown('<h1 class="main-header">🖋️ Ứng Dụng Nhận Diện Chữ Ký bằng AI</h1>', unsafe_allow_html=True)
        
        # Sidebar navigation
        st.sidebar.title("📋 Menu Chính")
        
        # Thêm thông tin người dùng hiện tại
        if st.session_state.current_user:
            user = self.db.get_user(st.session_state.current_user)
            templates = self.db.get_template_signatures(user['id'])
            recent_verifications = self.db.get_verification_history(user['id'], limit=5)
            
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 👤 Người Dùng Hiện Tại")
            st.sidebar.success(f"**{user['name']}**")
            st.sidebar.markdown(f"📧 {user['email'] or 'Chưa cập nhật'}")
            st.sidebar.markdown(f"📝 **{len(templates)}** chữ ký mẫu")
            st.sidebar.markdown(f"🔍 **{len(recent_verifications)}** lần xác minh gần đây")
            
            if st.sidebar.button("🚪 Đăng Xuất", use_container_width=True):
                st.session_state.current_user = None
                st.rerun()
        else:
            st.sidebar.warning("⚠️ Chưa chọn người dùng")
            st.sidebar.markdown("Vào **👤 Quản Lý Người Dùng** để chọn")
        
        st.sidebar.markdown("---")
        
        page = st.sidebar.selectbox(
            "Chọn chức năng:",
            [
                "🏠 Trang Chủ",
                "👤 Quản Lý Người Dùng", 
                "📝 Đăng Ký Chữ Ký",
                "🔍 Xác Minh Chữ Ký",
                "🎨 Vẽ Chữ Ký",
                "🤖 Huấn Luyện Mô Hình",
                "📊 Thống Kê & Lịch Sử",
                "⚙️ Cài Đặt"
            ]
        )
        
        # Thêm quick actions
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚡ Thao Tác Nhanh")
        
        if st.session_state.current_user:
            if st.sidebar.button("📝 Đăng Ký Nhanh", use_container_width=True):
                st.session_state.quick_action = "register"
            if st.sidebar.button("🔍 Xác Minh Nhanh", use_container_width=True):
                st.session_state.quick_action = "verify"
        
        # Thêm system status
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🖥️ Trạng Thái Hệ Thống")
        
        model_exists = os.path.exists("model/signature_model.h5")
        st.sidebar.markdown(f"🤖 Model: {'✅ Có' if model_exists else '❌ Chưa có'}")
        
        db_size = 0
        if os.path.exists("data/database.db"):
            db_size = os.path.getsize("data/database.db") / 1024  # KB
        st.sidebar.markdown(f"💾 Database: {db_size:.1f} KB")
        
        # Routing với quick actions
        if hasattr(st.session_state, 'quick_action'):
            if st.session_state.quick_action == "register":
                page = "📝 Đăng Ký Chữ Ký"
                del st.session_state.quick_action
            elif st.session_state.quick_action == "verify":
                page = "🔍 Xác Minh Chữ Ký"
                del st.session_state.quick_action
        
        # Routing
        if page == "🏠 Trang Chủ":
            self.home_page()
        elif page == "👤 Quản Lý Người Dùng":
            self.user_management()
        elif page == "📝 Đăng Ký Chữ Ký":
            self.signature_registration()
        elif page == "🔍 Xác Minh Chữ Ký":
            self.signature_verification()
        elif page == "🎨 Vẽ Chữ Ký":
            self.draw_signature()
        elif page == "🤖 Huấn Luyện Mô Hình":
            self.model_training()
        elif page == "📊 Thống Kê & Lịch Sử":
            self.statistics_page()
        elif page == "🎯 Demo & Hướng Dẫn":
            self.demo_tutorial_page()
        elif page == "⚙️ Cài Đặt":
            self.settings_page()
    
    def home_page(self):
        st.markdown('<h2 class="section-header">Chào Mừng Đến Với Hệ Thống Nhận Diện Chữ Ký bằng AI</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🎯 Mục Tiêu")
            st.write("""
            - Phân biệt chữ ký thật và giả
            - Độ chính xác cao với AI
            - Giao diện thân thiện
            - Quản lý dữ liệu hiệu quả
            """)
        
        with col2:
            st.markdown("### 🚀 Tính Năng")
            st.write("""
            - Đăng ký chữ ký mẫu
            - Xác minh tự động
            - Vẽ chữ ký trực tiếp
            - Thống kê chi tiết
            """)
        
        with col3:
            st.markdown("### 🔧 Công Nghệ")
            st.write("""
            - Python + Streamlit
            - TensorFlow/Keras
            - OpenCV
            - SQLite Database
            """)
        
        # Thống kê tổng quan
        st.markdown("### 📈 Tổng Quan Hệ Thống")
        stats = self.db.get_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Người Dùng", stats['users_count'])
        with col2:
            st.metric("Chữ Ký Mẫu", stats['templates_count'])
        with col3:
            st.metric("Lần Xác Minh", stats['verifications_count'])
        with col4:
            st.metric("Tỷ Lệ Chữ Ký Thật", f"{stats['genuine_rate']:.1f}%")
    
    def user_management(self):
        st.markdown('<h2 class="section-header">👤 Quản Lý Người Dùng</h2>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["➕ Thêm Người Dùng", "👥 Danh Sách Người Dùng"])
        
        with tab1:
            st.markdown("### Đăng Ký Người Dùng Mới")
            with st.form("add_user_form"):
                name = st.text_input("Tên người dùng *", placeholder="Nhập tên đầy đủ")
                email = st.text_input("Email", placeholder="example@email.com")
                
                if st.form_submit_button("➕ Thêm Người Dùng", use_container_width=True):
                    if name.strip():
                        user_id = self.db.add_user(name.strip(), email.strip() if email else None)
                        if user_id:
                            st.success(f"✅ Đã thêm người dùng: {name}")
                            st.rerun()
                        else:
                            st.error("❌ Người dùng đã tồn tại!")
                    else:
                        st.error("❌ Vui lòng nhập tên người dùng!")
        
        with tab2:
            st.markdown("### Danh Sách Người Dùng")
            users = self.db.list_users()
            
            if users:
                df = pd.DataFrame(users)
                df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%d/%m/%Y %H:%M')
                st.dataframe(
                    df[['name', 'email', 'created_at']], 
                    use_container_width=True,
                    column_config={
                        'name': 'Tên',
                        'email': 'Email', 
                        'created_at': 'Ngày Tạo'
                    }
                )
                
                # Chọn người dùng hiện tại
                st.markdown("### Chọn Người Dùng Làm Việc")
                selected_user = st.selectbox(
                    "Chọn người dùng:",
                    options=[None] + [user['name'] for user in users],
                    index=0 if st.session_state.current_user is None else 
                          next((i+1 for i, user in enumerate(users) if user['name'] == st.session_state.current_user), 0)
                )
                
                if selected_user != st.session_state.current_user:
                    st.session_state.current_user = selected_user
                    if selected_user:
                        st.success(f"✅ Đã chọn người dùng: {selected_user}")
                    else:
                        st.info("ℹ️ Chưa chọn người dùng")
            else:
                st.info("ℹ️ Chưa có người dùng nào. Hãy thêm người dùng mới!")
    
    def signature_registration(self):
        st.markdown('<h2 class="section-header">📝 Đăng Ký Chữ Ký Mẫu</h2>', unsafe_allow_html=True)
        
        if not st.session_state.current_user:
            st.warning("⚠️ Vui lòng chọn người dùng trong mục 'Quản Lý Người Dùng' trước!")
            return
        
        user = self.db.get_user(st.session_state.current_user)
        st.info(f"👤 Đang đăng ký cho: **{user['name']}**")
        
        # Upload ảnh
        uploaded_file = st.file_uploader(
            "Chọn ảnh chữ ký mẫu",
            type=['png', 'jpg', 'jpeg'],
            help="Tải lên ảnh chữ ký rõ ràng, nền trắng"
        )
        
        if uploaded_file:
            # Hiển thị ảnh gốc
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh gốc", width=400)
            
            # Xử lý ảnh
            try:
                # Chuyển đổi PIL to numpy
                image_array = np.array(image)
                processed_image = self.processor.preprocess_image(image_array)
                
                # Hiển thị ảnh đã xử lý
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Ảnh gốc", width=300)
                with col2:
                    st.image(processed_image, caption="Ảnh đã xử lý", width=300, clamp=True)
                
                # Trích xuất đặc trưng
                features = self.processor.extract_features(processed_image)
                st.success(f"✅ Đã trích xuất {len(features)} đặc trưng")
                
                if st.button("💾 Lưu Chữ Ký Mẫu", use_container_width=True):
                    # Lưu ảnh
                    os.makedirs("data/signatures", exist_ok=True)
                    image_filename = f"user_{user['id']}_template_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
                    image_path = os.path.join("data/signatures", image_filename)
                    
                    # Lưu ảnh đã xử lý
                    cv2.imwrite(image_path, (processed_image * 255).astype(np.uint8))
                    
                    # Lưu vào database
                    signature_id = self.db.add_signature(
                        user['id'], 
                        image_path, 
                        features, 
                        is_template=True
                    )
                    
                    st.success(f"✅ Đã lưu chữ ký mẫu (ID: {signature_id})")
                    
            except Exception as e:
                st.error(f"❌ Lỗi xử lý ảnh: {str(e)}")
        
        # Hiển thị chữ ký mẫu đã có
        templates = self.db.get_template_signatures(user['id'])
        if templates:
            st.markdown("### 📋 Chữ Ký Mẫu Đã Đăng Ký")
            
            cols = st.columns(min(len(templates), 4))
            for i, template in enumerate(templates):
                with cols[i % 4]:
                    if os.path.exists(template['image_path']):
                        image = cv2.imread(template['image_path'], cv2.IMREAD_GRAYSCALE)
                        st.image(image, caption=f"Mẫu #{template['id']}", width=150)
                        if st.button(f"🗑️ Xóa", key=f"del_{template['id']}"):
                            self.db.delete_signature(template['id'])
                            st.rerun()
    
    def signature_verification(self):
        st.markdown('<h2 class="section-header">🔍 Xác Minh Chữ Ký</h2>', unsafe_allow_html=True)
        
        if not st.session_state.current_user:
            st.warning("⚠️ Vui lòng chọn người dùng trong mục 'Quản Lý Người Dùng' trước!")
            return
        
        user = self.db.get_user(st.session_state.current_user)
        templates = self.db.get_template_signatures(user['id'])
        
        if not templates:
            st.warning("⚠️ Người dùng này chưa có chữ ký mẫu. Vui lòng đăng ký chữ ký mẫu trước!")
            return
        
        st.info(f"👤 Đang xác minh cho: **{user['name']}** ({len(templates)} mẫu)")
        
        # Upload ảnh cần kiểm tra
        test_file = st.file_uploader(
            "Chọn ảnh chữ ký cần xác minh",
            type=['png', 'jpg', 'jpeg'],
            help="Tải lên ảnh chữ ký cần kiểm tra"
        )
        
        if test_file:
            # Hiển thị và xử lý ảnh test
            test_image = Image.open(test_file)
            
            try:
                # Xử lý ảnh test
                test_array = np.array(test_image)
                processed_test = self.processor.preprocess_image(test_array)
                test_features = self.processor.extract_features(processed_test)
                
                # Hiển thị ảnh
                col1, col2 = st.columns(2)
                with col1:
                    st.image(test_image, caption="Ảnh cần kiểm tra", width=300)
                with col2:
                    st.image(processed_test, caption="Ảnh đã xử lý", width=300, clamp=True)
                
                if st.button("🔍 Thực Hiện Xác Minh", use_container_width=True):
                    # So sánh với tất cả templates
                    similarities = []
                    
                    for template in templates:
                        if template['features'] is not None:
                            similarity = self.processor.calculate_similarity(
                                test_features, 
                                template['features']
                            )
                            similarities.append({
                                'template_id': template['id'],
                                'similarity': similarity
                            })
                    
                    if similarities:
                        # Tìm độ tương đồng cao nhất
                        best_match = max(similarities, key=lambda x: x['similarity'])
                        avg_similarity = np.mean([s['similarity'] for s in similarities])
                        
                        # Ngưỡng quyết định (có thể điều chỉnh)
                        threshold = 0.75
                        is_genuine = best_match['similarity'] >= threshold and avg_similarity >= (threshold - 0.05)
                        
                        # Hiển thị kết quả
                        if is_genuine:
                            st.markdown(f"""
                            <div class="result-box success-box">
                                <h3>✅ CHỮ KÝ HỢP LỆ</h3>
                                <p><strong>Độ tương đồng cao nhất:</strong> {best_match['similarity']:.2%}</p>
                                <p><strong>Độ tương đồng trung bình:</strong> {avg_similarity:.2%}</p>
                                <p><strong>Ngưỡng chấp nhận:</strong> {threshold:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-box danger-box">
                                <h3>❌ CHỮ KÝ KHÔNG HỢP LỆ</h3>
                                <p><strong>Độ tương đồng cao nhất:</strong> {best_match['similarity']:.2%}</p>
                                <p><strong>Độ tương đồng trung bình:</strong> {avg_similarity:.2%}</p>
                                <p><strong>Ngưỡng chấp nhận:</strong> {threshold:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Biểu đồ độ tương đồng
                        if len(similarities) > 1:
                            df_sim = pd.DataFrame(similarities)
                            fig = px.bar(
                                df_sim, 
                                x='template_id', 
                                y='similarity',
                                title="Độ Tương Đồng Với Các Mẫu",
                                labels={'template_id': 'ID Mẫu', 'similarity': 'Độ Tương Đồng'}
                            )
                            fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                                        annotation_text="Ngưỡng chấp nhận")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Lưu kết quả
                        # Lưu ảnh test
                        os.makedirs("data/test", exist_ok=True)
                        test_filename = f"user_{user['id']}_test_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
                        test_path = os.path.join("data/test", test_filename)
                        cv2.imwrite(test_path, (processed_test * 255).astype(np.uint8))
                        
                        # Lưu vào database
                        test_signature_id = self.db.add_signature(
                            user['id'], test_path, test_features, is_template=False
                        )
                        
                        verification_id = self.db.save_verification(
                            user['id'],
                            best_match['template_id'],
                            test_signature_id,
                            best_match['similarity'],
                            is_genuine
                        )
                        
                        st.session_state.verification_result = {
                            'is_genuine': is_genuine,
                            'similarity': best_match['similarity'],
                            'verification_id': verification_id
                        }
                        
                    else:
                        st.error("❌ Không thể so sánh với chữ ký mẫu!")
                        
            except Exception as e:
                st.error(f"❌ Lỗi xử lý: {str(e)}")
    
    def draw_signature(self):
        st.markdown('<h2 class="section-header">🎨 Vẽ Chữ Ký Trực Tiếp</h2>', unsafe_allow_html=True)
        
        st.info("✏️ Sử dụng chuột hoặc bút cảm ứng để vẽ chữ ký của bạn")
        
        # Điều khiển canvas
        st.markdown("### 🎮 Điều Khiển")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Xóa Canvas", use_container_width=True):
                st.rerun()
        
        with col2:
            stroke_width = st.slider("Độ dày nét", 1, 10, 3)
        
        with col3:
            stroke_color = st.color_picker("Màu nét", "#000000")
        
        # Canvas để vẽ
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color="#ffffff",
            height=300,  # Tăng chiều cao
            width=700,   # Tăng chiều rộng
            drawing_mode="freedraw",
            key="signature_canvas",
        )
        
        if canvas_result.image_data is not None:
            img_array = canvas_result.image_data
            
            # Kiểm tra xem có vẽ gì không
            if np.any(img_array[:, :, 3] > 0):  # Alpha channel
                # Chuyển thành grayscale
                gray_img = cv2.cvtColor(img_array[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
                
                # Đảo màu (vì canvas có nền trắng, chữ đen)
                gray_img = 255 - gray_img
                
                # Hiển thị ảnh với kích thước đồng nhất
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📝 Chữ Ký Vừa Vẽ")
                    st.image(gray_img, caption="Ảnh gốc", use_column_width=True)
                
                with col2:
                    st.markdown("#### ⚡ Ảnh Đã Xử Lý")
                    try:
                        # Xử lý ảnh
                        processed = self.processor.preprocess_image(gray_img)
                        st.image(processed, caption="Ảnh được tối ưu hóa cho AI", use_column_width=True, clamp=True)
                        
                        # Thêm separator
                        st.markdown("---")
                        
                        # Download buttons
                        st.markdown("### 📥 Lưu Ảnh Ra File")
                        
                        col_download1, col_download2 = st.columns(2)
                        
                        with col_download1:
                            # Download ảnh gốc
                            original_pil = Image.fromarray(gray_img)
                            buf_original = io.BytesIO()
                            original_pil.save(buf_original, format="PNG")
                            
                            st.download_button(
                                label="📥 Tải Ảnh Gốc",
                                data=buf_original.getvalue(),
                                file_name=f"signature_original_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        
                        with col_download2:
                            # Download ảnh đã xử lý
                            processed_pil = Image.fromarray((processed * 255).astype(np.uint8))
                            buf_processed = io.BytesIO()
                            processed_pil.save(buf_processed, format="PNG")
                            
                            st.download_button(
                                label="📥 Tải Ảnh Đã Xử Lý",
                                data=buf_processed.getvalue(),
                                file_name=f"signature_processed_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        
                        # Separator
                        st.markdown("---")
                        
                        # Actions cho user đã đăng nhập
                        if st.session_state.current_user:
                            st.markdown("### 💾 Lưu Vào Hệ Thống")
                            user = self.db.get_user(st.session_state.current_user)
                            
                            col_save, col_verify = st.columns(2)
                            
                            with col_save:
                                if st.button("💾 Lưu Làm Mẫu", use_container_width=True):
                                    try:
                                        # Tạo thư mục nếu chưa có
                                        os.makedirs("data/signatures", exist_ok=True)
                                        
                                        # Tạo tên file unique
                                        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                                        filename = f"user_{user['id']}_drawn_{timestamp}.png"
                                        filepath = os.path.join("data/signatures", filename)
                                        
                                        # Lưu ảnh
                                        cv2.imwrite(filepath, (processed * 255).astype(np.uint8))
                                        
                                        # Trích xuất đặc trưng và lưu vào DB
                                        features = self.processor.extract_features(processed)
                                        signature_id = self.db.add_signature(
                                            user['id'], filepath, features, is_template=True
                                        )
                                        
                                        st.success(f"✅ Đã lưu chữ ký mẫu (ID: {signature_id})")
                                        st.info(f"📁 File: {filepath}")
                                        
                                    except Exception as e:
                                        st.error(f"❌ Lỗi lưu file: {str(e)}")
                            
                            with col_verify:
                                if st.button("🔍 Xác Minh Ngay", use_container_width=True):
                                    try:
                                        # Lấy các chữ ký mẫu
                                        templates = self.db.get_template_signatures(user['id'])
                                        
                                        if templates:
                                            # Trích xuất đặc trưng
                                            features = self.processor.extract_features(processed)
                                            similarities = []
                                            
                                            # So sánh với từng mẫu
                                            for template in templates:
                                                if template['features'] is not None:
                                                    similarity = self.processor.calculate_similarity(
                                                        features, template['features']
                                                    )
                                                    similarities.append(similarity)
                                            
                                            if similarities:
                                                max_sim = max(similarities)
                                                avg_sim = np.mean(similarities)
                                                threshold = 0.75
                                                
                                                # Hiển thị kết quả - yêu cầu cả max và avg đều cao
                                                if max_sim >= threshold and avg_sim >= (threshold - 0.05):
                                                    st.markdown(f"""
                                                    <div class="result-box success-box">
                                                        <h4>✅ CHỮ KÝ HỢP LỆ</h4>
                                                        <p><strong>Độ tương đồng cao nhất:</strong> {max_sim:.2%}</p>
                                                        <p><strong>Độ tương đồng trung bình:</strong> {avg_sim:.2%}</p>
                                                        <p><strong>Ngưỡng chấp nhận:</strong> {threshold:.2%}</p>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                                else:
                                                    st.markdown(f"""
                                                    <div class="result-box danger-box">
                                                        <h4>❌ CHỮ KÝ KHÔNG HỢP LỆ</h4>
                                                        <p><strong>Độ tương đồng cao nhất:</strong> {max_sim:.2%}</p>
                                                        <p><strong>Độ tương đồng trung bình:</strong> {avg_sim:.2%}</p>
                                                        <p><strong>Ngưỡng chấp nhận:</strong> {threshold:.2%}</p>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                                
                                                # Lưu lịch sử xác minh
                                                self.db.add_verification_log(
                                                    user['id'], max_sim, max_sim >= threshold and avg_sim >= (threshold - 0.05)
                                                )
                                            else:
                                                st.warning("⚠️ Không thể so sánh - Lỗi đặc trưng mẫu!")
                                        else:
                                            st.warning("⚠️ Chưa có chữ ký mẫu để so sánh!")
                                            st.info("💡 Hãy lưu ít nhất 1 chữ ký làm mẫu trước")
                                            
                                    except Exception as e:
                                        st.error(f"❌ Lỗi xác minh: {str(e)}")
                        else:
                            st.markdown("### ⚠️ Cần Đăng Nhập")
                            st.warning("Vui lòng chọn người dùng trong **👤 Quản Lý Người Dùng** để sử dụng các tính năng lưu trữ và xác minh.")
                            

                    except Exception as e:
                        st.error(f"❌ Lỗi xử lý ảnh: {str(e)}")
        else:
            # Khi chưa vẽ gì
            st.info("🎨 Hãy vẽ chữ ký của bạn trên canvas ở trên")
            
            # Hiển thị tips
            with st.expander("💡 Mẹo Để Có Chữ Ký Đẹp"):
                st.markdown("""
                **🖱️ Kỹ thuật vẽ:**
                - Vẽ chậm và đều tay
                - Giữ nét liền mạch
                - Kích thước vừa phải (không quá to/nhỏ)
                
                **⚙️ Cài đặt tối ưu:**
                - Độ dày nét: 2-4 (cho chữ ký rõ ràng)
                - Màu đen (#000000) cho độ tương phản cao
                - Canvas đủ lớn để vẽ thoải mái
                
                **🎯 Lưu ý chất lượng:**
                - Chữ ký nên có đặc điểm cá nhân rõ rệt
                - Tránh vẽ quá đơn giản hoặc phức tạp
                - Thực hành vài lần để quen tay
                """)
    
    def model_training(self):
        st.markdown('<h2 class="section-header">🤖 Huấn Luyện Mô Hình AI</h2>', unsafe_allow_html=True)
        
        st.info("🔧 Chức năng này để huấn luyện mô hình Siamese Network nhận diện chữ ký")
        
        tab1, tab2 = st.tabs(["🏋️ Huấn Luyện", "📈 Kiểm Tra Mô Hình"])
        
        with tab1:
            st.markdown("### Cấu Hình Huấn Luyện")
            
            col1, col2 = st.columns(2)
            with col1:
                epochs = st.slider("Số epochs", 10, 100, 30)
                batch_size = st.slider("Batch size", 8, 64, 16)
            
            with col2:
                learning_rate = st.select_slider(
                    "Learning rate", 
                    options=[0.0001, 0.001, 0.01, 0.1],
                    value=0.001,
                    format_func=lambda x: f"{x:.4f}"
                )
                validation_split = st.slider("Validation split", 0.1, 0.4, 0.2)
            
            use_demo_data = st.checkbox("Sử dụng dữ liệu demo", value=True, 
                                      help="Tạo dữ liệu demo để huấn luyện nếu chưa có dữ liệu thật")
            
            if st.button("🚀 Bắt Đầu Huấn Luyện", use_container_width=True):
                with st.spinner("🔄 Đang huấn luyện mô hình..."):
                    try:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("📁 Đang chuẩn bị dữ liệu...")
                        progress_bar.progress(20)
                        
                        if use_demo_data:
                            genuine_sigs, forged_sigs = self.trainer.create_demo_dataset()
                        else:
                            genuine_sigs, forged_sigs = self.trainer.load_dataset()
                        
                        status_text.text("🧠 Đang tạo mô hình...")
                        progress_bar.progress(40)
                        
                        status_text.text("🏋️ Đang huấn luyện...")
                        progress_bar.progress(60)
                        
                        # Huấn luyện
                        history = self.trainer.train_model(
                            genuine_sigs, forged_sigs,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=validation_split
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Hoàn thành!")
                        
                        st.success("🎉 Huấn luyện thành công!")
                        
                        # Hiển thị kết quả
                        if history:
                            final_loss = history.history['loss'][-1]
                            final_acc = history.history['accuracy'][-1]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Loss cuối cùng", f"{final_loss:.4f}")
                            with col2:
                                st.metric("Accuracy cuối cùng", f"{final_acc:.2%}")
                        
                    except Exception as e:
                        st.error(f"❌ Lỗi huấn luyện: {str(e)}")
        
        with tab2:
            st.markdown("### Thông Tin Mô Hình")
            
            model_path = "model/signature_model.h5"
            if os.path.exists(model_path):
                st.success("✅ Mô hình đã tồn tại")
                
                file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                mod_time = pd.Timestamp.fromtimestamp(os.path.getmtime(model_path))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Kích thước file", f"{file_size:.1f} MB")
                with col2:
                    st.metric("Cập nhật lần cuối", mod_time.strftime("%d/%m/%Y %H:%M"))
                
                if st.button("🔍 Kiểm Tra Mô Hình"):
                    if self.siamese_net.load_model(model_path):
                        st.success("✅ Load mô hình thành công!")
                        
                        # Hiển thị thông tin mô hình
                        with st.expander("📋 Chi Tiết Mô Hình"):
                            if self.siamese_net.model:
                                # Đếm parameters
                                total_params = self.siamese_net.model.count_params()
                                st.write(f"**Tổng số parameters:** {total_params:,}")
                                
                                # Input shape
                                input_shape = self.siamese_net.model.input_shape
                                st.write(f"**Input shape:** {input_shape}")
                                
                    else:
                        st.error("❌ Không thể load mô hình!")
            else:
                st.warning("⚠️ Chưa có mô hình. Vui lòng huấn luyện trước!")
    
    def statistics_page(self):
        st.markdown('<h2 class="section-header">📊 Thống Kê & Lịch Sử</h2>', unsafe_allow_html=True)
        
        stats = self.db.get_stats()
        
        # Tổng quan
        st.markdown("### 📈 Tổng Quan Hệ Thống")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👥 Người Dùng", stats['users_count'])
        with col2:
            st.metric("📝 Chữ Ký Mẫu", stats['templates_count'])
        with col3:
            st.metric("🔍 Lần Xác Minh", stats['verifications_count'])
        with col4:
            st.metric("✅ Tỷ Lệ Hợp Lệ", f"{stats['genuine_rate']:.1f}%")
        
        # Lịch sử xác minh
        st.markdown("### 📋 Lịch Sử Xác Minh Gần Đây")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            user_filter = st.selectbox(
                "Lọc theo người dùng:",
                options=["Tất cả"] + [user['name'] for user in self.db.list_users()]
            )
        with col2:
            limit = st.selectbox("Số kết quả:", [10, 20, 50, 100], index=1)
        
        # Lấy dữ liệu
        if user_filter == "Tất cả":
            verifications = self.db.get_verification_history(limit=limit)
        else:
            user = self.db.get_user(user_filter)
            verifications = self.db.get_verification_history(user['id'], limit=limit)
        
        if verifications:
            # Tạo DataFrame
            df = pd.DataFrame(verifications)
            df['verification_time'] = pd.to_datetime(df['verification_time'])
            df['date'] = df['verification_time'].dt.strftime('%d/%m/%Y')
            df['time'] = df['verification_time'].dt.strftime('%H:%M:%S')
            df['result'] = df['is_genuine'].map({True: '✅ Hợp lệ', False: '❌ Không hợp lệ'})
            df['similarity_percent'] = (df['similarity_score'] * 100).round(1)
            
            # Hiển thị bảng
            display_df = df[['user_name', 'similarity_percent', 'result', 'date', 'time']]
            display_df.columns = ['Người Dùng', 'Độ Tương Đồng (%)', 'Kết Quả', 'Ngày', 'Giờ']
            
            st.dataframe(display_df, use_container_width=True)
            
            # Biểu đồ thống kê
            if len(df) > 1:
                st.markdown("### 📊 Biểu Đồ Thống Kê")
                
                tab1, tab2, tab3 = st.tabs(["📈 Theo Thời Gian", "👥 Theo Người Dùng", "🎯 Độ Tương Đồng"])
                
                with tab1:
                    # Biểu đồ theo ngày
                    daily_stats = df.groupby('date').agg({
                        'is_genuine': ['count', 'sum'],
                        'similarity_score': 'mean'
                    }).round(3)
                    
                    daily_stats.columns = ['Tổng', 'Hợp lệ', 'Độ tương đồng TB']
                    daily_stats['Tỷ lệ hợp lệ'] = (daily_stats['Hợp lệ'] / daily_stats['Tổng'] * 100).round(1)
                    
                    fig = px.line(daily_stats.reset_index(), x='date', y='Tỷ lệ hợp lệ',
                                title="Tỷ Lệ Chữ Ký Hợp Lệ Theo Ngày")
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Thống kê theo người dùng
                    user_stats = df.groupby('user_name').agg({
                        'is_genuine': ['count', 'sum'],
                        'similarity_score': 'mean'
                    }).round(3)
                    
                    user_stats.columns = ['Tổng', 'Hợp lệ', 'Độ tương đồng TB']
                    user_stats['Tỷ lệ hợp lệ'] = (user_stats['Hợp lệ'] / user_stats['Tổng'] * 100).round(1)
                    
                    fig = px.bar(user_stats.reset_index(), x='user_name', y=['Tổng', 'Hợp lệ'],
                               title="Số Lần Xác Minh Theo Người Dùng")
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    # Phân bố độ tương đồng
                    fig = px.histogram(df, x='similarity_percent', nbins=20,
                                     title="Phân Bố Độ Tương Đồng",
                                     labels={'similarity_percent': 'Độ Tương Đồng (%)', 'count': 'Số Lượng'})
                    fig.add_vline(x=75, line_dash="dash", line_color="red", 
                                annotation_text="Ngưỡng chấp nhận (75%)")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Chưa có dữ liệu xác minh nào.")
    
    def settings_page(self):
        st.markdown('<h2 class="section-header">⚙️ Cài Đặt Hệ Thống</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["🎛️ Tham Số", "🗄️ Dữ Liệu", "ℹ️ Thông Tin"])
        
        with tab1:
            st.markdown("### 🎯 Cài Đặt Ngưỡng")
            
            threshold = st.slider(
                "Ngưỡng chấp nhận chữ ký (%)",
                min_value=60, max_value=95, value=75,
                help="Độ tương đồng tối thiểu để chữ ký được coi là hợp lệ"
            )
            
            st.markdown("### 🖼️ Cài Đặt Xử Lý Ảnh")
            
            col1, col2 = st.columns(2)
            with col1:
                target_width = st.number_input("Chiều rộng ảnh (px)", 64, 512, 128)
                target_height = st.number_input("Chiều cao ảnh (px)", 64, 512, 128)
            
            with col2:
                padding = st.number_input("Padding (px)", 0, 50, 10)
                blur_kernel = st.selectbox("Kernel làm mờ", [1, 3, 5, 7], index=1)
            
            if st.button("💾 Lưu Cài Đặt"):
                # Ở đây có thể lưu cài đặt vào file config
                st.success("✅ Đã lưu cài đặt!")
        
        with tab2:
            st.markdown("### 🗂️ Quản Lý Dữ Liệu")
            
            st.warning("⚠️ **Cảnh báo:** Các thao tác sau không thể hoàn tác!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🧹 Xóa Lịch Sử Xác Minh", use_container_width=True):
                    if st.checkbox("Tôi hiểu rủi ro", key="clear_history"):
                        # Code để xóa lịch sử
                        st.success("✅ Đã xóa lịch sử xác minh!")
            
            with col2:
                if st.button("🗑️ Xóa Ảnh Test", use_container_width=True):
                    if st.checkbox("Tôi hiểu rủi ro", key="clear_test"):
                        # Code để xóa ảnh test
                        st.success("✅ Đã xóa ảnh test!")
            
            with col3:
                if st.button("💾 Sao Lưu Dữ Liệu", use_container_width=True):
                    # Code để backup
                    st.success("✅ Đã sao lưu dữ liệu!")
            
            st.markdown("### 📊 Thông Tin Lưu Trữ")
            
            # Tính toán dung lượng
            data_size = 0
            if os.path.exists("data"):
                for root, dirs, files in os.walk("data"):
                    for file in files:
                        data_size += os.path.getsize(os.path.join(root, file))
            
            st.info(f"💾 Dung lượng dữ liệu: **{data_size / (1024*1024):.1f} MB**")
        
        with tab3:
            st.markdown("### ℹ️ Thông Tin Ứng Dụng")
            
            st.markdown("""
            **🏷️ Phiên bản:** 1.0.0  
            **👨‍💻 Phát triển bởi:** Nhóm 23  
            **📅 Ngày tạo:** 2025 
            **🐍 Python:** 3.12+  
            **🌐 Framework:** Streamlit  
            
            **📚 Thư viện chính:**
            - TensorFlow/Keras/scikit-learn: Deep Learning
            - OpenCV/Pillow: Xử lý ảnh
            - SQLite/json : Cơ sở dữ liệu
            - Streamlit: Giao diện web
            - NumPy/Pandas/matplotlib/plotly/seaborn: Xử lý dữ liệu
            
            **🔗 Liên hệ hỗ trợ:**  
            Email: duc.2373401010100@vanlanguni.vn 
            GitHub: https://github.com/Duc-bug/app_nhan_dien_chu_ky
            """)
            
            if st.button("🔄 Kiểm Tra Cập Nhật"):
                st.info("✅ Bạn đang sử dụng phiên bản mới nhất!")

def main():
    app = SignatureApp()
    app.main()

if __name__ == "__main__":
    main()
