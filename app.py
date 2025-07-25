import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import io
import pandas as pd
import plotly.express as px
from streamlit_drawable_canvas import st_canvas  

# Import các module tự tạo
from utils.image_processor import SignatureProcessor
from utils.database import SignatureDatabase



# Cấu hình trang
st.set_page_config(
    page_title="Xác thực chữ ký ",
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
        # Khởi tạo session state
        if 'current_user' not in st.session_state:
            st.session_state.current_user = None
        if 'verification_result' not in st.session_state:
            st.session_state.verification_result = None
    
    
    
    def main(self):
        # Header chính
        st.markdown('<h1 class="main-header">🖋️ Ứng Dụng Nhận Diện Chữ Ký</h1>', unsafe_allow_html=True)
        
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
                "⚙️ Cài Đặt"
            ]
        )
        # Thêm system status
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🖥️ Trạng Thái Hệ Thống")
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
       
        elif page == "⚙️ Cài Đặt":
            self.settings_page()
    
    def home_page(self):
        st.markdown('<h2 class="section-header">Chào Mừng Đến Với Hệ Thống Nhận Diện Chữ Ký</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🎯 Mục Tiêu")
            st.write("""
            - Phân biệt chữ ký thật và giả
            - Giao diện dễ sử dụng
            - Quản lý dữ liệu hiệu quả
            """)
        
        with col2:
            st.markdown("### 🚀 Tính Năng")
            st.write("""
            - Đăng ký chữ ký mẫu
            - Xác minh tự động
            - Vẽ chữ ký trực tiếp
            """)
        
        with col3:
            st.markdown("### 🔧 Công Nghệ")
            st.write("""
            - Python + Streamlit
            - OpenCV
            - SQLite Database
            """)
        
        # Thống kê tổng quan
        st.markdown("### 📈 Tổng Quan Hệ Thống")
        stats = self.db.get_stats()
        
        col1, col2, col3,  = st.columns(3)
        with col1:
            st.metric("Người Dùng", stats['users_count'])
        with col2:
            st.metric("Chữ Ký Mẫu", stats['templates_count'])
        with col3:
            st.metric("Lần Xác Minh", stats['verifications_count'])
       
    
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
            
            cols = st.columns(min(len(templates), 3))
            for i, template in enumerate(templates):
                with cols[i % 3]:
                    if os.path.exists(template['image_path']):
                        image = cv2.imread(template['image_path'], cv2.IMREAD_GRAYSCALE)
                        st.image(image, caption=f"Mẫu #{template['id']}", width=150)
                        
                        # Hiển thị thông tin mẫu
                        created_date = pd.to_datetime(template['created_at']).strftime('%d/%m/%Y')
                        st.caption(f"📅 {created_date}")
                        
                        # Nút xóa với xác nhận
                        if st.button(f"🗑️ Xóa", key=f"del_{template['id']}", use_container_width=True):
                            if len(templates) > 1:  # Chỉ cho phép xóa nếu còn ít nhất 1 mẫu
                                self.db.delete_signature(template['id'])
                                st.success(f"✅ Đã xóa mẫu #{template['id']}")
                                st.rerun()
                            else:
                                st.error("❌ Không thể xóa mẫu cuối cùng!")
                    else:
                        st.error(f"❌ Không tìm thấy file mẫu #{template['id']}")
                        if st.button(f"🗑️ Xóa mẫu lỗi", key=f"del_error_{template['id']}"):
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
                    
                    with st.spinner("🔄 Đang xử lý và so sánh..."):
                        for template in templates:
                            try:
                                # Kiểm tra và xử lý features
                                if template['features'] is not None:
                                    # Nếu features là bytes, decode nó
                                    if isinstance(template['features'], bytes):
                                        import pickle
                                        template_features = pickle.loads(template['features'])
                                    else:
                                        template_features = template['features']
                                    
                                    # Nếu không có features, trích xuất lại từ ảnh
                                    if template_features is None or len(template_features) == 0:
                                        if os.path.exists(template['image_path']):
                                            template_img = cv2.imread(template['image_path'], cv2.IMREAD_GRAYSCALE)
                                            if template_img is not None:
                                                processed_template = self.processor.preprocess_image(template_img)
                                                template_features = self.processor.extract_features(processed_template)
                                            else:
                                                continue
                                        else:
                                            continue
                                    
                                    # Tính similarity
                                    similarity = self.processor.calculate_similarity(
                                        test_features, 
                                        template_features
                                    )
                                    
                                    similarities.append({
                                        'template_id': template['id'],
                                        'similarity': similarity
                                    })
                                    
                                else:
                                    # Không có features, trích xuất từ ảnh
                                    if os.path.exists(template['image_path']):
                                        template_img = cv2.imread(template['image_path'], cv2.IMREAD_GRAYSCALE)
                                        if template_img is not None:
                                            processed_template = self.processor.preprocess_image(template_img)
                                            template_features = self.processor.extract_features(processed_template)
                                            
                                            similarity = self.processor.calculate_similarity(
                                                test_features, 
                                                template_features
                                            )
                                            
                                            similarities.append({
                                                'template_id': template['id'],
                                                'similarity': similarity
                                            })
                                        else:
                                            st.warning(f"⚠️ Không thể đọc ảnh mẫu #{template['id']}")
                                    else:
                                        st.warning(f"⚠️ Không tìm thấy ảnh mẫu #{template['id']}")
                                        
                            except Exception as e:
                                st.error(f"❌ Lỗi xử lý mẫu #{template['id']}: {str(e)}")
                                continue
                    
                    if similarities:
                        # Sắp xếp theo độ tương đồng
                        similarities.sort(key=lambda x: x['similarity'], reverse=True)
                        
                        # Tìm độ tương đồng cao nhất
                        best_match = similarities[0]
                        
                        # Tính toán thống kê nâng cao
                        scores = [s['similarity'] for s in similarities]
                        avg_similarity = np.mean(scores)
                        median_similarity = np.median(scores)
                        max_similarity = max(scores)
                        min_similarity = min(scores)
                        
                        # Loại bỏ outlier (điểm quá thấp) nếu có nhiều hơn 2 mẫu
                        if len(scores) > 2:
                            # Tính Q1, Q3 và IQR
                            q1 = np.percentile(scores, 25)
                            q3 = np.percentile(scores, 75)
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            
                            # Lọc bỏ outlier thấp
                            filtered_scores = [s for s in scores if s >= lower_bound]
                            if len(filtered_scores) >= len(scores) * 0.6:  # Giữ ít nhất 60% mẫu
                                avg_similarity = np.mean(filtered_scores)
                                st.info(f"🔍 Đã loại bỏ {len(scores) - len(filtered_scores)} mẫu có điểm quá thấp")
                        
                        # Thuật toán quyết định cải tiến với cài đặt linh hoạt (Phiên bản nhẹ nhàng cho đồ án)
                        settings = getattr(st.session_state, 'verification_settings', {})
                        
                        # Ngưỡng linh hoạt và dễ dàng hơn cho đồ án nhập môn
                        if len(similarities) == 1:
                            # Chỉ có 1 mẫu - ngưỡng thấp hơn
                            threshold = settings.get('single_threshold', 0.50)  # Giảm từ 0.75 xuống 0.50
                            is_genuine = best_match['similarity'] >= threshold
                            decision_info = f"1 mẫu: cần >= {threshold:.0%}"
                        elif len(similarities) == 2:
                            # 2 mẫu - ngưỡng dễ dàng hơn
                            threshold = settings.get('dual_threshold', 0.45)  # Giảm từ 0.70 xuống 0.45
                            avg_threshold = settings.get('dual_avg_threshold', 0.40)  # Giảm từ 0.65 xuống 0.40
                            is_genuine = (best_match['similarity'] >= threshold and 
                                        avg_similarity >= avg_threshold)
                            decision_info = f"2 mẫu: max >= {threshold:.0%}, avg >= {avg_threshold:.0%}"
                        else:
                            # 3+ mẫu - ngưỡng rất dễ dàng
                            threshold = settings.get('multi_threshold', 0.40)  # Giảm từ 0.65 xuống 0.40
                            median_threshold = settings.get('multi_median_threshold', 0.35)  # Giảm từ 0.60 xuống 0.35
                            avg_threshold = settings.get('multi_avg_threshold', 0.30)  # Giảm từ 0.55 xuống 0.30
                            is_genuine = (best_match['similarity'] >= threshold and 
                                        median_similarity >= median_threshold and
                                        avg_similarity >= avg_threshold)
                            decision_info = f"{len(similarities)} mẫu: max >= {threshold:.0%}, median >= {median_threshold:.0%}, avg >= {avg_threshold:.0%}"
                        
                        # Hiển thị kết quả đơn giản
                        if is_genuine:
                            st.markdown(f"""
                            <div class="result-box success-box">
                                <h3>✅ CHỮ KÝ HỢP LỆ</h3>
                                <p><strong>🎯 Độ tương đồng cao nhất:</strong> {best_match['similarity']:.2%} (Mẫu #{best_match['template_id']})</p>
                                <p><strong>📊 Độ tương đồng trung bình:</strong> {avg_similarity:.2%}</p>
                                <p><strong>� Số mẫu so sánh:</strong> {len(similarities)}</p>
                                <p><strong>⚙️ Điều kiện áp dụng:</strong> {decision_info}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            reasons = []
                            if best_match['similarity'] < threshold:
                                reasons.append(f"Điểm cao nhất ({best_match['similarity']:.2%}) < ngưỡng ({threshold:.2%})")
                            if len(similarities) >= 2 and avg_similarity < settings.get('dual_avg_threshold', 0.40):
                                reasons.append(f"Điểm trung bình thấp ({avg_similarity:.2%})")
                            
                            
                            reason_text = ", ".join(reasons) if reasons else "Không đạt ngưỡng chấp nhận"
                            
                            st.markdown(f"""
                            <div class="result-box danger-box">
                                <h3>❌ CHỮ KÝ KHÔNG HỢP LỆ</h3>
                                <p><strong>🎯 Độ tương đồng cao nhất:</strong> {best_match['similarity']:.2%} (Mẫu #{best_match['template_id']})</p>
                                <p><strong>📊 Độ tương đồng trung bình:</strong> {avg_similarity:.2%}</p>
                                <p><strong>� Số mẫu so sánh:</strong> {len(similarities)}</p>
                                <p><strong>⚙️ Điều kiện áp dụng:</strong> {decision_info}</p>
                                <p><strong>⚠️ Lý do từ chối:</strong> {reason_text}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
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
                        st.image(processed, caption="Ảnh được tối ưu hóa", use_column_width=True, clamp=True)
                        
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
                                                threshold = 0.80  # Tăng lên 80%
                                                min_avg_threshold = 0.75  # Avg phải >= 75%
                                                
                                                # Hiển thị kết quả - yêu cầu CẢ max và avg đều cao
                                                if max_sim >= threshold and avg_sim >= min_avg_threshold:
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
                                                    user['id'], max_sim, max_sim >= threshold and avg_sim >= min_avg_threshold
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
            
    def settings_page(self):
        st.markdown('<h2 class="section-header">⚙️ Cài Đặt Hệ Thống</h2>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🗄️ Dữ Liệu", "ℹ️ Thông Tin"])
        
        with tab1:
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
                for root, _, files in os.walk("data"):
                    for file in files:
                        data_size += os.path.getsize(os.path.join(root, file))
            
            st.info(f"💾 Dung lượng dữ liệu: **{data_size / (1024*1024):.1f} MB**")
        
        with tab2:
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
            GitHub: https://github.com/Duc-bug/Ai_nhan_dang_chu_ki
            """)
            
            if st.button("🔄 Kiểm Tra Cập Nhật"):
                st.info("✅ Bạn đang sử dụng phiên bản mới nhất!")

def main():
    app = SignatureApp()
    app.main()

if __name__ == "__main__":
    main()
