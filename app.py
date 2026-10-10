import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN & TẢI MÔ HÌNH
# ==========================================
st.set_page_config(page_title="CDSS - Phân tích Y tế", page_icon="🏥", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("diabetes_readmission_model.pkl")

model = load_model()

# ==========================================
# 2. HÀM VẼ ĐỒNG HỒ ĐO RỦI RO (PLOTLY GAUGE)
# ==========================================
def draw_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        number = {'suffix': "%", 'font': {'size': 40, 'color': 'black'}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Tỉ lệ Rủi ro Tái nhập viện", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black", 'thickness': 0.2},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': "rgba(46, 204, 113, 0.8)"},   # Xanh lá: An toàn
                {'range': [30, 60], 'color': "rgba(241, 196, 15, 0.8)"},  # Vàng: Nguy cơ nhẹ
                {'range': [60, 100], 'color': "rgba(231, 76, 60, 0.8)"}], # Đỏ: Nguy hiểm
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100}
        }))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# ==========================================
# 3. TIÊU ĐỀ CHÍNH CỦA WEB
# ==========================================
st.title("🏥 HỆ THỐNG CẢNH BÁO SỚM RỦI RO TÁI NHẬP VIỆN")
st.markdown("""
<div style='background-color: #e8f4f8; padding: 15px; border-left: 5px solid #2980b9; border-radius: 5px;'>
    <b>Công cụ Hỗ trợ Quyết định Lâm sàng (CDSS)</b> - Phân tích Hồ sơ bệnh nhân Đái tháo đường bằng AI (Stacking Ensemble).
</div>
<br>
""", unsafe_allow_html=True)

# ==========================================
# 4. KHU VỰC NHẬP LIỆU (SIDEBAR)
# ==========================================
st.sidebar.header("📋 THÔNG TIN LÂM SÀNG")
st.sidebar.markdown("Vui lòng điền các chỉ số trước khi bệnh nhân xuất viện:")

def user_input_features():
    age = st.sidebar.selectbox("Độ tuổi", ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'], index=6)
    time_in_hospital = st.sidebar.slider("Số ngày lưu viện đợt này", 1, 14, 4)
    num_lab_procedures = st.sidebar.slider("Số lần xét nghiệm", 1, 132, 45)
    num_medications = st.sidebar.slider("Số loại thuốc đang dùng", 1, 81, 18)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Chỉ số Đánh giá Đặc biệt:**")
    comorbidity_score = st.sidebar.slider("Điểm bệnh lý đi kèm (0-3)", 0, 3, 2)
    total_meds_used = st.sidebar.slider("Số loại thuốc Đái tháo đường", 0, 4, 1)
    total_meds_changed = st.sidebar.slider("Số lần đổi phác đồ đợt này", 0, 4, 1)
    
   # Khởi tạo một Dictionary chứa TẤT CẢ các cột (để không bị lỗi Pipeline)
    data = {
        'encounter_id': 0, 'patient_nbr': 0, # <-- ĐÃ THÊM 2 CỘT MÃ ẢO VÀO ĐÂY
        'race': 'Caucasian', 'gender': 'Female', 'age': age,
        'admission_type_id': 1, 'discharge_disposition_id': 1, 'admission_source_id': 7,
        'time_in_hospital': time_in_hospital, 'num_lab_procedures': num_lab_procedures,
        'num_procedures': 0, 'num_medications': num_medications,
        'number_outpatient': 0, 'number_emergency': 0, 'number_inpatient': 0,
        'number_diagnoses': 9, 'max_glu_serum': 'None', 'A1Cresult': 'None',
        'metformin': 'No', 'repaglinide': 'No', 'nateglinide': 'No', 'chlorpropamide': 'No',
        'glimepiride': 'No', 'acetohexamide': 'No', 'glipizide': 'No', 'glyburide': 'No',
        'tolbutamide': 'No', 'pioglitazone': 'No', 'rosiglitazone': 'No', 'acarbose': 'No',
        'miglitol': 'No', 'troglitazone': 'No', 'tolazamide': 'No', 'examide': 'No',
        'citoglipton': 'No', 'insulin': 'No', 'glyburide-metformin': 'No',
        'glipizide-metformin': 'No', 'glimepiride-pioglitazone': 'No',
        'metformin-rosiglitazone': 'No', 'metformin-pioglitazone': 'No',
        'change': 'Ch', 'diabetesMed': 'Yes',
        'diag_1_group': 'Circulatory', 'diag_2_group': 'Diabetes', 'diag_3_group': 'Other',
        'comorbidity_score': comorbidity_score,
        'total_meds_used': total_meds_used,
        'total_meds_changed': total_meds_changed
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Hiển thị nút bấm
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🚀 KÍCH HOẠT AI ĐÁNH GIÁ RỦI RO", use_container_width=True, type="primary")

st.markdown("---")

# ==========================================
# 5. XỬ LÝ DỰ ĐOÁN & TRÌNH BÀY KẾT QUẢ WOW
# ==========================================
if predict_button:
    # Lấy xác suất rơi vào nhãn 1 (Nguy cơ tái nhập viện) từ mô hình Stacking
    prediction_proba = model.predict_proba(input_df)[0][1]
    
    col_res1, col_res2 = st.columns([1, 1])
    
    # Cột 1: Hiển thị Đồng hồ Rủi ro
    with col_res1:
        st.markdown("<h4 style='text-align: center;'>Thước đo Rủi ro</h4>", unsafe_allow_html=True)
        st.plotly_chart(draw_gauge_chart(prediction_proba), use_container_width=True)
        
    # Cột 2: Diễn giải & Phác đồ
    with col_res2:
        st.markdown("<h4>Khuyến nghị Y tế:</h4>", unsafe_allow_html=True)
        if prediction_proba >= 0.6:
            st.error("🚨 **CẢNH BÁO ĐỎ:** Nguy cơ tái nhập viện dưới 30 ngày cực kỳ cao!")
            st.markdown("""
            **Hành động tức thời:**
            - Khẩn trương hội chẩn lại hồ sơ xuất viện.
            - Phân tích kỹ các biến động về đường huyết/liều lượng Insulin gần đây.
            - **Tránh rủi ro tài chính:** Việc giữ bệnh nhân lại theo dõi thêm 1-2 ngày sẽ rẻ hơn rất nhiều so với khoản phạt 50 triệu VNĐ nếu họ phải đi cấp cứu lại.
            """)
        elif prediction_proba >= 0.3:
            st.warning("⚠️ **CẢNH BÁO VÀNG:** Nguy cơ ở mức ranh giới.")
            st.markdown("""
            **Hành động:** 
            - Có thể cho xuất viện nhưng cần kích hoạt quy trình theo dõi tại nhà. 
            - Giao y tá liên hệ hỏi thăm tình trạng sau 3 ngày.
            """)
        else:
            st.success("✅ **AN TOÀN:** Bệnh nhân ổn định, rủi ro rất thấp.")
            st.markdown("""
            **Hành động:** 
            - Hoàn tất thủ tục xuất viện bình thường.
            - Hướng dẫn chế độ dinh dưỡng và cấp đơn thuốc ngoại trú.
            """)

    st.markdown("---")
    
    # Cột hiển thị Explainable AI (SHAP)
    st.markdown("### 🧠 AI XAI: TẠI SAO HỆ THỐNG LẠI DỰ ĐOÁN NHƯ VẬY?")
    with st.expander("Bấm vào đây để xem Phân tích Nhân quả (Dành cho Giám khảo chuyên môn)"):
        st.write("Biểu đồ dưới đây bóc tách mức độ đóng góp của từng chỉ số lâm sàng vào quyết định tăng/giảm rủi ro của AI (Màu đỏ = Làm tăng rủi ro, Màu xanh = Làm giảm rủi ro).")
        
        # Trích xuất Preprocessor và XGBoost từ bên trong Stacking Pipeline để tính SHAP
        preprocessor = model.named_steps["preprocessor"]
        xgb_model = model.named_steps["model"].estimators_[1] # Lấy mô hình XGBoost ở Tầng 1
        
        # Xử lý dữ liệu đầu vào
        X_processed = preprocessor.transform(input_df)
        feature_names = preprocessor.get_feature_names_out()
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
        
        # Vẽ SHAP
        # Làm sạch tên đặc trưng để loại bỏ các ký tự đặc biệt như [, ], < mà SHAP/XGBoost từ chối
        feature_names = [str(col).replace('[', '_').replace(']', '_').replace('<', '_') for col in preprocessor.get_feature_names_out()]
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
        
        # Vẽ SHAP
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_processed_df)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.summary_plot(shap_values, X_processed_df, plot_type="bar", show=False, max_display=7)
        plt.title("Top 7 Yếu tố ảnh hưởng mạnh nhất đến bệnh nhân này", fontsize=12, fontweight='bold')
        st.pyplot(fig)