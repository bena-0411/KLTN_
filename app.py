import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

# ==============================================================================
# 1. CẤU HÌNH GIAO DIỆN & TẢI MÔ HÌNH (CÓ AUTO-TRAIN DỰ PHÒNG CHO CLOUD)
# ==============================================================================
st.set_page_config(
    page_title="CDSS - Dự đoán Tái nhập viện Tiểu đường",
    page_icon="🏥",
    layout="wide"
)

def map_icd9(icd9_code):
    if pd.isna(icd9_code) or str(icd9_code).strip() in ['?', '']:
        return 'Missing'
    icd9_code = str(icd9_code).upper().strip()
    if icd9_code.startswith('V') or icd9_code.startswith('E'):
        return 'Other'
    try:
        code_num = float(icd9_code)
        if (390 <= code_num <= 459) or code_num == 785: return 'Circulatory'
        elif (460 <= code_num <= 519) or code_num == 786: return 'Respiratory'
        elif (520 <= code_num <= 579) or code_num == 787: return 'Digestive'
        elif np.floor(code_num) == 250: return 'Diabetes'
        elif 800 <= code_num <= 999: return 'Injury'
        elif 710 <= code_num <= 739: return 'Musculoskeletal'
        elif (580 <= code_num <= 629) or code_num == 788: return 'Genitourinary'
        elif 140 <= code_num <= 239: return 'Neoplasms'
        else: return 'Other'
    except Exception:
        return 'Other'

def calc_lace_L(days):
    if days <= 3: return int(days)
    elif days <= 6: return 4
    elif days <= 13: return 5
    else: return 7

@st.cache_resource
def load_model():
    model_path = "diabetes_readmission_model.pkl"
    if os.path.exists(model_path):
        try:
            loaded = joblib.load(model_path)
            # Kiểm tra xem mô hình có khớp bộ 53 biến mới không
            if hasattr(loaded, "feature_names_in_") and "lace_index_score" in loaded.feature_names_in_:
                return loaded
        except Exception:
            pass

    # --- TỰ ĐỘNG HUẤN LUYỆN DỰ PHÒNG TRÊN STREAMLIT CLOUD NẾU LỆCH PYTHON ---
   df_raw = pd.read_csv("diabetic_data.csv", na_values=['?', ''], keep_default_na=False)
    dead_ids = [11, 13, 14, 19, 20, 21]
    df_cls = df_raw[~df_raw['discharge_disposition_id'].isin(dead_ids)].copy()
    df_cls = df_cls[df_cls['gender'] != 'Unknown/Invalid'].copy()
    df_cls['event'] = np.where(df_cls['readmitted'] == '<30', 1, 0)

    df_cls = df_cls.sort_values(by=['patient_nbr', 'encounter_id']).reset_index(drop=True)
    df_cls['prior_hospitalizations'] = df_cls.groupby('patient_nbr').cumcount()

    cols_to_drop = [
        'encounter_id', 'patient_nbr', 'weight', 'payer_code',
        'medical_specialty', 'examide', 'citoglipton', 'readmitted'
    ]
    df_fe = df_cls.drop(columns=cols_to_drop)

    for col in ['diag_1', 'diag_2', 'diag_3']:
        df_fe[f'{col}_group'] = df_fe[col].apply(map_icd9)
    df_fe = df_fe.drop(columns=['diag_1', 'diag_2', 'diag_3'])

    def count_comorbidities(row):
        diseases = set([row['diag_1_group'], row['diag_2_group'], row['diag_3_group']])
        diseases.discard('Missing')
        diseases.discard('Diabetes')
        return len(diseases)

    df_fe['comorbidity_score'] = df_fe.apply(count_comorbidities, axis=1)

    medication_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'insulin',
        'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    df_fe['total_meds_used'] = (df_fe[medication_cols] != 'No').sum(axis=1)
    df_fe['total_meds_changed'] = ((df_fe[medication_cols] == 'Up') | (df_fe[medication_cols] == 'Down')).sum(axis=1)

    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45,
        '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95
    }
    df_fe['age_mid'] = df_fe['age'].map(age_map).fillna(65)
    df_fe['is_emergency_admission'] = np.where(df_fe['admission_type_id'] == 1, 1, 0)
    df_fe['discharged_to_care_facility'] = np.where(df_fe['discharge_disposition_id'] != 1, 1, 0)

    for id_col in ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']:
        df_fe[id_col] = df_fe[id_col].astype(str)

    df_fe['total_visits_last_year'] = (
        df_fe['number_outpatient'] + df_fe['number_emergency'] + df_fe['number_inpatient']
    )
    df_fe['labs_per_day'] = (df_fe['num_lab_procedures'] / df_fe['time_in_hospital']).round(2)
    df_fe['meds_per_day'] = (df_fe['num_medications'] / df_fe['time_in_hospital']).round(2)

    L_score = df_fe['time_in_hospital'].apply(calc_lace_L)
    A_score = df_fe['is_emergency_admission'] * 3
    C_score = df_fe['comorbidity_score'].clip(upper=3)
    E_score = df_fe['number_emergency'].clip(upper=4)
    df_fe['lace_index_score'] = L_score + A_score + C_score + E_score

    X = df_fe.drop(columns=['event'])
    y = df_fe['event']
    weight_ratio = (y == 0).sum() / (y == 1).sum()

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    rf_base = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=10, class_weight='balanced', random_state=42, n_jobs=-1)
    xgb_base = XGBClassifier(n_estimators=120, learning_rate=0.05, max_depth=4, scale_pos_weight=weight_ratio, eval_metric='logloss', random_state=42, n_jobs=-1)
    lgbm_base = LGBMClassifier(n_estimators=120, learning_rate=0.05, max_depth=5, num_leaves=25, scale_pos_weight=weight_ratio, random_state=42, verbose=-1, n_jobs=-1)

    stacking_clf = StackingClassifier(
        estimators=[('rf', rf_base), ('xgb', xgb_base), ('lgbm', lgbm_base)],
        final_estimator=LogisticRegression(C=1.0, random_state=42),
        cv=3,
        n_jobs=-1
    )

    final_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", stacking_clf)])
    final_pipeline.fit(X, y)
    try:
        joblib.dump(final_pipeline, model_path)
    except Exception:
        pass
    return final_pipeline

model = load_model()

# ==============================================================================
# 2. HÀM VẼ ĐỒNG HỒ ĐO RỦI RO & XÂY DỰNG DỮ LIỆU ĐẦU VÀO
# ==============================================================================
def draw_gauge_chart(probability, title_text="Xác suất Tái nhập viện (<30 ngày)"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': "%", 'font': {'size': 38, 'color': 'black'}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title_text, 'font': {'size': 17}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black", 'thickness': 0.22},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 15], 'color': "rgba(46, 204, 113, 0.8)"},   # Xanh: Nguy cơ thấp
                {'range': [15, 30], 'color': "rgba(241, 196, 15, 0.8)"},  # Vàng: Nguy cơ trung bình
                {'range': [30, 100], 'color': "rgba(231, 76, 60, 0.8)"}   # Đỏ: Nguy cơ cao
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(height=310, margin=dict(l=20, r=20, t=45, b=15))
    return fig

# ==============================================================================
# 3. TIÊU ĐỀ & THANH NHẬP LIỆU BỆNH NHÂN (SIDEBAR)
# ==============================================================================
st.title("🏥 HỆ THỐNG HỖ TRỢ QUYẾT ĐỊNH LÂM SÀNG (CDSS)")
st.markdown("""
<div style='background-color: #e8f4f8; padding: 14px; border-left: 5px solid #2980b9; border-radius: 5px;'>
    <b>Khóa luận Tốt nghiệp:</b> Dự đoán nguy cơ tái nhập viện trong 30 ngày của bệnh nhân Đái tháo đường bằng kiến trúc <b>Stacking Ensemble</b> hiệu chỉnh xác suất, đối chứng thang điểm y khoa <b>LACE Index</b> và Giải thích Nhân quả <b>XAI (SHAP Waterfall)</b>.
</div>
""", unsafe_allow_html=True)

st.sidebar.header("📋 HỒ SƠ LÂM SÀNG BỆNH NHÂN")

age = st.sidebar.selectbox("1. Nhóm tuổi", ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'], index=7)
gender = st.sidebar.selectbox("2. Giới tính", ['Female', 'Male'], format_func=lambda x: "Nữ (Female)" if x == 'Female' else "Nam (Male)")
is_emergency = st.sidebar.selectbox("3. Hình thức nhập viện", [1, 0], format_func=lambda x: "Nhập viện Cấp cứu (Emergency)" if x == 1 else "Nhập viện Thường quy / Theo lịch")
discharged_care = st.sidebar.selectbox("4. Hướng xử trí xuất viện", [0, 1], format_func=lambda x: "Về nhà tự chăm sóc (Home)" if x == 0 else "Chuyển cơ sở điều dưỡng / Y tế khác")

st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Tiền sử & Thời gian điều trị:**")
time_in_hospital = st.sidebar.slider("Số ngày nằm viện đợt này", 1, 14, 4)
number_inpatient = st.sidebar.slider("Số lần nằm viện nội trú (1 năm qua)", 0, 15, 1)
number_emergency = st.sidebar.slider("Số lần cấp cứu (1 năm qua)", 0, 15, 0)
number_outpatient = st.sidebar.slider("Số lần khám ngoại trú (1 năm qua)", 0, 15, 0)
prior_hosp = st.sidebar.slider("Số đợt điều trị trước đó tại viện", 0, 10, 0)

st.sidebar.markdown("---")
st.sidebar.markdown("**🧪 Xét nghiệm & Phác đồ Thuốc:**")
a1c_result = st.sidebar.selectbox("Kết quả HbA1c (A1Cresult)", ['None', 'Norm', '>7', '>8'], format_func=lambda x: "Không chỉ định (None)" if x == 'None' else x)
insulin_status = st.sidebar.selectbox("Phác đồ Insulin", ['No', 'Steady', 'Up', 'Down'], index=1)
diag_1_group = st.sidebar.selectbox("Nhóm bệnh chính (ICD-9)", ['Circulatory', 'Diabetes', 'Respiratory', 'Digestive', 'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other'])
comorbidity_score = st.sidebar.slider("Điểm bệnh lý đồng mắc (0-3)", 0, 3, 2)
num_lab_procedures = st.sidebar.slider("Tổng số xét nghiệm (Lab)", 1, 132, 45)
num_medications = st.sidebar.slider("Tổng số loại thuốc sử dụng", 1, 81, 16)
total_meds_used = st.sidebar.slider("Số thuốc Đái tháo đường đang dùng", 0, 6, 1)
total_meds_changed = st.sidebar.slider("Số thuốc phải tăng/giảm liều", 0, 4, 1)

def create_patient_df(
    _age, _gender, _is_emerg, _disch_care, _days, _inp, _emerg, _outp,
    _prior, _a1c, _insulin, _diag1, _comorb, _labs, _meds, _d_meds, _changed
):
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45,
        '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95
    }
    lace_score = calc_lace_L(_days) + (_is_emerg * 3) + min(3, _comorb) + min(4, _emerg)

    row_dict = {
        'race': 'Caucasian', 'gender': _gender, 'age': _age,
        'admission_type_id': '1' if _is_emerg == 1 else '3',
        'discharge_disposition_id': '3' if _disch_care == 1 else '1',
        'admission_source_id': '7' if _is_emerg == 1 else '1',
        'time_in_hospital': _days,
        'num_lab_procedures': _labs,
        'num_procedures': 1,
        'num_medications': _meds,
        'number_outpatient': _outp,
        'number_emergency': _emerg,
        'number_inpatient': _inp,
        'number_diagnoses': max(3, _comorb + 2),
        'max_glu_serum': 'None',
        'A1Cresult': _a1c,
        'metformin': 'Steady' if _d_meds >= 2 else 'No',
        'repaglinide': 'No', 'nateglinide': 'No', 'chlorpropamide': 'No',
        'glimepiride': 'No', 'acetohexamide': 'No', 'glipizide': 'No', 'glyburide': 'No',
        'tolbutamide': 'No', 'pioglitazone': 'No', 'rosiglitazone': 'No', 'acarbose': 'No',
        'miglitol': 'No', 'troglitazone': 'No', 'tolazamide': 'No',
        'insulin': _insulin,
        'glyburide-metformin': 'No', 'glipizide-metformin': 'No',
        'glimepiride-pioglitazone': 'No', 'metformin-rosiglitazone': 'No',
        'metformin-pioglitazone': 'No',
        'change': 'Ch' if _changed > 0 else 'No',
        'diabetesMed': 'Yes' if (_d_meds > 0 or _insulin != 'No') else 'No',
        'prior_hospitalizations': _prior,
        'diag_1_group': _diag1,
        'diag_2_group': 'Diabetes',
        'diag_3_group': 'Other' if _comorb >= 2 else 'Missing',
        'comorbidity_score': _comorb,
        'total_meds_used': _d_meds,
        'total_meds_changed': _changed,
        'age_mid': age_map.get(_age, 65),
        'is_emergency_admission': _is_emerg,
        'discharged_to_care_facility': _disch_care,
        'total_visits_last_year': _outp + _emerg + _inp,
        'labs_per_day': round(_labs / max(1, _days), 2),
        'meds_per_day': round(_meds / max(1, _days), 2),
        'lace_index_score': lace_score
    }
    return pd.DataFrame([row_dict])

input_df = create_patient_df(
    age, gender, is_emergency, discharged_care, time_in_hospital,
    number_inpatient, number_emergency, number_outpatient, prior_hosp,
    a1c_result, insulin_status, diag_1_group, comorbidity_score,
    num_lab_procedures, num_medications, total_meds_used, total_meds_changed
)

# ==============================================================================
# 4. BA PHÂN HỆ CHỨC NĂNG CHÍNH (3 TABS)
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "🩺 1. Chẩn đoán Cá thể & Giả lập Phác đồ (What-If)",
    "📂 2. Sàng lọc Danh sách Bệnh nhân Hàng loạt",
    "💰 3. Mô phỏng Hiệu quả Kinh tế Bệnh viện (ROI)"
])

# ------------------------------------------------------------------------------
# TAB 1: CHẨN ĐOÁN CÁ THỂ, ĐỐI CHỨNG LACE INDEX, GIẢ LẬP WHAT-IF & SHAP WATERFALL
# ------------------------------------------------------------------------------
with tab1:
    prob_now = float(model.predict_proba(input_df)[0][1])
    lace_val = int(input_df['lace_index_score'].iloc[0])

    m1, m2, m3 = st.columns(3)
    m1.metric("Xác suất Tái nhập viện (AI Stacking)", f"{prob_now*100:.1f}%", delta="Nguy cơ Cao" if prob_now >= 0.20 else "An toàn", delta_color="inverse")
    m2.metric("Điểm Y khoa Truyền thống (LACE Index)", f"{lace_val} / 19 điểm", delta="LACE >= 10 (Nguy cơ cao)" if lace_val >= 10 else "LACE < 10 (Trung bình/Thấp)", delta_color="inverse")
    m3.metric("Tổng lượt Cấp cứu & Nội trú năm qua", f"{int(input_df['total_visits_last_year'].iloc[0])} lượt")

    col_g, col_rec = st.columns([1, 1])
    with col_g:
        st.plotly_chart(draw_gauge_chart(prob_now), use_container_width=True)

    with col_rec:
        st.markdown("#### 💡 Khuyến nghị Lâm sàng & Phác đồ Xử trí:")
        if prob_now >= 0.30:
            st.error("🚨 **CẢNH BÁO ĐỎ (Nguy cơ rất cao >= 30%):**")
            st.markdown("""
            - **Chỉ định giữ viện:** Đề nghị hội chẩn chuyên khoa Nội tiết và Tim mạch, giữ bệnh nhân theo dõi thêm **1–2 ngày** để ổn định đường huyết.
            - **Điều chỉnh thuốc:** Rà soát kỹ phác đồ Insulin, hướng dẫn bệnh nhân kỹ thuật tiêm và tự theo dõi đường huyết mao mạch tại nhà.
            - **Hậu xuất viện:** Bắt buộc kích hoạt gói chăm sóc từ xa (gọi điện kiểm tra sau 48–72 giờ) và hẹn tái khám sớm trong vòng 7 ngày.
            """)
        elif prob_now >= 0.15:
            st.warning("⚠️ **CẢNH BÁO VÀNG (Nguy cơ trung bình 15% – 30%):**")
            st.markdown("""
            - **Chỉ định xuất viện có điều kiện:** Có thể hoàn tất thủ tục xuất viện nhưng cần đưa vào danh sách y tá gọi điện theo dõi sau 3 ngày.
            - **Tư vấn dược lâm sàng:** Kiểm tra sự tuân thủ thuốc của bệnh nhân và dặn dò các dấu hiệu hạ/tăng đường huyết cấp tính.
            """)
        else:
            st.success("✅ **NHÓM AN TOÀN (Nguy cơ thấp < 15%):**")
            st.markdown("""
            - **Chỉ định:** Xuất viện theo quy trình tiêu chuẩn.
            - **Theo dõi định kỳ:** Cấp đơn thuốc ngoại trú và hẹn tái khám định kỳ sau 1 tháng.
            """)

    st.markdown("---")
    st.markdown("### 🔄 Giả lập Can thiệp Lâm sàng (What-If Simulation)")
    st.caption("Thử nghiệm thay đổi kế hoạch điều trị trước khi xuất viện để quan sát mức độ giảm thiểu rủi ro tái nhập viện:")
    w1, w2, w3 = st.columns(3)
    sim_days = w1.slider("Giả lập: Điều chỉnh số ngày nằm viện", 1, 14, time_in_hospital, key="sim_d")
    sim_insulin = w2.selectbox("Giả lập: Ổn định liều Insulin", ['No', 'Steady', 'Up', 'Down'], index=['No', 'Steady', 'Up', 'Down'].index(insulin_status), key="sim_i")
    sim_changed = w3.slider("Giả lập: Số thuốc bị thay đổi liều đột ngột", 0, 4, total_meds_changed, key="sim_c")

    sim_df = create_patient_df(
        age, gender, is_emergency, discharged_care, sim_days,
        number_inpatient, number_emergency, number_outpatient, prior_hosp,
        a1c_result, sim_insulin, diag_1_group, comorbidity_score,
        num_lab_procedures, num_medications, total_meds_used, sim_changed
    )
    prob_sim = float(model.predict_proba(sim_df)[0][1])
    diff_pct = (prob_sim - prob_now) * 100
    st.info(f"📊 **Kết quả Giả lập:** Xác suất tái nhập viện sau điều chỉnh phác đồ là **{prob_sim*100:.1f}%** (Thay đổi: **{diff_pct:+.1f}%** so với ban đầu).")

    st.markdown("---")
    st.markdown("### 🧠 Giải thích Nhân quả XAI (SHAP Waterfall Plot)")
    with st.expander("Bấm để mở rộng Biểu đồ Thác nước SHAP (Màu Đỏ = Tăng rủi ro | Màu Xanh = Giảm rủi ro)", expanded=True):
        try:
            preprocessor = model.named_steps["preprocessor"]
            xgb_model = model.named_steps["model"].named_estimators_["xgb"]

            X_proc = preprocessor.transform(input_df)
            feat_names = [
                str(col).replace('[', '_').replace(']', '_').replace('<', 'lt_').replace('>', 'gt_')
                for col in preprocessor.get_feature_names_out()
            ]
            X_proc_df = pd.DataFrame(X_proc, columns=feat_names)

            explainer = shap.TreeExplainer(xgb_model)
            shap_explanation = explainer(X_proc_df)

            fig, ax = plt.subplots(figsize=(9, 4.5))
            shap.plots.waterfall(shap_explanation[0], max_display=8, show=False)
            plt.title("Bóc tách mức đóng góp của từng yếu tố lâm sàng trên bệnh nhân này", fontsize=11, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Không thể hiển thị biểu đồ SHAP chi tiết: {e}")

# ------------------------------------------------------------------------------
# TAB 2: SÀNG LỌC HỒ SƠ HÀNG LOẠT CHO KHOA NỘI TIẾT (BATCH SCREENING)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("📂 Sàng lọc & Phân tầng Rủi ro Hàng loạt trước Xuất viện")
    st.write("Cho phép Khoa Nội tiết tải lên danh sách bệnh nhân chuẩn bị xuất viện trong ngày (hoặc chạy thử nghiệm trên mẫu dữ liệu bệnh viện) để tự động xếp hạng ưu tiên can thiệp.")

    sample_size = st.slider("Số lượng hồ sơ bệnh nhân trích xuất kiểm thử nhanh:", 10, 100, 25)
    if st.button("🔍 Chạy Sàng lọc Tự động trên Danh sách Bệnh nhân", type="primary"):
        if os.path.exists("diabetic_data.csv"):
            df_batch_raw = pd.read_csv("diabetic_data.csv", na_values=['?', ''], keep_default_na=False, nrows=sample_size * 2)
            df_batch_raw = df_batch_raw[~df_batch_raw['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])].head(sample_size).copy()

            df_b = df_batch_raw.copy()
            df_b['prior_hospitalizations'] = 0
            for col in ['diag_1', 'diag_2', 'diag_3']:
                df_b[f'{col}_group'] = df_b[col].apply(map_icd9)

            def count_comorb(row):
                d = set([row['diag_1_group'], row['diag_2_group'], row['diag_3_group']])
                d.discard('Missing')
                d.discard('Diabetes')
                return len(d)

            df_b['comorbidity_score'] = df_b.apply(count_comorb, axis=1)
            med_cols = [
                'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                'miglitol', 'troglitazone', 'tolazamide', 'insulin',
                'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
                'metformin-rosiglitazone', 'metformin-pioglitazone'
            ]
            df_b['total_meds_used'] = (df_b[med_cols] != 'No').sum(axis=1)
            df_b['total_meds_changed'] = ((df_b[med_cols] == 'Up') | (df_b[med_cols] == 'Down')).sum(axis=1)
            age_map = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95}
            df_b['age_mid'] = df_b['age'].map(age_map).fillna(65)
            df_b['is_emergency_admission'] = np.where(df_b['admission_type_id'] == 1, 1, 0)
            df_b['discharged_to_care_facility'] = np.where(df_b['discharge_disposition_id'] != 1, 1, 0)
            for id_col in ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']:
                df_b[id_col] = df_b[id_col].astype(str)
            df_b['total_visits_last_year'] = df_b['number_outpatient'] + df_b['number_emergency'] + df_b['number_inpatient']
            df_b['labs_per_day'] = (df_b['num_lab_procedures'] / df_b['time_in_hospital']).round(2)
            df_b['meds_per_day'] = (df_b['num_medications'] / df_b['time_in_hospital']).round(2)
            df_b['lace_index_score'] = (
                df_b['time_in_hospital'].apply(calc_lace_L)
                + df_b['is_emergency_admission'] * 3
                + df_b['comorbidity_score'].clip(upper=3)
                + df_b['number_emergency'].clip(upper=4)
            )

            probs = model.predict_proba(df_b)[:, 1]
            result_table = pd.DataFrame({
                "Mã Hồ sơ": df_batch_raw['encounter_id'].values,
                "Mã Bệnh nhân": df_batch_raw['patient_nbr'].values,
                "Độ tuổi": df_batch_raw['age'].values,
                "Số ngày nằm viện": df_batch_raw['time_in_hospital'].values,
                "Điểm LACE": df_b['lace_index_score'].values,
                "Xác suất Tái nhập viện (%)": (probs * 100).round(1),
                "Phân tầng Cảnh báo": np.where(probs >= 0.30, "🚨 ĐỎ (Can thiệp ngay)", np.where(probs >= 0.15, "⚠️ VÀNG (Theo dõi)", "✅ XANH (An toàn)"))
            }).sort_values(by="Xác suất Tái nhập viện (%)", ascending=False).reset_index(drop=True)

            st.dataframe(result_table, use_container_width=True)
        else:
            st.warning("Không tìm thấy file diabetic_data.csv trong thư mục.")

# ------------------------------------------------------------------------------
# TAB 3: MÔ PHỎNG HIỆU QUẢ KINH TẾ BỆNH VIỆN (COST-BENEFIT ROI SIMULATOR)
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("💰 Mô phỏng Bài toán Kinh tế Y tế & Tối ưu Ngân sách")
    st.write("Công cụ hỗ trợ Ban Giám đốc Bệnh viện ước tính ngân sách tiết kiệm được khi áp dụng AI sàng lọc có xét đến tỷ lệ can thiệp lâm sàng thành công thực tế.")

    c1, c2, c3 = st.columns(3)
    cost_pen = c1.number_input("Chi phí điều trị/phạt mỗi ca tái nhập viện (VNĐ)", value=50_000_000, step=5_000_000)
    cost_int = c2.number_input("Chi phí gói chăm sóc dự phòng/ca (VNĐ)", value=2_000_000, step=500_000)
    efficacy = c3.slider("Hiệu quả ngăn ngừa tái nhập viện của gói chăm sóc (%)", 10, 80, 35) / 100.0

    # Số liệu thực nghiệm từ tập Kiểm thử (Test set ~ 19,869 bệnh nhân: TP=1403, FP=6514, FN=860, TN=11092)
    TP, FP, FN, TN = 1403, 6514, 860, 11092
    cost_no_ai = (TP + FN) * cost_pen
    cost_with_ai = (TP + FP) * cost_int + (TP * (1.0 - efficacy) * cost_pen) + (FN * cost_pen)
    net_saved = cost_no_ai - cost_with_ai

    k1, k2, k3 = st.columns(3)
    k1.metric("Chi phí Rủi ro nếu KHÔNG dùng AI", f"{cost_no_ai/1e9:.2f} Tỷ VNĐ")
    k2.metric("Tổng Chi phí khi Áp dụng CDSS AI", f"{cost_with_ai/1e9:.2f} Tỷ VNĐ")
    k3.metric("Ngân sách Tiết kiệm Thực tế", f"{net_saved/1e9:.2f} Tỷ VNĐ", delta="Hiệu quả dương" if net_saved > 0 else "Thâm hụt")
