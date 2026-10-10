import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

print("1. Dang doc du lieu...")
df_cls = pd.read_csv("diabetic_data.csv", na_values='?')

print("2. Dang tien xu ly dac trung y khoa...")
df_cls = df_cls.drop(columns=['weight', 'payer_code', 'medical_specialty'])
df_cls = df_cls[~df_cls['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])]
df_cls['event'] = np.where(df_cls['readmitted'] == '<30', 1, 0)
df_cls = df_cls.drop(columns=['readmitted'])

def map_icd9(icd9_code):
    if pd.isna(icd9_code) or icd9_code == '?': return 'Missing'
    icd9_code = str(icd9_code).upper()
    if icd9_code.startswith('V') or icd9_code.startswith('E'): return 'Other'
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
    except:
        return 'Other'

df_cls['diag_1_group'] = df_cls['diag_1'].apply(map_icd9)
df_cls['diag_2_group'] = df_cls['diag_2'].apply(map_icd9)
df_cls['diag_3_group'] = df_cls['diag_3'].apply(map_icd9)
df_cls = df_cls.drop(columns=['diag_1', 'diag_2', 'diag_3'])

def count_comorbidities(row):
    diseases = set([row['diag_1_group'], row['diag_2_group'], row['diag_3_group']])
    diseases.discard('Missing')
    diseases.discard('Diabetes')
    return len(diseases)

df_cls['comorbidity_score'] = df_cls.apply(count_comorbidities, axis=1)

medication_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                   'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                   'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                   'miglitol', 'troglitazone', 'tolazamide', 'examide',
                   'citoglipton', 'insulin', 'glyburide-metformin',
                   'glipizide-metformin', 'glimepiride-pioglitazone',
                   'metformin-rosiglitazone', 'metformin-pioglitazone']

df_cls['total_meds_used'] = (df_cls[medication_cols] != 'No').sum(axis=1)
df_cls['total_meds_changed'] = ((df_cls[medication_cols] == 'Up') | (df_cls[medication_cols] == 'Down')).sum(axis=1)

X = df_cls.drop(columns=['event'])
y = df_cls['event']

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("3. Dang huan luyen AI (Stacking Ensemble)... (Doi 1-2 phut nhe)")
rf_base = RandomForestClassifier(n_estimators=50, max_depth=4, class_weight='balanced', random_state=42)
xgb_base = XGBClassifier(n_estimators=50, learning_rate=0.05, max_depth=4, scale_pos_weight=8.0, eval_metric='logloss', random_state=42)
lgbm_base = LGBMClassifier(n_estimators=50, learning_rate=0.05, max_depth=4, scale_pos_weight=8.0, random_state=42, verbose=-1)

stacking_clf = StackingClassifier(
    estimators=[('rf', rf_base), ('xgb', xgb_base), ('lgbm', lgbm_base)],
    final_estimator=LogisticRegression(class_weight='balanced'),
    cv=2
)

final_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", stacking_clf)])
final_pipeline.fit(X, y)

print("4. Dang luu mo hinh...")
joblib.dump(final_pipeline, "diabetes_readmission_model.pkl")
print("HOAN TAT! File .pkl da duoc tao ra chuan 100% cho Windows.")