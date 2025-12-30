"""
Two-Stage Heart Disease Prediction Model Training
Fixed for your specific datasets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TWO-STAGE HEART DISEASE PREDICTION MODEL TRAINING")
print("=" * 60)

# ============================================================
# STAGE 1: LIFESTYLE SCREENING MODEL
# ============================================================
print("\n[STAGE 1] Training Lifestyle Screening Model...")

# Load your lifestyle dataset
try:
    print("Loading lifestyle dataset...")
    lifestyle_data = pd.read_csv("C:\\vs code\\cpp programming\\machine-learning-algo\\heart-disease-ai\\updated_version.csv")
    print(f"✓ Loaded {len(lifestyle_data)} samples")
    print(f"✓ Columns: {list(lifestyle_data.columns)}")
except FileNotFoundError:
    print("ERROR: 'updated_version.csv' not found!")
    exit(1)

# Configure for YOUR dataset
LIFESTYLE_FEATURES = [
    'age',
    'sex',
    'total_cholesterol',
    'ldl',
    'hdl',
    'systolic_bp',
    'diastolic_bp',
    'smoking',
    'diabetes'
]

LIFESTYLE_TARGET = 'heart_attack'

print("\nUsing features:", LIFESTYLE_FEATURES)
print("Target column:", LIFESTYLE_TARGET)

# Data preprocessing
print("\nPreprocessing lifestyle data...")

try:
    X_life = lifestyle_data[LIFESTYLE_FEATURES]
    y_life = lifestyle_data[LIFESTYLE_TARGET]
except KeyError as e:
    print(f"ERROR: Column not found: {e}")
    exit(1)

print("\nData types:")
print(X_life.dtypes)

# Handle missing values
print(f"\nMissing values before: {X_life.isnull().sum().sum()}")
if X_life.isnull().sum().sum() > 0:
    X_life = X_life.fillna(X_life.median())
    print(f"Missing values after: {X_life.isnull().sum().sum()}")

print(f"\nTarget distribution:")
print(y_life.value_counts())
print(f"Percentage with heart disease: {y_life.mean()*100:.1f}%")

# Split data
X_life_train, X_life_test, y_life_train, y_life_test = train_test_split(
    X_life, y_life, test_size=0.2, random_state=42, stratify=y_life
)
print(f"\nTraining samples: {len(X_life_train)}")
print(f"Testing samples: {len(X_life_test)}")

# Scale features
scaler_life = StandardScaler()
X_life_train_scaled = scaler_life.fit_transform(X_life_train)
X_life_test_scaled = scaler_life.transform(X_life_test)

# Train lifestyle model
print("\nTraining Random Forest Classifier...")
lifestyle_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
lifestyle_model.fit(X_life_train_scaled, y_life_train)

# Evaluate lifestyle model
y_pred_life = lifestyle_model.predict(X_life_test_scaled)
y_proba_life = lifestyle_model.predict_proba(X_life_test_scaled)[:, 1]

print("\n" + "="*50)
print("LIFESTYLE MODEL PERFORMANCE")
print("="*50)
print(classification_report(y_life_test, y_pred_life, 
                          target_names=['No Heart Attack', 'Heart Attack']))
print(f"ROC-AUC Score: {roc_auc_score(y_life_test, y_proba_life):.4f}")

cm = confusion_matrix(y_life_test, y_pred_life)
print("\nConfusion Matrix:")
print(f"True Negatives:  {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives:  {cm[1,1]}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': LIFESTYLE_FEATURES,
    'importance': lifestyle_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance.to_string(index=False))

# Save lifestyle model
with open('lifestyle_model.pkl', 'wb') as f:
    pickle.dump(lifestyle_model, f)
with open('scaler_life.pkl', 'wb') as f:
    pickle.dump(scaler_life, f)
print("\n✓ Lifestyle model saved")

# ============================================================
# STAGE 2: CLINICAL DIAGNOSIS MODEL
# ============================================================
print("\n" + "=" * 60)
print("[STAGE 2] Training Clinical Diagnosis Model...")

# Load clinical dataset
try:
    print("Loading clinical dataset...")
    clinical_data = pd.read_csv("C:\\vs code\\cpp programming\\machine-learning-algo\\heart-disease-ai\\Heart_Disease_Prediction.csv")
    print(f"✓ Loaded {len(clinical_data)} samples")
    print(f"✓ Columns: {list(clinical_data.columns)}")
except FileNotFoundError:
    print("ERROR: 'Heart_Disease_Prediction.csv' not found!")
    exit(1)

# **FIX: Configure clinical features based on YOUR dataset columns**
CLINICAL_FEATURES = [
    'Age',
    'Sex',
    'Chest pain type',
    'BP',
    'Cholesterol',
    'FBS over 120',
    'EKG results',
    'Max HR',
    'Exercise angina',
    'ST depression',
    'Slope of ST',
    'Number of vessels fluro',
    'Thallium'
]

# **FIX: Use the correct target column name**
CLINICAL_TARGET = 'Heart Disease'

print(f"\nUsing clinical features: {CLINICAL_FEATURES}")
print(f"Using clinical target: {CLINICAL_TARGET}")

# Data preprocessing
print("\nPreprocessing clinical data...")

try:
    X_clin = clinical_data[CLINICAL_FEATURES]
    y_clin = clinical_data[CLINICAL_TARGET]
except KeyError as e:
    print(f"ERROR: Column not found: {e}")
    print("Available columns:", list(clinical_data.columns))
    exit(1)

# Handle missing values
print(f"\nMissing values before: {X_clin.isnull().sum().sum()}")
if X_clin.isnull().sum().sum() > 0:
    X_clin = X_clin.fillna(X_clin.median())
    print(f"Missing values after: {X_clin.isnull().sum().sum()}")

print(f"\nTarget distribution:")
print(y_clin.value_counts())

# Split data
X_clin_train, X_clin_test, y_clin_train, y_clin_test = train_test_split(
    X_clin, y_clin, test_size=0.2, random_state=42, stratify=y_clin
)
print(f"\nTraining samples: {len(X_clin_train)}")
print(f"Testing samples: {len(X_clin_test)}")

# Scale features
scaler_clin = StandardScaler()
X_clin_train_scaled = scaler_clin.fit_transform(X_clin_train)
X_clin_test_scaled = scaler_clin.transform(X_clin_test)

# Train clinical model
print("\nTraining Gradient Boosting Classifier...")
clinical_model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
clinical_model.fit(X_clin_train_scaled, y_clin_train)

# Evaluate clinical model
y_pred_clin = clinical_model.predict(X_clin_test_scaled)
y_proba_clin = clinical_model.predict_proba(X_clin_test_scaled)[:, 1]

print("\n" + "="*50)
print("CLINICAL MODEL PERFORMANCE")
print("="*50)
print(classification_report(y_clin_test, y_pred_clin,
                          target_names=['No Disease', 'Disease']))
print(f"ROC-AUC Score: {roc_auc_score(y_clin_test, y_proba_clin):.4f}")

cm = confusion_matrix(y_clin_test, y_pred_clin)
print("\nConfusion Matrix:")
print(f"True Negatives:  {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives:  {cm[1,1]}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': CLINICAL_FEATURES,
    'importance': clinical_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance.to_string(index=False))

# Save clinical model
with open('clinical_model.pkl', 'wb') as f:
    pickle.dump(clinical_model, f)
with open('scaler_clin.pkl', 'wb') as f:
    pickle.dump(scaler_clin, f)
print("\n✓ Clinical model saved")

# Save feature names
feature_info = {
    'lifestyle_features': LIFESTYLE_FEATURES,
    'clinical_features': CLINICAL_FEATURES,
    'lifestyle_target': LIFESTYLE_TARGET,
    'clinical_target': CLINICAL_TARGET
}

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_info, f)
print("✓ Feature names saved")

# ============================================================
# FUSION TESTING
# ============================================================
print("\n" + "=" * 60)
print("[FUSION] Testing Two-Stage Pipeline...")

alpha = 0.4
min_samples = min(len(y_proba_life), len(y_proba_clin))
P_L = y_proba_life[:min_samples]
P_C = y_proba_clin[:min_samples]
P_final = alpha * P_L + (1 - alpha) * P_C

print(f"\nSample Predictions (First 10):")
print(f"{'#':>3} {'Lifestyle':>12} {'Clinical':>12} {'Final':>12} {'Risk':>15}")
print("-" * 60)
for i in range(min(10, min_samples)):
    risk = "Low" if P_final[i] < 0.3 else ("Moderate" if P_final[i] < 0.6 else "High")
    print(f"{i+1:3d} {P_L[i]:12.3f} {P_C[i]:12.3f} {P_final[i]:12.3f} {risk:>15}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("TRAINING COMPLETE! ✓")
print("=" * 60)
print("\nGenerated Files:")
print("  ✓ lifestyle_model.pkl")
print("  ✓ clinical_model.pkl")
print("  ✓ scaler_life.pkl")
print("  ✓ scaler_clin.pkl")
print("  ✓ feature_names.pkl")
print("\nNext Steps:")
print("  1. Run: streamlit run app.py")
print("  2. Test the application")
print("  3. Deploy to Streamlit Cloud")
print("=" * 60)