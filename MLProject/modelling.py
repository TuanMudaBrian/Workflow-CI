"""
modelling.py
============
Melatih model Random Forest untuk prediksi Telco Customer Churn
menggunakan MLflow Tracking UI dengan autolog (tanpa tuning).

Author  : Brian Makmur
Dataset : Telco Customer Churn (telco_preprocessing.csv)
Task    : Binary Classification
Level   : Basic
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Konfigurasi
# ─────────────────────────────────────────────
DATA_PATH       = 'telco_preprocessing.csv'
TARGET_COL      = 'Churn'
TEST_SIZE       = 0.2
RANDOM_STATE    = 42
EXPERIMENT_NAME = "Telco_Churn_Classification"

# ─────────────────────────────────────────────
# 1. Load & Persiapkan Data
# ─────────────────────────────────────────────
print("=" * 55)
print("  MODELLING — Telco Customer Churn")
print("  Mode  : MLflow Autolog (tanpa hyperparameter tuning)")
print("  Model : RandomForestClassifier")
print("=" * 55)

df = pd.read_csv(DATA_PATH)

# Bersihkan nama kolom — hindari error MLflow/model
df.columns = (
    df.columns
    .str.replace(' ', '_')
    .str.replace('(', '', regex=False)
    .str.replace(')', '', regex=False)
    .str.replace('-', '_')
)

# Konversi boolean ke int (hasil pd.get_dummies)
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

print(f"\nDataset shape : {df.shape}")
print(f"Target kolom  : {TARGET_COL}")
print(f"Distribusi target:\n{df[TARGET_COL].value_counts().to_string()}")

# Pisahkan fitur dan target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Split train-test (stratified agar proporsi kelas seimbang)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = TEST_SIZE,
    random_state = RANDOM_STATE,
    stratify     = y
)

print(f"\nTrain size : {X_train.shape[0]:,} baris x {X_train.shape[1]} fitur")
print(f"Test size  : {X_test.shape[0]:,} baris x {X_test.shape[1]} fitur")

# ─────────────────────────────────────────────
# 2. MLflow Setup — Autolog & SQLite
# ─────────────────────────────────────────────
# Memaksa sinkronisasi ke file database SQLite lokal laptop
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

# Aktifkan autolog — otomatis log params, metrics, dan model
mlflow.sklearn.autolog(
    log_input_examples  = True,
    log_model_signatures= True,
    log_models          = True,
)

# ─────────────────────────────────────────────
# 3. Training Model
# ─────────────────────────────────────────────
print("\nMulai training ...")

with mlflow.start_run(run_name="RandomForest_Baseline"):

    # Definisikan model tanpa hyperparameter tuning
    model = RandomForestClassifier(
        n_estimators      = 100,
        max_depth         = 10,
        min_samples_split = 5,
        min_samples_leaf  = 2,
        class_weight      = 'balanced',
        random_state      = RANDOM_STATE,
        n_jobs            = -1
    )

    # Fit model — autolog otomatis mencatat params & model
    model.fit(X_train, y_train)

    # Prediksi
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Hitung metrik evaluasi
    acc       = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average='binary')
    precision = precision_score(y_test, y_pred, average='binary')
    recall    = recall_score(y_test, y_pred, average='binary')
    roc_auc   = roc_auc_score(y_test, y_pred_prob)

    # Tampilkan hasil evaluasi
    print("\n=== Hasil Evaluasi (Test Set) ===")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")

print("\n✅ Training selesai! Artefak tersimpan di MLflow.")
print("\n" + "─" * 55)
print("📊 Untuk melihat dashboard MLflow, jalankan:")
print("   mlflow ui")
print("   Lalu buka browser: http://127.0.0.1:5000")
print("─" * 55)