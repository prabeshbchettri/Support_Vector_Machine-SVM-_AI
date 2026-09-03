"""
==============================================================================
Enhanced Support Vector Classifier with Atmospheric Physics Feature Engineering
Kathmandu Valley Air Quality & Pollution Inversion Risk Classification
==============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, f1_score

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes meteorological & atmospheric dispersion physics features."""
    df = df.copy()
    # 1. Dew point deficit proxy: (100 - RH) / 5 (in °C)
    df['Dew_Point_Deficit'] = (100.0 - df['Humidity_Pct']) / 5.0
    
    # 2. Temperature gradient proxies
    df['Temp_Apparent_Diff'] = df['Temperature_C'] - df['Apparent_Temp_C']
    df['Stability_Proxy'] = df['Temp_Apparent_Diff'] / (df['Wind_Speed_10m'] + 1.0)
    
    # 3. Ventilation & wind transport features
    df['Ventilation_Index'] = df['Wind_Speed_10m'] * (df['Wind_Speed_100m'] + 0.1)
    if 'Wind_Shear' not in df.columns:
        df['Wind_Shear'] = df['Wind_Speed_100m'] - df['Wind_Speed_10m']
    df['Wind_Ratio'] = df['Wind_Speed_100m'] / (df['Wind_Speed_10m'] + 0.5)
        
    # 4. Soil moisture column ratio
    df['Soil_Moisture_Ratio'] = df['Soil_Moisture_Surface'] / (df['Soil_Moisture_Deep'] + 1e-5)
    
    # 5. Cyclical time features
    if 'Hour_Sin' not in df.columns and 'Hour' in df.columns:
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0)
    if 'Month_Sin' not in df.columns and 'Month' in df.columns:
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)
        
    return df

def train_and_save_enhanced_model(data_path="02_svm_classification_nepal_air_quality/data/kathmandu_air_quality.csv", model_path="02_svm_classification_nepal_air_quality/enhanced_svc_model.joblib"):
    if not os.path.exists(data_path):
        data_path = "data/kathmandu_air_quality.csv"
        model_path = "enhanced_svc_model.joblib"
    
    df = pd.read_csv(data_path)
    df = engineer_features(df)
    
    num_features = [
        'Temperature_C', 'Humidity_Pct', 'Apparent_Temp_C',
        'Wind_Speed_10m', 'Wind_Speed_100m', 'Wind_Shear',
        'Soil_Moisture_Surface', 'Soil_Moisture_Deep',
        'Dew_Point_Deficit', 'Temp_Apparent_Diff', 'Stability_Proxy',
        'Ventilation_Index', 'Wind_Ratio', 'Soil_Moisture_Ratio',
        'Hour_Sin', 'Hour_Cos', 'Month_Sin', 'Month_Cos'
    ]
    cat_features = ['Season']
    
    X = df[num_features + cat_features].copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype(str)
    y = np.array(df['AQI_Risk_Level'].values, dtype=str)
    
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]
    
    preprocessor = ColumnTransformer([
        ('num', RobustScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
    ])
    
    # 1. Base Peak Accuracy SVC (C=100, gamma='scale', class_weight='balanced')
    base_svc = SVC(
        kernel='rbf',
        C=100.0,
        gamma='scale',
        class_weight='balanced',
        random_state=42
    )
    
    # 2. Calibrated Probability SVC
    calibrated_svc = CalibratedClassifierCV(estimator=base_svc, method='sigmoid', cv=3)
    
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('svc', calibrated_svc)
    ])
    
    print("[+] Training Enhanced Physics-Aware Calibrated SVC...")
    pipeline.fit(X_train, y_train)
    
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_macro_f1 = f1_score(y_test, y_test_pred, average='macro')
    
    print(f"[+] Train Accuracy : {train_acc*100:.2f}%")
    print(f"[+] Test Accuracy  : {test_acc*100:.2f}% (Macro F1: {test_macro_f1:.4f})")
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred, digits=4))
    
    joblib.dump({
        'pipeline': pipeline,
        'num_features': num_features,
        'cat_features': cat_features,
        'classes': list(pipeline.classes_),
        'metrics': {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'test_macro_f1': test_macro_f1
        }
    }, model_path)
    print(f"[+] Successfully serialized enhanced model bundle to {model_path}")
    return pipeline

if __name__ == '__main__':
    train_and_save_enhanced_model()
