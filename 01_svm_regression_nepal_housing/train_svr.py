"""
==============================================================================
Project 1: Support Vector Regression (SVR)
Task: Kathmandu Valley House Price Prediction (Nepal Real Estate)
==============================================================================
Author: AI Pair Programming Project
Algorithm: Support Vector Machine for Regression (SVR)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'


def load_and_preprocess_data(data_path="data/nepal_house_data.csv"):
    """Loads the cleaned Nepal housing dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"[+] Loaded {len(df)} housing records from {data_path}")
    
    # Feature columns and target
    num_features = ['Area_SqFt', 'Bedroom', 'Bathroom', 'Floors', 'Parking', 'Road_Width_Ft']
    cat_features = ['City', 'Face']
    
    X = df[num_features + cat_features]
    # We predict log(Price_Lakhs) to stabilize high variance in real estate pricing
    y_raw = df['Price_Lakhs'].values
    y_log = np.log1p(y_raw)
    
    return X, y_log, y_raw, num_features, cat_features


def build_preprocessor(num_features, cat_features):
    """Creates a ColumnTransformer for numerical scaling and categorical encoding."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
        ]
    )
    return preprocessor


def train_and_compare_kernels(X_train, y_train, X_test, y_test, preprocessor):
    """Trains SVR with Linear, RBF, and Polynomial kernels and evaluates them."""
    kernels = ['linear', 'rbf', 'poly']
    results = {}
    
    print("\n" + "="*60)
    print(" 1. BASELINE KERNEL COMPARISON (SVR)")
    print("="*60)
    
    for kernel in kernels:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('svr', SVR(kernel=kernel, C=10.0, epsilon=0.1))
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred_log = pipeline.predict(X_test)
        
        # Convert back from log scale to Lakhs NPR
        y_pred_lakhs = np.expm1(y_pred_log)
        y_test_lakhs = np.expm1(y_test)
        
        r2 = r2_score(y_test_lakhs, y_pred_lakhs)
        rmse = np.sqrt(mean_squared_error(y_test_lakhs, y_pred_lakhs))
        mae = mean_absolute_error(y_test_lakhs, y_pred_lakhs)
        
        results[kernel] = {
            'model': pipeline,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'y_pred_lakhs': y_pred_lakhs,
            'y_test_lakhs': y_test_lakhs
        }
        
        print(f" Kernel: {kernel.upper():<7} | R² Score: {r2:6.4f} | RMSE: {rmse:7.2f} Lakhs NPR | MAE: {mae:6.2f} Lakhs NPR")
        
    return results


def tune_best_svr(X_train, y_train, preprocessor):
    """Performs Hyperparameter Tuning for RBF SVR using GridSearchCV."""
    print("\n" + "="*60)
    print(" 2. HYPERPARAMETER TUNING VIA GRIDSEARCHCV")
    print("="*60)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('svr', SVR(kernel='rbf'))
    ])
    
    param_grid = {
        'svr__C': [1.0, 10.0, 50.0, 100.0],
        'svr__epsilon': [0.01, 0.05, 0.1, 0.2],
        'svr__gamma': ['scale', 'auto', 0.01, 0.1]
    }
    
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    
    print(f"[+] Best SVR Parameters: {grid.best_params_}")
    print(f"[+] Best Cross-Validation R² Score: {grid.best_score_:.4f}")
    
    return grid.best_estimator_


def generate_plots(results, best_model, X_test, y_test, output_dir="plots"):
    """Generates and saves professional regression evaluation plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Kernel Comparison Plot
    kernels = list(results.keys())
    r2_scores = [results[k]['r2'] for k in kernels]
    mae_scores = [results[k]['mae'] for k in kernels]
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    plot_r2 = [max(r, -0.5) for r in r2_scores]
    sns.barplot(x=kernels, y=plot_r2, hue=kernels, palette="viridis", legend=False, ax=ax[0])
    ax[0].set_title("SVR Kernel Performance ($R^2$ Score)", fontsize=13, fontweight='bold')
    ax[0].set_ylabel("R² Score (Higher is better)")
    ax[0].set_xlabel("Kernel")
    for i, v in enumerate(r2_scores):
        txt = f"{v:.3f}" if v > -1 else "< -1.0"
        ax[0].text(i, max(v, -0.4) + 0.03, txt, ha='center', fontweight='bold')
        
    plot_mae = [min(m, 300) for m in mae_scores]
    sns.barplot(x=kernels, y=plot_mae, hue=kernels, palette="magma", legend=False, ax=ax[1])
    ax[1].set_title("Mean Absolute Error (Lakhs NPR)", fontsize=13, fontweight='bold')
    ax[1].set_ylabel("MAE in Lakhs NPR (Lower is better)")
    ax[1].set_xlabel("Kernel")
    for i, v in enumerate(mae_scores):
        txt = f"{v:.1f}L" if v < 500 else "> 500L"
        ax[1].text(i, min(v, 280) + 5, txt, ha='center', fontweight='bold')
        
    plt.tight_layout()
    plot_path1 = os.path.join(output_dir, "kernel_comparison.png")
    plt.savefig(plot_path1, dpi=300)
    plt.close()
    print(f"[+] Saved kernel comparison plot to {plot_path1}")
    
    # 2. Actual vs. Predicted Plot for Tuned Model
    y_pred_log = best_model.predict(X_test)
    y_pred_lakhs = np.expm1(y_pred_log)
    y_test_lakhs = np.expm1(y_test)
    
    plt.figure(figsize=(8, 7))
    plt.scatter(y_test_lakhs, y_pred_lakhs, alpha=0.6, color="#2b5c8f", edgecolors='k', label='Predicted Properties')
    max_val = max(y_test_lakhs.max(), y_pred_lakhs.max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Ideal Fit Line ($y=x$)')
    
    # SVR Epsilon tolerance visualization
    plt.fill_between([0, max_val], [0, max_val*0.85], [0, max_val*1.15], color='red', alpha=0.1, label='±15% Tolerance Band')
    
    plt.xlabel("Actual Price (in Lakhs NPR)", fontsize=12)
    plt.ylabel("SVR Predicted Price (in Lakhs NPR)", fontsize=12)
    plt.title("Kathmandu House Price: Actual vs. SVR Predicted", fontsize=14, fontweight='bold')
    plt.legend(loc="upper left")
    plt.tight_layout()
    plot_path2 = os.path.join(output_dir, "actual_vs_predicted.png")
    plt.savefig(plot_path2, dpi=300)
    plt.close()
    print(f"[+] Saved actual vs predicted plot to {plot_path2}")
    
    # 3. Residual Distribution Plot
    residuals = y_test_lakhs - y_pred_lakhs
    plt.figure(figsize=(9, 5))
    sns.histplot(residuals, kde=True, color="#2e7d32", bins=30)
    plt.axvline(0, color='red', linestyle='--', lw=2)
    plt.title("SVR Residual Distribution (Error in Lakhs NPR)", fontsize=14, fontweight='bold')
    plt.xlabel("Residual Error (Actual - Predicted) [Lakhs NPR]", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plot_path3 = os.path.join(output_dir, "residuals.png")
    plt.savefig(plot_path3, dpi=300)
    plt.close()
    print(f"[+] Saved residuals plot to {plot_path3}")


def main():
    print("="*60)
    print(" SUPPORT VECTOR REGRESSION (SVR) — KATHMANDU HOUSING")
    print("="*60)
    
    # 1. Load Data
    X, y_log, y_raw, num_features, cat_features = load_and_preprocess_data()
    
    # 2. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.20, random_state=42
    )
    print(f"[+] Dataset Split: {len(X_train)} Train samples, {len(X_test)} Test samples")
    
    # 3. Build Preprocessing Pipeline
    preprocessor = build_preprocessor(num_features, cat_features)
    
    # 4. Compare Kernels
    results = train_and_compare_kernels(X_train, y_train, X_test, y_test, preprocessor)
    
    # 5. Tune SVR with Grid Search
    best_svr_model = tune_best_svr(X_train, y_train, preprocessor)
    
    # Final Evaluation of Tuned Model
    y_pred_best_log = best_svr_model.predict(X_test)
    y_pred_best_lakhs = np.expm1(y_pred_best_log)
    y_test_lakhs = np.expm1(y_test)
    
    final_r2 = r2_score(y_test_lakhs, y_pred_best_lakhs)
    final_rmse = np.sqrt(mean_squared_error(y_test_lakhs, y_pred_best_lakhs))
    final_mae = mean_absolute_error(y_test_lakhs, y_pred_best_lakhs)
    
    print("\n" + "="*60)
    print(" 3. FINAL TUNED SVR TEST PERFORMANCE")
    print("="*60)
    print(f" Best Model R² Score : {final_r2:.4f}")
    print(f" Root Mean Sq Error  : {final_rmse:.2f} Lakhs NPR (NPR {final_rmse*100000:,.0f})")
    print(f" Mean Absolute Error : {final_mae:.2f} Lakhs NPR (NPR {final_mae*100000:,.0f})")
    
    # 6. Generate Plots
    generate_plots(results, best_svr_model, X_test, y_test)
    
    # 7. Save Best Model
    model_filename = "best_svr_model.joblib"
    joblib.dump(best_svr_model, model_filename)
    print(f"[+] Serialized tuned SVR pipeline to '{model_filename}'")
    print("="*60)


if __name__ == "__main__":
    main()
