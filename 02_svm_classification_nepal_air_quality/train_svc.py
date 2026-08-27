"""
==============================================================================
Project 2: Support Vector Classification (SVC)
Task: Kathmandu Valley Air Quality Index (AQI) Classification
==============================================================================
Author: AI Pair Programming Project
Algorithm: Support Vector Classifier (SVC)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Styling
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'


def load_data(data_path="data/kathmandu_air_quality.csv"):
    """Loads the Kathmandu Air Quality dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"[+] Loaded {len(df)} air quality records from {data_path}")
    
    num_features = [
        'Raw Conc.', 'NowCast Conc.', 'PM25_Lag1', 'PM25_Lag3', 'PM25_Roll6h',
        'Hour_Sin', 'Hour_Cos', 'Month_Sin', 'Month_Cos'
    ]
    cat_features = ['Season']
    
    X = df[num_features + cat_features]
    y = df['AQI_Category'].values
    
    return X, y, num_features, cat_features


def build_preprocessor(num_features, cat_features):
    """Constructs column transformer for scaling numericals and encoding categoricals."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
        ]
    )
    return preprocessor


def train_and_compare_kernels(X_train, y_train, X_test, y_test, preprocessor):
    """Compares Linear, RBF, and Poly kernels on air quality classification."""
    kernels = ['linear', 'rbf', 'poly']
    results = {}
    
    print("\n" + "="*65)
    print(" 1. BASELINE KERNEL COMPARISON (SVC)")
    print("="*65)
    
    for kernel in kernels:
        pipeline = Pipeline([
            ('prep', preprocessor),
            ('svc', SVC(kernel=kernel, C=10.0, class_weight='balanced', random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        results[kernel] = {
            'model': pipeline,
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'y_pred': y_pred
        }
        
        print(f" Kernel: {kernel.upper():<7} | Accuracy: {acc*100:6.2f}% | Macro F1: {f1_macro:6.4f} | Weighted F1: {f1_weighted:6.4f}")
        
    return results


def tune_best_svc(X_train, y_train, preprocessor):
    """Optimizes hyperparameters C and gamma for RBF SVC."""
    print("\n" + "="*65)
    print(" 2. HYPERPARAMETER TUNING VIA GRIDSEARCHCV")
    print("="*65)
    
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('svc', SVC(kernel='rbf', class_weight='balanced', random_state=42))
    ])
    
    param_grid = {
        'svc__C': [1.0, 10.0, 50.0],
        'svc__gamma': ['scale', 0.05, 0.1]
    }
    
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    
    print(f"[+] Best SVC Parameters: {grid.best_params_}")
    print(f"[+] Best 3-Fold Cross-Validation Macro F1: {grid.best_score_:.4f}")
    
    return grid.best_estimator_


def generate_plots(results, best_model, X_train, y_train, X_test, y_test, preprocessor, output_dir="plots"):
    """Generates and saves professional classification plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Kernel Comparison Plot
    kernels = list(results.keys())
    accuracies = [results[k]['accuracy'] * 100 for k in kernels]
    f1_macros = [results[k]['f1_macro'] for k in kernels]
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x=kernels, y=accuracies, hue=kernels, palette="Blues_r", legend=False, ax=ax[0])
    ax[0].set_title("SVC Kernel Accuracy (%)", fontsize=13, fontweight='bold')
    ax[0].set_ylabel("Accuracy (%)")
    ax[0].set_xlabel("Kernel")
    ax[0].set_ylim(0, 105)
    for i, v in enumerate(accuracies):
        ax[0].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
        
    sns.barplot(x=kernels, y=f1_macros, hue=kernels, palette="Greens_r", legend=False, ax=ax[1])
    ax[1].set_title("Macro F1-Score (Balances All AQI Classes)", fontsize=13, fontweight='bold')
    ax[1].set_ylabel("Macro F1-Score")
    ax[1].set_xlabel("Kernel")
    ax[1].set_ylim(0, 1.05)
    for i, v in enumerate(f1_macros):
        ax[1].text(i, v + 0.03, f"{v:.3f}", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plot_path1 = os.path.join(output_dir, "kernel_comparison.png")
    plt.savefig(plot_path1, dpi=300)
    plt.close()
    print(f"[+] Saved kernel comparison plot to {plot_path1}")
    
    # 2. Confusion Matrix Heatmap
    y_pred_best = best_model.predict(X_test)
    labels = ['Good', 'Moderate', 'Unhealthy_Sensitive', 'Unhealthy', 'Hazardous']
    # Filter labels present
    unique_labels = [l for l in labels if l in np.unique(y_test)]
    cm = confusion_matrix(y_test, y_pred_best, labels=unique_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title("Kathmandu AQI Classification Confusion Matrix", fontsize=13, fontweight='bold')
    plt.xlabel("Predicted AQI Category", fontsize=11)
    plt.ylabel("Actual AQI Category", fontsize=11)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plot_path2 = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(plot_path2, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrix to {plot_path2}")
    
    # 3. 2D PCA Decision Boundary Visualization
    # Transform features with preprocessor then 2D PCA
    X_train_proc = preprocessor.fit_transform(X_train)
    pca = PCA(n_components=2, random_state=42)
    X_train_pca = pca.fit_transform(X_train_proc)
    
    # Train 2D SVC for visualization
    le = LabelEncoder()
    y_train_num = le.fit_transform(y_train)
    
    svc_2d = SVC(kernel='rbf', C=10.0, class_weight='balanced', random_state=42)
    svc_2d.fit(X_train_pca, y_train_num)
    
    # Create meshgrid
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    
    Z = svc_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(9, 7))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='Spectral')
    scatter = plt.scatter(
        X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_num, cmap='Spectral',
        edgecolors='k', alpha=0.6, s=25
    )
    # Highlight support vectors
    sv = svc_2d.support_vectors_
    plt.scatter(sv[:, 0], sv[:, 1], s=70, facecolors='none', edgecolors='black', linewidths=1.2, label=f'Support Vectors ({len(sv)})')
    
    plt.title("SVM Non-Linear Decision Boundaries (PCA 2D Projection)", fontsize=13, fontweight='bold')
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=11)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plot_path3 = os.path.join(output_dir, "decision_boundary_2d.png")
    plt.savefig(plot_path3, dpi=300)
    plt.close()
    print(f"[+] Saved 2D decision boundary plot to {plot_path3}")


def main():
    print("="*65)
    print(" SUPPORT VECTOR CLASSIFICATION (SVC) — KATHMANDU AIR QUALITY")
    print("="*65)
    
    # 1. Load Data
    X, y, num_features, cat_features = load_data()
    
    # 2. Train-Test Split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Dataset Split: {len(X_train)} Train samples, {len(X_test)} Test samples")
    
    # 3. Build Preprocessing Pipeline
    preprocessor = build_preprocessor(num_features, cat_features)
    
    # 4. Train and Compare Kernels
    results = train_and_compare_kernels(X_train, y_train, X_test, y_test, preprocessor)
    
    # 5. Tune SVR with Grid Search
    best_svc_model = tune_best_svc(X_train, y_train, preprocessor)
    
    # Final Evaluation of Tuned Model
    y_pred_best = best_svc_model.predict(X_test)
    final_acc = accuracy_score(y_test, y_pred_best)
    final_macro_f1 = f1_score(y_test, y_pred_best, average='macro')
    final_weighted_f1 = f1_score(y_test, y_pred_best, average='weighted')
    
    print("\n" + "="*65)
    print(" 3. FINAL TUNED SVC TEST PERFORMANCE")
    print("="*65)
    print(f" Test Accuracy  : {final_acc*100:.2f}%")
    print(f" Macro F1-Score : {final_macro_f1:.4f}")
    print(f" Weighted F1    : {final_weighted_f1:.4f}")
    print("\n Detailed Classification Report:\n")
    print(classification_report(y_test, y_pred_best, digits=4))
    
    # 6. Generate Plots
    generate_plots(results, best_svc_model, X_train, y_train, X_test, y_test, preprocessor)
    
    # 7. Save Best Model
    model_filename = "best_svc_model.joblib"
    joblib.dump(best_svc_model, model_filename)
    print(f"[+] Serialized tuned SVC pipeline to '{model_filename}'")
    print("="*65)


if __name__ == "__main__":
    main()
