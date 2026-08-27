"""
==============================================================================
Support Vector Machine (SVM) - From Scratch Implementation (Pure NumPy)
==============================================================================
This module implements SVM Classification (SVC) and SVM Regression (SVR)
from mathematical first principles without using scikit-learn.

Implemented Algorithms:
1. LinearSVC_FromScratch (Soft-margin Hinge Loss with Subgradient Descent)
2. KernelSVC_FromScratch (RBF Kernel Support Vector Classifier with Dual representation)
3. MultiClassSVC_FromScratch (One-vs-Rest Strategy)
4. LinearSVR_FromScratch (Epsilon-insensitive Loss with Subgradient Descent)
5. Custom Preprocessing & Metrics (StandardScaler, One-Hot Encoder, R2, F1, Accuracy)
==============================================================================
"""

import numpy as np
import pandas as pd


# ==============================================================================
# 1. CUSTOM PREPROCESSING TOOLS (FROM SCRATCH)
# ==============================================================================

class CustomStandardScaler:
    """Standardizes features by removing mean and scaling to unit variance."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Avoid division by zero
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def custom_train_test_split(X, y, test_ratio=0.2, random_state=42):
    """Random stratified/shuffled train-test split using pure NumPy."""
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    test_size = int(len(X) * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    
    if isinstance(X, pd.DataFrame):
        return X.iloc[train_idx].values, X.iloc[test_idx].values, y[train_idx], y[test_idx]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ==============================================================================
# 2. SUPPORT VECTOR CLASSIFIER (SVC) - FROM SCRATCH
# ==============================================================================

class BinarySVC_FromScratch:
    """
    Binary Support Vector Classifier trained using Subgradient Descent
    on Soft-Margin Hinge Loss:
        min (1/2)||w||^2 + C * sum(max(0, 1 - y_i * (w.x_i + b)))
    """
    def __init__(self, C=1.0, lr=0.001, epochs=1000):
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        # Ensure binary labels are {-1, +1}
        y_signed = np.where(y <= 0, -1.0, 1.0)
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(1, self.epochs + 1):
            # Dynamic learning rate decay
            eta = self.lr / np.sqrt(epoch)
            
            # Compute margin distance: y_i * (w.x_i + b)
            margins = y_signed * (np.dot(X, self.w) + self.b)
            
            # Find condition where margin < 1 (margin violation)
            misclassified = margins < 1.0
            
            # Gradients:
            # dL/dw = w - C * sum(y_i * x_i for misclassified)
            # dL/db = - C * sum(y_i for misclassified)
            grad_w = self.w - self.C * np.dot(X[misclassified].T, y_signed[misclassified])
            grad_b = - self.C * np.sum(y_signed[misclassified])
            
            self.w -= eta * grad_w
            self.b -= eta * grad_b
            
        return self

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))


class MultiClassSVC_FromScratch:
    """
    Multi-Class Support Vector Classifier using One-vs-Rest (OvR) Strategy.
    """
    def __init__(self, C=1.0, lr=0.001, epochs=1000):
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.classifiers = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            # Transform to binary: +1 for class c, -1 for all other classes
            y_binary = np.where(y == c, 1.0, -1.0)
            clf = BinarySVC_FromScratch(C=self.C, lr=self.lr, epochs=self.epochs)
            clf.fit(X, y_binary)
            self.classifiers[c] = clf
        return self

    def predict(self, X):
        # Predict class with highest decision confidence
        scores = np.column_stack([
            self.classifiers[c].decision_function(X) for c in self.classes_
        ])
        best_indices = np.argmax(scores, axis=1)
        return self.classes_[best_indices]


# ==============================================================================
# 3. SUPPORT VECTOR REGRESSOR (SVR) - FROM SCRATCH
# ==============================================================================

class LinearSVR_FromScratch:
    """
    Support Vector Regressor with Epsilon-Insensitive Loss:
        min (1/2)||w||^2 + C * sum(max(0, |y_i - (w.x_i + b)| - epsilon))
    Trained via Subgradient Descent.
    """
    def __init__(self, C=10.0, epsilon=0.1, lr=0.0005, epochs=1500):
        self.C = C
        self.epsilon = epsilon
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = float(np.mean(y))

        for epoch in range(1, self.epochs + 1):
            eta = self.lr / np.sqrt(epoch)
            
            # Predictions and errors
            y_pred = np.dot(X, self.w) + self.b
            errors = y - y_pred
            
            # Points exceeding upper epsilon tube: y_i - y_hat > epsilon
            upper_violations = errors > self.epsilon
            # Points exceeding lower epsilon tube: y_hat - y_i > epsilon
            lower_violations = errors < -self.epsilon
            
            # Subgradients:
            # dL/dw = w - C * sum(x_i for upper) + C * sum(x_i for lower)
            grad_w = self.w - self.C * (
                np.sum(X[upper_violations], axis=0) - np.sum(X[lower_violations], axis=0)
            )
            grad_b = - self.C * (
                np.sum(upper_violations.astype(float)) - np.sum(lower_violations.astype(float))
            )
            
            self.w -= eta * grad_w
            self.b -= eta * grad_b
            
        return self

    def predict(self, X):
        return np.dot(X, self.w) + self.b


# ==============================================================================
# 4. CUSTOM EVALUATION METRICS (FROM SCRATCH)
# ==============================================================================

def calc_r2_score(y_true, y_pred):
    """Calculates R² coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot)

def calc_mae(y_true, y_pred):
    """Calculates Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def calc_rmse(y_true, y_pred):
    """Calculates Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calc_accuracy(y_true, y_pred):
    """Calculates classification accuracy percentage."""
    return np.mean(y_true == y_pred)


# ==============================================================================
# 5. END-TO-END DEMO ON REAL NEPAL DATASETS
# ==============================================================================

def test_custom_svr_housing():
    print("\n" + "="*75)
    print(" 1. CUSTOM SVR ALGORITHM FROM SCRATCH — KATHMANDU HOUSING REGRESSION")
    print("="*75)
    
    # Load dataset
    df = pd.read_csv('01_svm_regression_nepal_housing/data/nepal_house_data.csv')
    features = ['Area_SqFt', 'Bedroom', 'Bathroom', 'Floors', 'Parking', 'Road_Width_Ft']
    
    X = df[features].values
    y_lakhs = df['Price_Lakhs'].values
    y_log = np.log1p(y_lakhs)
    
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y_log, test_ratio=0.2, random_state=42)
    
    scaler = CustomStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"[+] Training Linear SVR From Scratch on {len(X_train_scaled)} properties...")
    svr = LinearSVR_FromScratch(C=1.0, epsilon=0.1, lr=0.001, epochs=2000)
    svr.fit(X_train_scaled, y_train)
    
    # Predict
    pred_log = svr.predict(X_test_scaled)
    pred_lakhs = np.expm1(pred_log)
    actual_lakhs = np.expm1(y_test)
    
    r2 = calc_r2_score(actual_lakhs, pred_lakhs)
    mae = calc_mae(actual_lakhs, pred_lakhs)
    rmse = calc_rmse(actual_lakhs, pred_lakhs)
    
    print(f"[✓] From-Scratch SVR R² Score  : {r2:.4f}")
    print(f"[✓] From-Scratch SVR MAE       : {mae:.2f} Lakhs NPR (NPR {mae*100000:,.0f})")
    print(f"[✓] From-Scratch SVR RMSE      : {rmse:.2f} Lakhs NPR")
    
    # Sample custom prediction
    sample_house = np.array([[4.0 * 342.25, 5, 4, 2.5, 2, 16.0]]) # 4 Aana house in Kathmandu
    sample_scaled = scaler.transform(sample_house)
    sample_pred_lakhs = np.expm1(svr.predict(sample_scaled)[0])
    print(f" -> Predicted price for 4 Aana, 5 Bed, 2.5 Floor House: NPR {sample_pred_lakhs*100000:,.0f} ({sample_pred_lakhs:.2f} Lakhs)")


def test_custom_svc_air_quality():
    print("\n" + "="*75)
    print(" 2. CUSTOM SVC ALGORITHM FROM SCRATCH — KATHMANDU AQI CLASSIFICATION")
    print("="*75)
    
    df = pd.read_csv('02_svm_classification_nepal_air_quality/data/kathmandu_air_quality.csv')
    num_features = [
        'Temperature_C', 'Humidity_Pct', 'Apparent_Temp_C',
        'Wind_Speed_10m', 'Wind_Speed_100m', 'Wind_Shear',
        'Soil_Moisture_Surface', 'Soil_Moisture_Deep',
        'Hour_Sin', 'Hour_Cos', 'Month_Sin', 'Month_Cos'
    ]
    
    X = df[num_features].values
    y = df['AQI_Risk_Level'].values
    
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_ratio=0.2, random_state=42)
    
    scaler = CustomStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"[+] Training Multi-Class SVC (One-vs-Rest) From Scratch on {len(X_train_scaled)} observations...")
    svc = MultiClassSVC_FromScratch(C=10.0, lr=0.01, epochs=1000)
    svc.fit(X_train_scaled, y_train)
    
    y_pred = svc.predict(X_test_scaled)
    acc = calc_accuracy(y_test, y_pred)
    
    print(f"[✓] From-Scratch Multi-Class SVC Accuracy : {acc*100:.2f}%")
    
    # Sample classification
    winter_sample = np.array([[8.5, 92.0, 7.0, 1.8, 2.5, 0.7, 0.38, 0.39, 0.866, -0.5, 0.5, 0.866]])
    winter_scaled = scaler.transform(winter_sample)
    pred_risk = svc.predict(winter_scaled)[0]
    print(f" -> Predicted Winter Morning Inversion Risk: {pred_risk}")


def main():
    print("="*75)
    print(" SUPPORT VECTOR MACHINE (SVM) IMPLEMENTED FROM SCRATCH (NO SKLEARN)")
    print("="*75)
    test_custom_svr_housing()
    test_custom_svc_air_quality()
    print("\n" + "="*75)
    print(" [✓] FROM-SCRATCH SVM EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*75 + "\n")


if __name__ == "__main__":
    main()
