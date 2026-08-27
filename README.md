# Support Vector Machine (SVM) AI Projects — Nepal Real-World Datasets

This repository contains two complete, clean Machine Learning projects built using **Support Vector Machine (SVM)** algorithms on authentic datasets from **Nepal**:

1. **[01_svm_regression_nepal_housing/](file:///home/subeshyadav3/Projects/svm/01_svm_regression_nepal_housing)**: Support Vector Regression (SVR) for **Kathmandu Valley House Price Prediction**
2. **[02_svm_classification_nepal_air_quality/](file:///home/subeshyadav3/Projects/svm/02_svm_classification_nepal_air_quality)**: Support Vector Classification (SVC) for **Kathmandu Valley Air Quality Index (AQI) Classification**

---

## 📁 Repository Structure

```
/home/subeshyadav3/Projects/svm/
├── requirements.txt                              # Core dependencies
├── run_all.py                                    # Master runner script
├── README.md                                     # Main documentation & theory
│
├── 01_svm_regression_nepal_housing/             # Project 1: SVR Regression
│   ├── data/
│   │   ├── nepal_house_data_raw.csv             # Raw listings
│   │   └── nepal_house_data.csv                 # Cleaned dataset (1,469 samples)
│   ├── plots/
│   │   ├── actual_vs_predicted.png              # Scatter plot with ±15% tolerance tube
│   │   ├── residuals.png                        # Residual distribution
│   │   └── kernel_comparison.png                # Linear vs RBF vs Poly performance
│   ├── train_svr.py                             # SVR training & GridSearch pipeline
│   ├── predict.py                               # Custom property price estimator
│   ├── nepal_housing_svr.ipynb                  # Interactive Jupyter notebook
│   ├── best_svr_model.joblib                    # Serialized trained model
│   └── README.md
│
└── 02_svm_classification_nepal_air_quality/     # Project 2: SVC Classification
    ├── data/
    │   ├── kathmandu_air_quality_raw.csv        # Raw sensor readings
    │   └── kathmandu_air_quality.csv            # Cleaned dataset (8,215 records)
    ├── plots/
    │   ├── confusion_matrix.png                 # Multi-class confusion matrix
    │   ├── decision_boundary_2d.png             # 2D PCA non-linear decision boundary
    │   └── kernel_comparison.png                # Accuracy & F1-score comparison
    ├── train_svc.py                             # SVC training & GridSearch pipeline
    ├── predict.py                               # AQI prediction & health advisory
    ├── kathmandu_air_quality_svc.ipynb          # Interactive Jupyter notebook
    ├── best_svc_model.joblib                    # Serialized trained model
    └── README.md
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Both Projects in One Command
```bash
python run_all.py
```

### 3. Run Individual Projects

#### Project 1: House Price Regression (SVR)
```bash
cd 01_svm_regression_nepal_housing
python train_svr.py    # Train and evaluate SVR models
python predict.py      # Run sample house price estimation
```

#### Project 2: Air Quality Classification (SVC)
```bash
cd 02_svm_classification_nepal_air_quality
python train_svc.py    # Train and evaluate SVC models
python predict.py      # Run sample AQI category prediction
```

---

## 🔬 Project Summaries & Performance

### 1. Regression (SVR) — Kathmandu Housing Valuation
- **Task**: Predict house prices in Lakhs NPR ($1\text{ Lakh} = 100,000\text{ NPR}$) based on Land Area (Aana / sq ft), Bedrooms, Bathrooms, Floors, Parking, Road Width, and Location.
- **Algorithms Compared**: Linear, RBF, and Polynomial SVR with $\varepsilon$-insensitive loss.
- **Best Model**: Tuned RBF Kernel ($C=1.0, \varepsilon=0.2, \gamma=\text{'scale'}$)
- **Results**:
  - Test $R^2$ Score: `0.5672` (CV) / `0.3402` (Holdout Test)
  - Mean Absolute Error (MAE): `115 Lakhs NPR`

### 2. Classification (SVC) — Kathmandu Air Quality Index (AQI)
- **Task**: Classify hourly pollution severity into 5 categories (`Good`, `Moderate`, `Unhealthy_Sensitive`, `Unhealthy`, `Hazardous`) using PM2.5 concentrations, lag features, diurnal cycles, and seasonal indicators.
- **Algorithms Compared**: Linear, RBF, and Polynomial SVC with One-vs-Rest and balanced class weighting.
- **Best Model**: Tuned RBF Kernel ($C=50.0, \gamma=0.05$)
- **Results**:
  - Test Accuracy: **96.59%**
  - Macro F1-Score: **0.9547**
  - Weighted F1-Score: **0.9660**
