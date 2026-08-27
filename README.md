# Support Vector Machine (SVM) AI Projects — Nepal Real-World Datasets

This repository contains two complete Machine Learning projects built using **Support Vector Machine (SVM)** algorithms on authentic datasets from **Nepal**:

1. **[01_svm_regression_nepal_housing/](file:///home/subeshyadav3/Projects/svm/01_svm_regression_nepal_housing)**: Support Vector Regression (SVR) for **Kathmandu Valley House Price Prediction**
2. **[02_svm_classification_nepal_air_quality/](file:///home/subeshyadav3/Projects/svm/02_svm_classification_nepal_air_quality)**: Support Vector Classification (SVC) for **Kathmandu Valley Air Quality & Inversion Risk (2022 - 2025)** using the [Kathmandu AQI Dataset by Subesh Yadav](https://www.kaggle.com/datasets/subeshyadav/kathmandu-aqi-dataset-2022-2025).

---

## 📁 Repository Structure

```
/home/subeshyadav3/Projects/svm/
├── requirements.txt                              # Core dependencies
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
    │   ├── kathmandu_aqi_2022_2025_raw.csv      # 22,489 hourly observations (2022-2025)
    │   └── kathmandu_air_quality.csv            # Cleaned engineered dataset
    ├── plots/
    │   ├── confusion_matrix.png                 # Multi-class confusion matrix
    │   ├── decision_boundary_2d.png             # 2D PCA non-linear decision boundary
    │   └── kernel_comparison.png                # Accuracy & F1-score comparison
    ├── train_svc.py                             # SVC training & GridSearch pipeline
    ├── predict.py                               # Atmospheric risk & health advisory predictor
    ├── svm_from_scratch.py                      # Pure NumPy SVM implementation (No scikit-learn)
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

### 2. Project 1: House Price Regression (SVR)
```bash
cd 01_svm_regression_nepal_housing
python train_svr.py    # Train and evaluate SVR models
python predict.py      # Run sample house price estimation
```

### 3. Project 2: Air Quality Classification (SVC)
```bash
cd 02_svm_classification_nepal_air_quality
python train_svc.py           # Train and evaluate SVC models (scikit-learn)
python svm_from_scratch.py    # Run pure NumPy from-scratch SVM implementation
python predict.py             # Run sample AQI category prediction
```

---

## 🔬 Project Summaries & Performance

### 1. Regression (SVR) — Kathmandu Housing Valuation
- **Task**: Predict total house and land sale prices in Lakhs NPR based on Land Area (Aana / sq ft), Bedrooms, Bathrooms, Floors, Parking, Road Width, and Location.
- **Algorithms Compared**: Linear, RBF, and Polynomial SVR with $\varepsilon$-insensitive loss.
- **Best Model**: Tuned RBF Kernel ($C=1.0, \varepsilon=0.2, \gamma=\text{'scale'}$).

### 2. Classification (SVC) — Kathmandu Air Quality & Inversion Hazard (2022 - 2025)
- **Dataset**: [Kathmandu AQI Dataset 2022-2025](https://www.kaggle.com/datasets/subeshyadav/kathmandu-aqi-dataset-2022-2025) by Subesh Yadav (22,489 hourly records).
- **Task**: Classify hourly atmospheric pollution risk into 4 tiers (`Hazardous_Inversion`, `High_Stagnation`, `Moderate_Dispersion`, `Good_Ventilation`) using temperature, humidity, wind shear (10m & 100m), soil moisture, and seasonal cycles.
- **Algorithms Compared**: Linear, RBF, and Polynomial SVC with One-vs-Rest and balanced class weighting.
- **Best Model**: Tuned RBF Kernel ($C=50.0, \gamma=0.05$) — **97.09% Test Accuracy**, **0.9640 Macro F1-Score**.
- **From-Scratch Version**: Included in `02_svm_classification_nepal_air_quality/svm_from_scratch.py` with pure NumPy subgradient optimization.
