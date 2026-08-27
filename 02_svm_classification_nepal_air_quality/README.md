# Project 2: Support Vector Classification (SVC) — Kathmandu Valley Air Quality Index (AQI)

## 📌 Project Overview
This project uses **Support Vector Classification (SVC)** to classify air pollution severity levels in Kathmandu Valley according to international EPA / Nepal environmental standards based on real monitoring stations.

## 📂 Dataset Information
- **Source**: Kathmandu Valley Environmental Monitoring Network (US Embassy & Phora Durbar stations).
- **Features**:
  - `Raw Conc.`: PM2.5 particulate concentration ($\mu g/m^3$).
  - `NowCast Conc.`: Dynamic weighted average concentration.
  - `PM25_Lag1`, `PM25_Lag3`: 1-hour and 3-hour lag concentrations.
  - `PM25_Roll6h`: 6-hour rolling average concentration.
  - `Hour_Sin`, `Hour_Cos`: Cyclical encoding of hour of the day.
  - `Month_Sin`, `Month_Cos`: Cyclical encoding of month of the year.
  - `Season`: Winter, Spring, Monsoon, Post-Monsoon.
- **Target (`AQI_Category`)**:
  - `Good` (0 - 12 $\mu g/m^3$)
  - `Moderate` (12.1 - 35.4 $\mu g/m^3$)
  - `Unhealthy_Sensitive` (35.5 - 55.4 $\mu g/m^3$)
  - `Unhealthy` (55.5 - 150.4 $\mu g/m^3$)
  - `Hazardous` (150.5+ $\mu g/m^3$)

## 🧠 SVC Theory & Mathematical Model
Support Vector Classifier constructs an optimal separating hyperplane that maximizes the geometric margin between classes:

$$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \xi_i$$
$$\text{subject to } y_i (w^T \phi(x_i) + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

- **Soft-margin penalty ($C$)**: Balances maximizing margin width and minimizing misclassification slack variables ($\xi_i$).
- **Kernel Trick**: Projects features into high-dimensional Hilbert spaces using the Radial Basis Function:
  $$K(x, x') = \exp(-\gamma \|x - x'\|^2)$$
- **Multi-class Strategy**: Implemented via One-vs-Rest (OvR) and One-vs-One (OvO) hyperplanes with balanced class weighting.

## 🚀 How to Run

### 1. Train Model & Generate Evaluation Plots
```bash
python train_svc.py
```

### 2. Predict Air Quality Category & Health Advisory
```bash
python predict.py
```

### 3. Run Interactive Jupyter Notebook
```bash
jupyter notebook kathmandu_air_quality_svc.ipynb
```

## 📊 Outputs
- `plots/confusion_matrix.png`: Multi-class confusion matrix heatmap.
- `plots/decision_boundary_2d.png`: PCA 2D non-linear decision boundaries & support vector map.
- `plots/kernel_comparison.png`: Accuracy & Macro F1-score comparison (Linear vs RBF vs Poly).
- `best_svc_model.joblib`: Serialized trained pipeline.
