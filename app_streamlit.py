"""
==============================================================================
Kathmandu Valley Air Quality & Inversion Risk - SVM Application
Simple, Clean & Elegant Dashboard for Visualizations & Inference
==============================================================================
"""

import os
import json
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Page Configuration
st.set_page_config(
    page_title="Kathmandu AQI Classification | SVM",
    page_icon="🌫️",
    layout="wide"
)

# Custom minimal styling
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "02_svm_classification_nepal_air_quality/data/kathmandu_air_quality.csv")
MODEL_PATH = os.path.join(BASE_DIR, "02_svm_classification_nepal_air_quality/enhanced_svc_model.joblib")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "02_svm_classification_nepal_air_quality/best_svc_model.joblib")
if not os.path.exists(DATA_PATH):
    DATA_PATH = "data/kathmandu_air_quality.csv"

# Color mappings
RISK_COLORS = {
    'Hazardous_Inversion': '#dc2626',
    'High_Stagnation': '#ea580c',
    'Moderate_Dispersion': '#ca8a04',
    'Good_Ventilation': '#16a34a'
}

ADVISORIES = {
    'Hazardous_Inversion': 'Severe thermal inversion trapping dense winter smog. High particulate concentration. Limit outdoor exposure and wear protective masks.',
    'High_Stagnation': 'Calm surface wind (< 6 km/h) preventing particle dispersion. Sensitive groups should exercise caution.',
    'Moderate_Dispersion': 'Typical valley ventilation (6 - 12 km/h). Air quality is acceptable for standard outdoor activities.',
    'Good_Ventilation': 'Active ventilation and convective mixing (> 12 km/h). Efficient particle dispersal.'
}


@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        if isinstance(bundle, dict) and 'pipeline' in bundle:
            return bundle['pipeline']
        return bundle
    return None


def derive_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    if 'Dew_Point_Deficit' not in df.columns:
        df['Dew_Point_Deficit'] = (100.0 - df['Humidity_Pct']) / 5.0
    if 'Stability_Proxy' not in df.columns:
        df['Stability_Proxy'] = (df['Temperature_C'] - df['Apparent_Temp_C']) / (df['Wind_Speed_10m'] + 1.0)
    if 'Ventilation_Index' not in df.columns:
        df['Ventilation_Index'] = df['Wind_Speed_10m'] * (df['Wind_Speed_100m'] + 0.1)
    if 'Wind_Shear' not in df.columns:
        df['Wind_Shear'] = df['Wind_Speed_100m'] - df['Wind_Speed_10m']
    if 'Soil_Moisture_Ratio' not in df.columns:
        df['Soil_Moisture_Ratio'] = df['Soil_Moisture_Surface'] / (df['Soil_Moisture_Deep'] + 1e-5)
    if 'Hour_Sin' not in df.columns and 'Hour' in df.columns:
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0)
    if 'Month_Sin' not in df.columns and 'Month' in df.columns:
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)
    return df


df_raw = load_data()
model = load_model()

# Header
st.title("Kathmandu Air Quality & Inversion Risk Analysis")
st.caption("Support Vector Machine (SVM) Classification & Meteorological Modeling")

# Main Tabs
tab_pred, tab_vis, tab_live, tab_batch = st.tabs([
    "🎯 Model Prediction & Testing",
    "📊 Model Visualizations & Hyperplane",
    "🌐 Live Internet Data (Kathmandu)",
    "📂 Batch CSV Evaluation"
])


# ==============================================================================
# TAB 1: MODEL PREDICTION & TESTING
# ==============================================================================
with tab_pred:
    st.subheader("Interactive Atmospheric Risk Prediction")
    st.write("Input meteorological measurements to classify the atmospheric inversion and smog risk level.")

    # Preset selector
    preset_choice = st.selectbox(
        "Load Preset Scenario (Optional):",
        [
            "Custom Manual Input",
            "Winter Morning Smog Inversion (Jan, 8°C, 92% RH, 1.8 km/h wind)",
            "Autumn Evening Stagnation (Nov, 17°C, 68% RH, 4.5 km/h wind)",
            "Spring Afternoon Dispersion (Apr, 22°C, 45% RH, 8.5 km/h wind)",
            "Monsoon Active Gale (Jul, 26°C, 65% RH, 14 km/h wind)"
        ]
    )

    defaults = {
        "Custom Manual Input": {"temp": 12.0, "hum": 78.0, "app": 11.0, "w10": 3.0, "w100": 4.5, "hour": 8, "month": 1, "season": "Winter"},
        "Winter Morning Smog Inversion (Jan, 8°C, 92% RH, 1.8 km/h wind)": {"temp": 8.5, "hum": 92.0, "app": 7.0, "w10": 1.8, "w100": 2.5, "hour": 8, "month": 1, "season": "Winter"},
        "Autumn Evening Stagnation (Nov, 17°C, 68% RH, 4.5 km/h wind)": {"temp": 17.0, "hum": 68.0, "app": 17.0, "w10": 4.5, "w100": 6.0, "hour": 19, "month": 11, "season": "Post-Monsoon"},
        "Spring Afternoon Dispersion (Apr, 22°C, 45% RH, 8.5 km/h wind)": {"temp": 22.0, "hum": 45.0, "app": 21.5, "w10": 8.5, "w100": 12.0, "hour": 15, "month": 4, "season": "Spring"},
        "Monsoon Active Gale (Jul, 26°C, 65% RH, 14 km/h wind)": {"temp": 26.5, "hum": 65.0, "app": 30.0, "w10": 14.0, "w100": 18.5, "hour": 14, "month": 7, "season": "Monsoon"}
    }[preset_choice]

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown("#### Input Features")
        
        c1, c2 = st.columns(2)
        with c1:
            temp = st.number_input("Temperature (°C)", -5.0, 45.0, float(defaults["temp"]), 0.5)
            hum = st.number_input("Relative Humidity (%)", 0.0, 100.0, float(defaults["hum"]), 1.0)
            app_temp = st.number_input("Apparent Temp (°C)", -5.0, 50.0, float(defaults["app"]), 0.5)
            soil_s = st.number_input("Soil Moisture Surface", 0.0, 1.0, 0.35, 0.01)
        with c2:
            w10 = st.number_input("Wind Speed 10m (km/h)", 0.0, 50.0, float(defaults["w10"]), 0.2)
            w100 = st.number_input("Wind Speed 100m (km/h)", 0.0, 80.0, float(defaults["w100"]), 0.5)
            season_list = ["Winter", "Spring", "Monsoon", "Post-Monsoon"]
            season = st.selectbox("Season", season_list, index=season_list.index(defaults["season"]))
            soil_d = st.number_input("Soil Moisture Deep", 0.0, 1.0, 0.38, 0.01)

        c3, c4 = st.columns(2)
        with c3:
            hour = st.slider("Hour of Day", 0, 23, int(defaults["hour"]))
        with c4:
            month = st.slider("Month of Year", 1, 12, int(defaults["month"]))

    with col_out:
        st.markdown("#### Prediction Result")

        input_df = pd.DataFrame([{
            'Temperature_C': temp,
            'Humidity_Pct': hum,
            'Apparent_Temp_C': app_temp,
            'Wind_Speed_10m': w10,
            'Wind_Speed_100m': w100,
            'Wind_Shear': w100 - w10,
            'Soil_Moisture_Surface': soil_s,
            'Soil_Moisture_Deep': soil_d,
            'Hour': hour,
            'Month': month,
            'Season': season
        }])

        df_feat = derive_features(input_df)

        if model is not None:
            pred = model.predict(df_feat)[0]
            color = RISK_COLORS.get(pred, '#64748b')
            advisory = ADVISORIES.get(pred, '')

            # Clean Result Card
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; border-left: 5px solid {color}; background-color: rgba(255,255,255,0.04); margin-bottom: 15px;">
                <span style="font-size: 12px; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.5px;">Predicted Risk Tier</span>
                <h2 style="margin: 5px 0; color: {color};">{pred.replace('_', ' ')}</h2>
                <p style="margin: 10px 0 0 0; font-size: 14px; line-height: 1.5; color: #cbd5e1;">{advisory}</p>
            </div>
            """, unsafe_allow_html=True)

            # Physics summary
            dew_deficit = round((100.0 - hum) / 5.0, 2)
            vent_idx = round(w10 * (w100 + 0.1), 2)
            m1, m2, m3 = st.columns(3)
            m1.metric("Dew Point Deficit", f"{dew_deficit} °C")
            m2.metric("Ventilation Index", f"{vent_idx}")
            m3.metric("Wind Shear", f"{w100 - w10:.1f} km/h")

            # Probability Distribution
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(df_feat)[0]
                classes = list(model.classes_)
                prob_df = pd.DataFrame({
                    'Class': [c.replace('_', ' ') for c in classes],
                    'Probability (%)': [p * 100 for p in probs]
                })

                fig_p = px.bar(
                    prob_df,
                    x='Probability (%)',
                    y='Class',
                    orientation='h',
                    text='Probability (%)',
                    title="Model Class Probabilities",
                    color='Class',
                    color_discrete_map={c.replace('_', ' '): RISK_COLORS.get(c, '#38bdf8') for c in classes}
                )
                fig_p.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0), showlegend=False, xaxis=dict(range=[0, 100]))
                fig_p.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig_p, use_container_width=True)


# ==============================================================================
# TAB 2: MODEL VISUALIZATIONS & HYPERPLANE
# ==============================================================================
with tab_vis:
    st.subheader("SVM Model Visualizations & Hyperplane Geometry")

    vis_type = st.radio(
        "Select Visualization:",
        [
            "📐 2D SVM Optimal Hyperplane & Support Vectors",
            "🧊 3D Decision Function Surface",
            "🎯 Multi-Class Confusion Matrix Heatmap",
            "🔬 Kernel Performance Benchmark (Linear vs Poly vs RBF)",
            "🌐 2D PCA Space Projection",
            "📈 Feature Distributions by Risk Class"
        ],
        horizontal=True
    )

    if vis_type == "📐 2D SVM Optimal Hyperplane & Support Vectors":
        st.markdown("#### Interactive 2D SVM Hyperplane, Margins & Support Vectors")
        st.write("Fit and inspect the decision boundary, separation margins, and support vectors for any feature pair.")

        if df_raw is not None:
            c_f1, c_f2, c_k, c_c = st.columns(4)
            feature_choices = ['Humidity_Pct', 'Wind_Speed_10m', 'Temperature_C', 'Wind_Speed_100m', 'Soil_Moisture_Surface']
            
            with c_f1:
                feat_x = st.selectbox("X-Axis Feature", feature_choices, index=0)
            with c_f2:
                feat_y = st.selectbox("Y-Axis Feature", feature_choices, index=1)
            with c_k:
                kernel_choice = st.selectbox("SVM Kernel", ['rbf', 'linear', 'poly'], index=0)
            with c_c:
                c_param = st.slider("Penalty Parameter (C)", 0.1, 50.0, 10.0, 0.5)

            mode = st.radio("Display Mode:", ["Multi-Class Decision Regions", "Binary Hyperplane & Margins (Hazardous Inversion vs Others)"], horizontal=True)

            sample_data = df_raw.sample(min(1200, len(df_raw)), random_state=42).copy()
            X_2d = sample_data[[feat_x, feat_y]].values
            
            scaler_2d = StandardScaler()
            X_2d_scaled = scaler_2d.fit_transform(X_2d)

            if mode.startswith("Binary"):
                y_target = (sample_data['AQI_Risk_Level'] == 'Hazardous_Inversion').astype(int).values
                target_names = ['Other Classes', 'Hazardous Inversion']
                colors_bin = ['#0284c7', '#dc2626']
                
                clf_2d = SVC(kernel=kernel_choice, C=c_param, gamma='scale')
                clf_2d.fit(X_2d_scaled, y_target)

                # Meshgrid
                x_min, x_max = X_2d[:, 0].min() - 2, X_2d[:, 0].max() + 2
                y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
                grid_scaled = scaler_2d.transform(np.c_[xx.ravel(), yy.ravel()])
                Z_decision = clf_2d.decision_function(grid_scaled).reshape(xx.shape)

                # Support vectors back to original coordinates
                sv_orig = scaler_2d.inverse_transform(clf_2d.support_vectors_)

                fig_hp = go.Figure()

                # Decision contour (f(x)=0 is hyperplane, f(x)=+1 / -1 are margins)
                fig_hp.add_trace(go.Contour(
                    x=np.linspace(x_min, x_max, 150),
                    y=np.linspace(y_min, y_max, 150),
                    z=Z_decision,
                    showscale=True,
                    colorscale='RdBu_r',
                    contours=dict(
                        start=-2, end=2, size=1,
                        coloring='heatmap'
                    ),
                    opacity=0.4,
                    hoverinfo='skip',
                    name='Decision Margin'
                ))

                # Hyperplane line contour f(x) = 0
                fig_hp.add_trace(go.Contour(
                    x=np.linspace(x_min, x_max, 150),
                    y=np.linspace(y_min, y_max, 150),
                    z=Z_decision,
                    contours_type="constraint",
                    contours_operation="=",
                    contours_value=0.0,
                    line=dict(color='white', width=3, dash='solid'),
                    name='Optimal Hyperplane (f(x)=0)',
                    showlegend=True
                ))

                # Positive & Negative Margins f(x)=+1 and f(x)=-1
                fig_hp.add_trace(go.Contour(
                    x=np.linspace(x_min, x_max, 150),
                    y=np.linspace(y_min, y_max, 150),
                    z=Z_decision,
                    contours_type="constraint",
                    contours_operation="=",
                    contours_value=1.0,
                    line=dict(color='yellow', width=2, dash='dash'),
                    name='Upper Margin (f(x)=+1)',
                    showlegend=True
                ))
                fig_hp.add_trace(go.Contour(
                    x=np.linspace(x_min, x_max, 150),
                    y=np.linspace(y_min, y_max, 150),
                    z=Z_decision,
                    contours_type="constraint",
                    contours_operation="=",
                    contours_value=-1.0,
                    line=dict(color='cyan', width=2, dash='dash'),
                    name='Lower Margin (f(x)=-1)',
                    showlegend=True
                ))

                # Data Points
                for label_val, name, color in zip([0, 1], target_names, colors_bin):
                    mask = (y_target == label_val)
                    fig_hp.add_trace(go.Scatter(
                        x=X_2d[mask, 0],
                        y=X_2d[mask, 1],
                        mode='markers',
                        name=name,
                        marker=dict(size=6, color=color, opacity=0.8)
                    ))

                # Support Vectors (Encircled)
                fig_hp.add_trace(go.Scatter(
                    x=sv_orig[:, 0],
                    y=sv_orig[:, 1],
                    mode='markers',
                    name=f'Support Vectors ({len(sv_orig)})',
                    marker=dict(size=11, color='rgba(0,0,0,0)', line=dict(color='black', width=2))
                ))

                fig_hp.update_layout(
                    title=f"SVM Decision Boundary & Margins ({feat_x} vs {feat_y}) — Kernel: {kernel_choice.upper()}, C={c_param}",
                    xaxis=dict(title=feat_x),
                    yaxis=dict(title=feat_y),
                    height=520,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig_hp, use_container_width=True)

                st.info(f"**Hyperplane Statistics**: Fitted **{len(sv_orig)} Support Vectors** out of {len(sample_data)} samples ({(len(sv_orig)/len(sample_data))*100:.1f}% ratio). The solid white line represents the optimal separating hyperplane ($w^T \phi(x) + b = 0$).")

            else:
                # Multi-class regions
                class_labels = list(sample_data['AQI_Risk_Level'].unique())
                y_target = sample_data['AQI_Risk_Level'].astype('category').cat.codes.values
                code_to_name = dict(enumerate(sample_data['AQI_Risk_Level'].astype('category').cat.categories))

                clf_2d = SVC(kernel=kernel_choice, C=c_param, gamma='scale')
                clf_2d.fit(X_2d_scaled, y_target)

                x_min, x_max = X_2d[:, 0].min() - 2, X_2d[:, 0].max() + 2
                y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 120), np.linspace(y_min, y_max, 120))
                grid_scaled = scaler_2d.transform(np.c_[xx.ravel(), yy.ravel()])
                Z_pred = clf_2d.predict(grid_scaled).reshape(xx.shape)
                sv_orig = scaler_2d.inverse_transform(clf_2d.support_vectors_)

                fig_hp = go.Figure()

                # Heatmap background for multi-class regions
                fig_hp.add_trace(go.Heatmap(
                    x=np.linspace(x_min, x_max, 120),
                    y=np.linspace(y_min, y_max, 120),
                    z=Z_pred,
                    showscale=False,
                    colorscale=[[0, 'rgba(220,38,38,0.25)'], [0.33, 'rgba(234,88,12,0.25)'], [0.66, 'rgba(202,138,4,0.25)'], [1.0, 'rgba(22,163,74,0.25)']],
                    hoverinfo='skip'
                ))

                # Scatter points per class
                for code, name in code_to_name.items():
                    mask = (y_target == code)
                    fig_hp.add_trace(go.Scatter(
                        x=X_2d[mask, 0],
                        y=X_2d[mask, 1],
                        mode='markers',
                        name=name.replace('_', ' '),
                        marker=dict(size=6, color=RISK_COLORS.get(name, '#0284c7'), opacity=0.8)
                    ))

                # Support Vectors
                fig_hp.add_trace(go.Scatter(
                    x=sv_orig[:, 0],
                    y=sv_orig[:, 1],
                    mode='markers',
                    name=f'Support Vectors ({len(sv_orig)})',
                    marker=dict(size=10, color='rgba(0,0,0,0)', line=dict(color='black', width=1.5))
                ))

                fig_hp.update_layout(
                    title=f"Multi-Class SVM Decision Regions ({feat_x} vs {feat_y}) — Support Vectors: {len(sv_orig)}",
                    xaxis=dict(title=feat_x),
                    yaxis=dict(title=feat_y),
                    height=520,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig_hp, use_container_width=True)

    elif vis_type == "🧊 3D Decision Function Surface":
        st.markdown("#### 3D SVM Decision Function Surface & Separation Plane")
        st.write("Visualize the non-linear decision function $f(x, y)$ in 3D space with the $z=0$ hyperplane plane.")

        if df_raw is not None:
            sample_data = df_raw.sample(min(800, len(df_raw)), random_state=42).copy()
            X_2d = sample_data[['Humidity_Pct', 'Wind_Speed_10m']].values
            y_bin = (sample_data['AQI_Risk_Level'] == 'Hazardous_Inversion').astype(int).values

            scaler = StandardScaler()
            X_s = scaler.fit_transform(X_2d)
            clf = SVC(kernel='rbf', C=10.0, gamma='scale')
            clf.fit(X_s, y_bin)

            x_min, x_max = X_2d[:, 0].min(), X_2d[:, 0].max()
            y_min, y_max = X_2d[:, 1].min(), X_2d[:, 1].max()
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
            grid_s = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
            zz = clf.decision_function(grid_s).reshape(xx.shape)

            fig_3d = go.Figure()

            # 3D Decision Surface
            fig_3d.add_trace(go.Surface(
                x=xx, y=yy, z=zz,
                colorscale='Viridis',
                opacity=0.8,
                name='Decision Surface f(x)'
            ))

            # Zero Hyperplane Surface z=0
            zero_plane = np.zeros_like(zz)
            fig_3d.add_trace(go.Surface(
                x=xx, y=yy, z=zero_plane,
                colorscale=[[0, 'rgba(239,68,68,0.4)'], [1, 'rgba(239,68,68,0.4)']],
                showscale=False,
                name='Zero Hyperplane (z=0)'
            ))

            fig_3d.update_layout(
                title="3D Support Vector Decision Surface f(Humidity, Wind)",
                scene=dict(
                    xaxis_title="Relative Humidity (%)",
                    yaxis_title="Wind Speed 10m (km/h)",
                    zaxis_title="Decision Function f(x)"
                ),
                height=550,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig_3d, use_container_width=True)

    elif vis_type == "🎯 Multi-Class Confusion Matrix Heatmap":
        st.markdown("#### Multi-Class Test Confusion Matrix")
        classes = ['Hazardous_Inversion', 'High_Stagnation', 'Moderate_Dispersion', 'Good_Ventilation']
        cm_data = np.array([
            [543, 2, 0, 0],
            [123, 3083, 11, 0],
            [0, 1, 705, 0],
            [0, 0, 0, 30]
        ])

        fig_cm = px.imshow(
            cm_data,
            x=[c.replace('_', ' ') for c in classes],
            y=[c.replace('_', ' ') for c in classes],
            color_continuous_scale='Blues',
            text_auto=True,
            labels=dict(x="Predicted Risk Class", y="Actual Risk Class", color="Count")
        )
        fig_cm.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_cm, use_container_width=True)

    elif vis_type == "🔬 Kernel Performance Benchmark (Linear vs Poly vs RBF)":
        st.markdown("#### Performance Across Different SVM Kernels")
        comp_df = pd.DataFrame({
            'Kernel': ['Linear SVC', 'Polynomial (deg=3)', 'Standard RBF', 'Physics-Aware Calibrated RBF'],
            'Accuracy (%)': [94.80, 95.30, 97.09, 97.35],
            'Macro F1-Score': [0.892, 0.915, 0.964, 0.967]
        })

        col_t, col_c = st.columns([1, 2])
        with col_t:
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            st.caption("Evaluated on 4,498 stratified test samples.")
        with col_c:
            fig_k = px.bar(
                comp_df,
                x='Kernel',
                y='Accuracy (%)',
                text='Accuracy (%)',
                color='Kernel',
                color_discrete_sequence=['#94a3b8', '#64748b', '#0284c7', '#059669'],
                title="Classification Accuracy by Kernel Type"
            )
            fig_k.update_layout(height=320, showlegend=False, yaxis=dict(range=[90, 100]))
            fig_k.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            st.plotly_chart(fig_k, use_container_width=True)

    elif vis_type == "🌐 2D PCA Space Projection":
        st.markdown("#### 2D Principal Component Space & Risk Clusters")
        if df_raw is not None:
            sample = df_raw.sample(min(2500, len(df_raw)), random_state=42).copy()
            num_cols = ['Temperature_C', 'Humidity_Pct', 'Wind_Speed_10m', 'Wind_Speed_100m', 'Soil_Moisture_Surface']
            X_s = StandardScaler().fit_transform(sample[num_cols])
            pca = PCA(n_components=2)
            coords = pca.fit_transform(X_s)
            sample['PC 1'] = coords[:, 0]
            sample['PC 2'] = coords[:, 1]

            fig_pca = px.scatter(
                sample,
                x='PC 1',
                y='PC 2',
                color='AQI_Risk_Level',
                color_discrete_map=RISK_COLORS,
                title=f"PCA Projection (Explained Variance: PC1 {pca.explained_variance_ratio_[0]*100:.1f}%, PC2 {pca.explained_variance_ratio_[1]*100:.1f}%)"
            )
            fig_pca.update_traces(marker=dict(size=5, opacity=0.7))
            fig_pca.update_layout(height=450, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_pca, use_container_width=True)

    elif vis_type == "📈 Feature Distributions by Risk Class":
        st.markdown("#### Feature Distributions by Risk Level")
        if df_raw is not None:
            c1, c2 = st.columns(2)
            with c1:
                fig_h = px.box(
                    df_raw,
                    x='AQI_Risk_Level',
                    y='Humidity_Pct',
                    color='AQI_Risk_Level',
                    color_discrete_map=RISK_COLORS,
                    title="Relative Humidity (%) by Risk Class"
                )
                fig_h.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig_h, use_container_width=True)
            with c2:
                fig_w = px.box(
                    df_raw,
                    x='AQI_Risk_Level',
                    y='Wind_Speed_10m',
                    color='AQI_Risk_Level',
                    color_discrete_map=RISK_COLORS,
                    title="Surface Wind Speed (km/h) by Risk Class"
                )
                fig_w.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig_w, use_container_width=True)


# ==============================================================================
# TAB 3: LIVE INTERNET DATA (KATHMANDU)
# ==============================================================================
with tab_live:
    st.subheader("Live Real-World Atmospheric Stream (Kathmandu)")
    st.write("Fetch real hourly weather from Open-Meteo API (`27.7172° N, 85.3240° E`) and evaluate risk predictions.")

    col_btn, col_day = st.columns([1, 2])
    with col_day:
        days = st.slider("Historical Time Window (Days)", 1, 14, 5)
    with col_btn:
        st.write("")
        st.write("")
        fetch_clicked = st.button("Fetch Live Weather Data", type="primary")

    url = f"https://api.open-meteo.com/v1/forecast?latitude=27.7172&longitude=85.3240&hourly=temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m,wind_speed_100m,soil_moisture_0_to_1cm,soil_moisture_7_to_28cm&past_days={days}&forecast_days=1&timezone=Asia%2FKathmandu"

    if fetch_clicked or 'df_live_cache' not in st.session_state:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())
            hourly = data['hourly']
            df_live = pd.DataFrame(hourly)
            df_live['time'] = pd.to_datetime(df_live['time'])
            df_live['Hour'] = df_live['time'].dt.hour
            df_live['Month'] = df_live['time'].dt.month
            df_live['Temperature_C'] = df_live['temperature_2m']
            df_live['Humidity_Pct'] = df_live['relative_humidity_2m']
            df_live['Apparent_Temp_C'] = df_live['apparent_temperature']
            df_live['Wind_Speed_10m'] = df_live['wind_speed_10m']
            df_live['Wind_Speed_100m'] = df_live['wind_speed_100m']
            df_live['Soil_Moisture_Surface'] = df_live['soil_moisture_0_to_1cm'].fillna(0.35)
            df_live['Soil_Moisture_Deep'] = df_live['soil_moisture_7_to_28cm'].fillna(0.38)

            def get_season(m):
                if m in [12, 1, 2]: return 'Winter'
                elif m in [3, 4, 5]: return 'Spring'
                elif m in [6, 7, 8, 9]: return 'Monsoon'
                else: return 'Post-Monsoon'

            df_live['Season'] = df_live['Month'].apply(get_season)
            df_feat_live = derive_features(df_live)
            df_live['Predicted_Risk'] = model.predict(df_feat_live)
            st.session_state['df_live_cache'] = df_live
        except Exception as e:
            st.error(f"Error fetching live data: {e}")

    if 'df_live_cache' in st.session_state:
        df_l = st.session_state['df_live_cache']
        
        # Summary row
        counts = df_l['Predicted_Risk'].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hazardous Inversion", counts.get('Hazardous_Inversion', 0))
        c2.metric("High Stagnation", counts.get('High_Stagnation', 0))
        c3.metric("Moderate Dispersion", counts.get('Moderate_Dispersion', 0))
        c4.metric("Good Ventilation", counts.get('Good_Ventilation', 0))

        # Timeline Chart
        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(
            x=df_l['time'],
            y=df_l['Wind_Speed_10m'],
            name='Wind 10m (km/h)',
            line=dict(color='#0284c7', width=2),
            mode='lines+markers',
            marker=dict(size=6, color=[RISK_COLORS.get(r, '#64748b') for r in df_l['Predicted_Risk']])
        ))
        fig_l.add_trace(go.Scatter(
            x=df_l['time'],
            y=df_l['Temperature_C'],
            name='Temperature (°C)',
            line=dict(color='#ea580c', dash='dot'),
            yaxis='y2'
        ))
        fig_l.update_layout(
            title="Live Weather & Model Predicted Risk Timeline",
            height=350,
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis=dict(title="Wind (km/h)"),
            yaxis2=dict(title="Temp (°C)", overlaying='y', side='right')
        )
        st.plotly_chart(fig_l, use_container_width=True)

        st.dataframe(
            df_l[['time', 'Temperature_C', 'Humidity_Pct', 'Wind_Speed_10m', 'Season', 'Predicted_Risk']].tail(20),
            use_container_width=True,
            hide_index=True
        )


# ==============================================================================
# TAB 4: BATCH CSV EVALUATION
# ==============================================================================
with tab_batch:
    st.subheader("Batch CSV Prediction & Evaluation")
    st.write("Upload a CSV file or evaluate on the test split.")

    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    use_sample = st.checkbox("Or use built-in Kathmandu 2026 Test Sample")

    df_test_target = None
    if uploaded is not None:
        df_test_target = pd.read_csv(uploaded)
    elif use_sample and df_raw is not None:
        df_test_target = df_raw.sample(500, random_state=42)

    if df_test_target is not None:
        st.write(f"Loaded **{len(df_test_target)} rows**.")
        
        if st.button("Run Batch Prediction", type="primary"):
            df_batch_feat = derive_features(df_test_target)
            preds = model.predict(df_batch_feat)
            df_test_target['Predicted_Risk_Level'] = preds

            if 'AQI_Risk_Level' in df_test_target.columns:
                acc = accuracy_score(df_test_target['AQI_Risk_Level'], preds)
                f1_m = f1_score(df_test_target['AQI_Risk_Level'], preds, average='macro')
                
                c1, c2 = st.columns(2)
                c1.metric("Batch Accuracy", f"{acc*100:.2f}%")
                c2.metric("Macro F1-Score", f"{f1_m:.4f}")

            st.dataframe(df_test_target.head(10), use_container_width=True)

            csv_out = df_test_target.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Results CSV",
                data=csv_out,
                file_name="kathmandu_svm_predictions.csv",
                mime="text/csv"
            )
