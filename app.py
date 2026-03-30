import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb

st.set_page_config(page_title="Concrete Strength Predictor", layout="centered")
st.title("💪 Compressive Strength Predictor")
st.markdown("### Choose a model and input mix parameters to predict strength (MPa)")

@st.cache_data
def load_data():
    df = pd.read_csv("compressive strength.csv")
    df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
    return df

data = load_data()

features = ['Cement (Kg/m3)', 'FAgg (Kg/m3)', 'CSA (Kg/m3)', 'Water (Kg/m3)', 'River sand (Kg/m3)',
            'Silica fume (Kg/m3)', 'SP (Kg/m3)', 'W/B (%)', 'Fibers (%)', 'Aspect ratio (%)',
            'Tempurature (℃)', 'Curing Age (Day)', 'Specimen width (cm3)',
            'Specimen length (cm3)', 'Specimen height (cm3)']
features = [f for f in features if f in data.columns]
target = "Compressive strength (MPa)"
X = data[features]
y = data[target]

@st.cache_resource
def train_models():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Gradient Boosting
    param_dist_gb = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [3, 4, 5, 6, 7, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }
    gb = GradientBoostingRegressor(random_state=42)
    random_search_gb = RandomizedSearchCV(gb, param_dist_gb, n_iter=50, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42, verbose=0)
    random_search_gb.fit(X_train, y_train)
    best_gb = random_search_gb.best_estimator_

    # XGBoost
    param_dist_xgb = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 12, 35, 50],
        'learning_rate': [0.01, 0.05, 0.1, 0.3],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 1, 10],
        'reg_lambda': [0.1, 1, 5, 10, 20],
        'gamma': [0, 0.1, 0.2, 0.5],
        'min_child_weight': [1, 3, 5, 7]
    }
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    random_search_xgb = RandomizedSearchCV(xgb_model, param_dist_xgb, n_iter=50, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42, verbose=0)
    random_search_xgb.fit(X_train, y_train)
    best_xgb = random_search_xgb.best_estimator_

    return best_gb, best_xgb, X_train, X_test, y_train, y_test

gb_model, xgb_model, X_train, X_test, y_train, y_test = train_models()

if st.sidebar.checkbox("Show model performance on test set"):
    y_pred_gb = gb_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)
    st.sidebar.markdown("**Gradient Boosting**")
    st.sidebar.write(f"R²: {r2_score(y_test, y_pred_gb):.3f}")
    st.sidebar.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_gb)):.2f} MPa")
    st.sidebar.write(f"MAE: {mean_absolute_error(y_test, y_pred_gb):.2f} MPa")
    st.sidebar.write(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred_gb)*100:.2f}%")
    st.sidebar.markdown("**XGBoost**")
    st.sidebar.write(f"R²: {r2_score(y_test, y_pred_xgb):.3f}")
    st.sidebar.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_xgb)):.2f} MPa")
    st.sidebar.write(f"MAE: {mean_absolute_error(y_test, y_pred_xgb):.2f} MPa")
    st.sidebar.write(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred_xgb)*100:.2f}%")

model_choice = st.sidebar.radio("Choose model:", ("Gradient Boosting", "XGBoost"))

st.markdown("---")
st.subheader("Enter mix parameters")

col_left, col_right = st.columns(2)
with col_left:
    cement = st.number_input("Cement (kg/m³)", min_value=0.0, value=300.0, step=10.0)
    fagg = st.number_input("FAgg (kg/m³)", min_value=0.0, value=1000.0, step=10.0)
    csa = st.number_input("CSA (kg/m³)", min_value=0.0, value=800.0, step=10.0)
    water = st.number_input("Water (kg/m³)", min_value=0.0, value=180.0, step=5.0)
    river_sand = st.number_input("River sand (kg/m³)", min_value=0.0, value=700.0, step=10.0)
    silica_fume = st.number_input("Silica fume (kg/m³)", min_value=0.0, value=0.0, step=5.0)
    sp = st.number_input("SP (kg/m³)", min_value=0.0, value=0.0, step=1.0)

with col_right:
    wb = st.number_input("W/B (%)", min_value=0.0, max_value=100.0, value=0.45, step=0.01)
    fibers = st.number_input("Fibers (%)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    aspect_ratio = st.number_input("Aspect ratio (%)", min_value=0.0, value=0.0, step=10.0)
    temp = st.number_input("Temperature (℃)", min_value=-50.0, max_value=100.0, value=20.0, step=1.0)
    curing_age = st.number_input("Curing Age (Days)", min_value=0, value=28, step=1)
    width = st.number_input("Specimen width (cm³)", min_value=0.0, value=10.0, step=1.0)
    length = st.number_input("Specimen length (cm³)", min_value=0.0, value=10.0, step=1.0)
    height = st.number_input("Specimen height (cm³)", min_value=0.0, value=10.0, step=1.0)

if st.button("Predict Compressive Strength", type="primary"):
    input_values = [
        cement, fagg, csa, water, river_sand, silica_fume, sp,
        wb, fibers, aspect_ratio, temp, curing_age, width, length, height
    ]
    input_df = pd.DataFrame([input_values], columns=features)
    if model_choice == "Gradient Boosting":
        prediction = gb_model.predict(input_df)[0]
        model_name = "Gradient Boosting"
    else:
        prediction = xgb_model.predict(input_df)[0]
        model_name = "XGBoost"
    st.markdown("---")
    st.success(f"✅ **Predicted Compressive Strength ({model_name})**")
    st.markdown(f"### {prediction:.2f} MPa")
    st.caption("Model trained on the provided dataset – results are estimates only.")    st.markdown(f"### {prediction:.2f} MPa")
    st.caption("Model trained on the provided dataset – results are estimates only.")
