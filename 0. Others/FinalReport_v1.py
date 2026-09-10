# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:20:22 2026

@author: Aaron Wang
"""

import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

# --- 1. CUSTOM METRICS ---
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# --- 2. DATA LOADING (Fair-Rite 95) ---
mat = scipy.io.loadmat('CoreLossData_FR95.mat')

try:
    # Standard extraction
    f, B, P = mat['f'].flatten(), mat['B'].flatten(), mat['P'].flatten()
except KeyError:
    # Table/Struct extraction fallback
    cld = mat['CoreLossData']
    f = cld['f'][0,0].flatten()
    B = cld['B'][0,0].flatten()
    P = cld['P'][0,0].flatten()

df = pd.DataFrame({'f': f, 'B': B, 'P': P})

# --- 3. FEATURE ENGINEERING & SPLITTING ---
df['log_f'] = np.log10(df['f'])
df['log_B'] = np.log10(df['B'])
df['log_P'] = np.log10(df['P'])

X = df[['log_f', 'log_B']]
y = df['log_P']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. MODEL TRAINING ---
models = {
    "Steinmetz (Baseline)": LinearRegression(),
    "Lasso Poly (Deg 2)": LassoCV(cv=5),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Neural Network (MLP)": MLPRegressor(
        hidden_layer_sizes=(64, 64), solver='lbfgs', max_iter=5000, random_state=42
    )
}

results = []
test_preds = {}

for name, model in models.items():
    if "Poly" in name:
        poly = PolynomialFeatures(degree=2)
        X_tr, X_te = poly.fit_transform(X_train), poly.transform(X_test)
    elif "MLP" in name:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test
    
    model.fit(X_tr, y_train)
    log_p = model.predict(X_te)
    y_act = 10**y_test
    y_pre = 10**log_p
    
    results.append({
        "Model": name, 
        "MAPE (%)": calculate_mape(y_act, y_pre),
        "R2": r2_score(y_test, log_p),
        "RMSE": np.sqrt(mean_squared_error(y_test, log_p))
    })
    test_preds[name] = y_pre

print("\n--- Final Performance Report ---")
print(pd.DataFrame(results).sort_values("MAPE (%)").to_string(index=False))

# --- 5. COMPREHENSIVE VISUAL EVALUATION ---
methods = ["Lasso Poly (Deg 2)", "Random Forest", "Neural Network (MLP)"]
colors = ['blue', 'red', 'green']
y_actual = 10**y_test
baseline = test_preds["Steinmetz (Baseline)"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for i, name in enumerate(methods):
    # ROW 1: Parity Plots (Comparison vs Baseline)
    ax_p = axes[0, i]
    ax_p.scatter(y_actual, baseline, alpha=0.2, color='gray', label='Steinmetz')
    ax_p.scatter(y_actual, test_preds[name], alpha=0.5, color=colors[i], label=name)
    ax_p.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'k--', alpha=0.7)
    ax_p.set_xscale('log'); ax_p.set_yscale('log')
    ax_p.set_title(f"Parity: {name} vs Baseline")
    ax_p.set_xlabel("Measured Loss"); ax_p.set_ylabel("Predicted Loss")
    ax_p.legend()

    # ROW 2: Residual Analysis (Error Distributions)
    ax_r = axes[1, i]
    err = ((test_preds[name] - y_actual) / y_actual) * 100
    ax_r.hist(err, bins=30, color=colors[i], edgecolor='black', alpha=0.7)
    ax_r.axvline(0, color='black', linestyle='--')
    ax_r.set_title(f"Residuals: {name}")
    ax_r.set_xlabel("Percentage Error (%)")
    ax_r.set_ylabel("Frequency")

plt.tight_layout()
plt.show()