# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:24:01 2026

@author: Aaron Wang
"""

import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, make_scorer

# --- 1. ROBUST DATA LOADING ---
def load_magnetic_data(file_name):
    mat = scipy.io.loadmat(file_name)
    try:
        # standard extraction
        if 'f' in mat and 'B' in mat and 'P' in mat:
            f, B, P = mat['f'].flatten(), mat['B'].flatten(), mat['P'].flatten()
        else:
            # handle MATLAB table/structured data
            key = [k for k in mat.keys() if not k.startswith('__')][0]
            data = mat[key]
            f, B, P = data['f'][0,0].flatten(), data['B'][0,0].flatten(), data['P'][0,0].flatten()
    except Exception:
        # Fallback if structure is unknown
        arrays = [v.flatten() for k, v in mat.items() if isinstance(v, np.ndarray) and v.size > 10]
        f, B, P = arrays[0], arrays[1], arrays[2]
        
    df = pd.DataFrame({'f': f, 'B': B, 'P': P})
    df['log_f'], df['log_B'], df['log_P'] = np.log10(df['f']), np.log10(df['B']), np.log10(df['P'])
    return df

# --- 2. CONFIGURATION & MODELS ---
df = load_magnetic_data('CoreLossData_FR95.mat')
X, y = df[['log_f', 'log_B']], df['log_P']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Global scalers for MLP and diagnostic consistency
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

def mape_calc(y_true_log, y_pred_log):
    return np.mean(np.abs((10**y_true_log - 10**y_pred_log) / (10**y_true_log))) * 100

mape_scorer = make_scorer(mape_calc, greater_is_better=False)

# Include baseline in the core dictionary to automate comparison
models = {
    "Steinmetz (Baseline)": LinearRegression(),
    "Lasso Poly (Deg 2)": LassoCV(cv=5),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "MLP": MLPRegressor(hidden_layer_sizes=(64, 64), solver='lbfgs', max_iter=5000, random_state=42)
}

# --- 3. EXECUTION: PERFORMANCE & RELIABILITY ---
results, test_preds, diagnostics = [], {}, {}
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

# Manual Permutation Importance (Ensures compatibility with all Sklearn versions)
def get_importance(model, X, y):
    baseline_r2 = r2_score(y, model.predict(X))
    importances = []
    for col in X.columns:
        X_shuffled = X.copy()
        X_shuffled[col] = np.random.permutation(X_shuffled[col])
        importances.append(baseline_r2 - r2_score(y, model.predict(X_shuffled)))
    return np.array(importances)

for name, model in models.items():
    print(f"Analyzing {name}...")
    
    # Feature Handling
    if "Poly" in name:
        poly = PolynomialFeatures(degree=2)
        X_tr, X_te = poly.fit_transform(X_train), poly.transform(X_test)
    elif "MLP" in name:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test
    
    # A. Point & CV Metrics
    model.fit(X_tr, y_train)
    preds_log = model.predict(X_te)
    test_preds[name] = 10**preds_log
    
    cv_scores = cross_val_score(model, X_tr, y_train, cv=cv_strategy, scoring=mape_scorer)
    
    results.append({
        "Model": name,
        "MAPE (%)": mape_calc(y_test, preds_log),
        "R2": r2_score(y_test, preds_log),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds_log)),
        "CV Stability (Mean MAPE)": -cv_scores.mean()
    })

    # B. Reliability (Learning Curves)
    t_sizes, t_scores, v_scores = learning_curve(
        model, X_tr, y_train, cv=cv_strategy, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10))
    
    # C. Sensitivity (Permutation Importance)
    # We use a DataFrame wrapper to ensure compatibility with model.predict
    imp = get_importance(model, pd.DataFrame(X_te), y_test)
    
    diagnostics[name] = {
        'lc_sizes': t_sizes, 'lc_train': -t_scores.mean(axis=1), 'lc_val': -v_scores.mean(axis=1),
        'importance': imp
    }

# --- 4. COMPREHENSIVE VISUALIZATION DASHBOARDS ---
methods = list(models.keys())
colors = ['gray', 'blue', 'red', 'green']
y_actual = 10**y_test


'''
fig1, axes1 = plt.subplots(4, 2, figsize=(15, 20))
for i, name in enumerate(methods):
    # Parity Plots
    axes1[i, 0].scatter(y_actual, test_preds[name], alpha=0.5, color=colors[i])
    axes1[i, 0].plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'k--')
    axes1[i, 0].set_xscale('log'); axes1[i, 0].set_yscale('log')
    axes1[i, 0].set_title(f"Parity: {name}")
    axes1[i, 0].set_xlabel("Measured Core Loss ($P$) [mW/cm³]")
    axes1[i, 0].set_ylabel("Predicted Core Loss ($\hat{P}$) [mW/cm³]")
    axes1[i, 0].legend()
    
    # Residual Distribution
    err = ((test_preds[name] - y_actual) / y_actual) * 100
    axes1[i, 1].hist(err, bins=30, color=colors[i], alpha=0.7, edgecolor='black')
    axes1[i, 1].axvline(0, color='red', linestyle='--')
    axes1[i, 1].set_title(f"Residuals (%): {name}")
    axes1[i, 1].set_xlabel("Percentage Relative Error (%)")
    axes1[i, 1].set_ylabel("Frequency")

plt.tight_layout()

fig2, axes2 = plt.subplots(4, 2, figsize=(15, 20))
for i, name in enumerate(methods):
    d = diagnostics[name]
    # Learning Curves
    axes2[i, 0].plot(d['lc_sizes'], d['lc_train'], 'o-', label="Training Error")
    axes2[i, 0].plot(d['lc_sizes'], d['lc_val'], 's-', label="Validation Error")
    axes2[i, 0].set_title(f"Learning Curve: {name}")
    axes2[i, 0].set_xlabel("Number of Training Samples")
    axes2[i, 0].set_ylabel("Mean Squared Error (MSE)")
    axes2[i, 0].legend()

for i, name in enumerate(methods):    
    # Permutation Sensitivity
    labels = ['f', 'B'] if "Poly" not in name else ['f', 'B', 'f2', 'fB', 'B2', '1'][:len(d['importance'])]
    axes2[i, 1].bar(labels[:len(d['importance'])], d['importance'], color=colors[i], alpha=0.7)
    axes2[i, 1].set_title(f"Sensitivity: {name}")
    axes2[i, 1].set_xlabel("Physical Feature Component")
    axes2[i, 1].set_ylabel("Drop in $R^2$ Metric")

plt.tight_layout()
'''
# DASHBOARD 1: Accuracy Comparison (Parity & Residuals)
for i, name in enumerate(methods):
    # Create a figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- 1. Parity Plot (Left) ---
    axes[0].scatter(y_actual, test_preds[name], alpha=0.5, color=colors[i], label='Data Points')
    axes[0].plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'k--', label='Ideal (y=x)')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_title(f"Parity Analysis: {name}", fontsize=14)
    axes[0].set_xlabel("Measured Core Loss ($P$) [mW/cm³]", fontsize=12)
    axes[0].set_ylabel("Predicted Core Loss ($\hat{P}$) [mW/cm³]", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, which="both", ls="-", alpha=0.2)

    # --- 2. Residual Distribution (Right) ---
    # Calculate percentage relative error
    err = ((test_preds[name] - y_actual) / y_actual) * 100
    axes[1].hist(err, bins=30, color=colors[i], alpha=0.7, edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1].set_title(f"Residual Distribution: {name}", fontsize=14)
    axes[1].set_xlabel("Percentage Relative Error (%)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].legend()

    plt.tight_layout()
    
# DASHBOARD 2: Reliability & Sensitivity (Learning Curves & Importance)    
for i, name in enumerate(methods):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    d = diagnostics[name]
    current_color = colors[i]
    
    # --- Learning Curves ---
    axes[0].plot(d['lc_sizes'], d['lc_train'], 'o-', color=current_color, label="Train")
    axes[0].plot(d['lc_sizes'], d['lc_val'], 's-', color='gray', alpha=0.5, label="Val")
    axes[0].set_title(f"Learning Curve: {name}", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Training Samples")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # --- Feature Sensitivity (Permutation) ---
    if "Poly" in name:
        labels = ['f', 'B', '$f^2$', '$f \cdot B$', '$B^2$', '1'][:len(d['importance'])]
    else:
        labels = ['f', 'B']
        
    axes[1].bar(labels, d['importance'], color=current_color, alpha=0.8, edgecolor='black')
    axes[1].set_title(f"Sensitivity Analysis: {name}", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Physical Feature")
    axes[1].set_ylabel("Drop in $R^2$")
    axes[1].grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    
plt.show()

# --- 5. FINAL STATISTICAL REPORT ---
report_df = pd.DataFrame(results).sort_values("MAPE (%)")
print("\n--- FINAL STATISTICAL PERFORMANCE COMPARISON (INCLUDING BASELINE) ---")
print(report_df.to_string(index=False))