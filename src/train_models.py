"""
Phase 4 — Baseline Modelling
Models: Linear Regression, Random Forest Regressor, Gradient Boosting Regressor
Datasets: train_linear/test_linear (LR), train_tree/test_tree (RF, GBM)
"""

import pandas as pd
import numpy as np
import json
import joblib
import time
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r'C:\Users\nigam\OneDrive\Desktop\Smart-Traffic-Congestion-Prediction\data\processed')
MODEL_DIR  = Path(r'C:\Users\nigam\OneDrive\Desktop\Smart-Traffic-Congestion-Prediction\models')
REPORT_DIR = Path(r'C:\Users\nigam\OneDrive\Desktop\Smart-Traffic-Congestion-Prediction\reports')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 'traffic_volume'

# ── Load feature name lists ───────────────────────────────────────────────────
with open(DATA_DIR / 'feature_names.json') as f:
    feature_names = json.load(f)

LINEAR_FEATURES = feature_names['linear']
TREE_FEATURES   = feature_names['tree']

# ── Load datasets ─────────────────────────────────────────────────────────────
print("Loading datasets...")
train_linear = pd.read_csv(DATA_DIR / 'train_linear.csv')
test_linear  = pd.read_csv(DATA_DIR / 'test_linear.csv')
train_tree   = pd.read_csv(DATA_DIR / 'train_tree.csv')
test_tree    = pd.read_csv(DATA_DIR / 'test_tree.csv')

# Verify target is NOT in features
assert TARGET not in LINEAR_FEATURES, "Leakage: target in linear features!"
assert TARGET not in TREE_FEATURES,   "Leakage: target in tree features!"

X_train_lin = train_linear[LINEAR_FEATURES]
y_train_lin = train_linear[TARGET]
X_test_lin  = test_linear[LINEAR_FEATURES]
y_test_lin  = test_linear[TARGET]

X_train_tree = train_tree[TREE_FEATURES]
y_train_tree = train_tree[TARGET]
X_test_tree  = test_tree[TREE_FEATURES]
y_test_tree  = test_tree[TARGET]

print(f"  Linear  — Train: {X_train_lin.shape}, Test: {X_test_lin.shape}")
print(f"  Tree    — Train: {X_train_tree.shape}, Test: {X_test_tree.shape}")
print(f"  Null check — Linear train: {X_train_lin.isna().sum().sum()} | Tree train: {X_train_tree.isna().sum().sum()}")
print()

# ── Metric helper ─────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred, train_time):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  MAE  : {mae:,.2f}")
    print(f"  RMSE : {rmse:,.2f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  Train time: {train_time:.1f}s")
    return {'Model': name, 'MAE': round(mae, 2), 'RMSE': round(rmse, 2),
            'R2': round(r2, 4), 'Train_time_s': round(train_time, 1)}

results = []

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — Linear Regression
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("MODEL 1 — Linear Regression")
print("=" * 55)
t0 = time.time()
lr = LinearRegression(n_jobs=-1)
lr.fit(X_train_lin, y_train_lin)
train_time = time.time() - t0

y_pred_lr = lr.predict(X_test_lin)
results.append(evaluate("Linear Regression", y_test_lin, y_pred_lr, train_time))
joblib.dump(lr, MODEL_DIR / 'linear_regression.pkl')
print(f"  Saved → models/linear_regression.pkl")

# Top 10 most influential features by absolute coefficient
coef_df = pd.DataFrame({
    'feature': LINEAR_FEATURES,
    'coef': lr.coef_
}).reindex(np.abs(lr.coef_).argsort()[::-1])
print(f"\n  Top 10 features by |coefficient|:")
print(coef_df.head(10).to_string(index=False))
print()

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — Random Forest Regressor
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("MODEL 2 — Random Forest Regressor")
print("=" * 55)
print("  Training (n_estimators=100, n_jobs=-1)... ", end='', flush=True)
t0 = time.time()
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_tree, y_train_tree)
train_time = time.time() - t0
print(f"done ({train_time:.1f}s)")

y_pred_rf = rf.predict(X_test_tree)
results.append(evaluate("Random Forest", y_test_tree, y_pred_rf, train_time))
joblib.dump(rf, MODEL_DIR / 'random_forest.pkl')
print(f"  Saved → models/random_forest.pkl")

# Feature importances top 15
fi_rf = pd.DataFrame({
    'feature': TREE_FEATURES,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\n  Top 15 feature importances:")
print(fi_rf.head(15).to_string(index=False))
print()

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — Gradient Boosting Regressor
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("MODEL 3 — Gradient Boosting Regressor")
print("=" * 55)
print("  Training (n_estimators=200, lr=0.1, max_depth=5)... ", end='', flush=True)
t0 = time.time()
gbm = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42
)
gbm.fit(X_train_tree, y_train_tree)
train_time = time.time() - t0
print(f"done ({train_time:.1f}s)")

y_pred_gbm = gbm.predict(X_test_tree)
results.append(evaluate("Gradient Boosting", y_test_tree, y_pred_gbm, train_time))
joblib.dump(gbm, MODEL_DIR / 'gradient_boosting.pkl')
print(f"  Saved → models/gradient_boosting.pkl")

# Feature importances top 15
fi_gbm = pd.DataFrame({
    'feature': TREE_FEATURES,
    'importance': gbm.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\n  Top 15 feature importances:")
print(fi_gbm.head(15).to_string(index=False))
print()

# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("MODEL COMPARISON")
print("=" * 55)
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('RMSE')
print(results_df.to_string(index=False))
results_df.to_csv(REPORT_DIR / 'model_comparison.csv', index=False)
print(f"\nSaved → reports/model_comparison.csv")

# Save per-model predictions for residual analysis later
preds_df = pd.DataFrame({
    'y_true':           y_test_tree.values,
    'pred_rf':          y_pred_rf,
    'pred_gbm':         y_pred_gbm,
    'pred_lr':          y_pred_lr[:len(y_pred_rf)],   # same length, same rows
    'residual_rf':      y_test_tree.values - y_pred_rf,
    'residual_gbm':     y_test_tree.values - y_pred_gbm,
    'residual_lr':      y_test_tree.values - y_pred_lr[:len(y_pred_rf)],
})
preds_df.to_csv(REPORT_DIR / 'test_predictions.csv', index=False)
print(f"Saved → reports/test_predictions.csv")

# Save feature importances
fi_rf.to_csv(REPORT_DIR / 'feature_importance_rf.csv', index=False)
fi_gbm.to_csv(REPORT_DIR / 'feature_importance_gbm.csv', index=False)
coef_df.to_csv(REPORT_DIR / 'feature_coef_lr.csv', index=False)
print(f"Saved → reports/feature_importance_rf/gbm.csv, feature_coef_lr.csv")
