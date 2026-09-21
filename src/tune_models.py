"""
Phase 5 — Hyperparameter Tuning (Fast version)
Strategy:
  - Use 50-tree proxy models to rank configs (fast ~5s each)
  - Refit winner at full n_estimators on full training data
  - Single chronological val fold (last 20% of train)

Total grid fits: 9 RF + 9 GBM = 18 proxy fits (~90s)
Final refits: 2 (one per model at best params) (~120s)
Estimated total runtime: ~5 minutes
"""

import pandas as pd
import numpy as np
import json, joblib, time, itertools
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR   = Path(r'C:\Users\nigam\OneDrive\Desktop\Smart-Traffic-Congestion-Prediction\data\processed')
MODEL_DIR  = Path(r'C:\Users\nigam\OneDrive\Desktop\Smart-Traffic-Congestion-Prediction\models')
REPORT_DIR = Path(r'C:\Users\nigam\OneDrive\Desktop\Smart-Traffic-Congestion-Prediction\reports')

TARGET = 'traffic_volume'
with open(DATA_DIR / 'feature_names.json') as f:
    TREE_FEATURES = json.load(f)['tree']

print("Loading data...")
train_df = pd.read_csv(DATA_DIR / 'train_tree.csv')
test_df  = pd.read_csv(DATA_DIR / 'test_tree.csv')

X_train_full = train_df[TREE_FEATURES].values
y_train_full = train_df[TARGET].values
X_test       = test_df[TREE_FEATURES].values
y_test       = test_df[TARGET].values

# Chronological inner split — last 20% of train = validation
split_idx = int(len(X_train_full) * 0.80)
X_tr, y_tr     = X_train_full[:split_idx], y_train_full[:split_idx]
X_val, y_val   = X_train_full[split_idx:], y_train_full[split_idx:]
print(f"  Inner train : {len(X_tr):,}  |  Inner val : {len(X_val):,}  |  Test : {len(X_test):,}")

def score(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2

# Quick leakage checks
assert TARGET not in TREE_FEATURES
assert train_df['year'].max() <= 2017 and test_df['year'].min() == 2018
print("Leakage checks: PASS\n")

# Baseline
print("=== BASELINE ===")
rf_base  = joblib.load(MODEL_DIR / 'random_forest.pkl')
gbm_base = joblib.load(MODEL_DIR / 'gradient_boosting.pkl')
mae_rfb,  rmse_rfb,  r2_rfb  = score(y_test, rf_base.predict(X_test))
mae_gbmb, rmse_gbmb, r2_gbmb = score(y_test, gbm_base.predict(X_test))
print(f"  RF  baseline : MAE={mae_rfb:.2f}  RMSE={rmse_rfb:.2f}  R2={r2_rfb:.4f}")
print(f"  GBM baseline : MAE={mae_gbmb:.2f}  RMSE={rmse_gbmb:.2f}  R2={r2_gbmb:.4f}\n")

PROXY_TREES = 50   # fast proxy; final refit uses full trees

all_rows = []

# ── RF GRID (proxy) ───────────────────────────────────────────────────────────
print("=== RF PROXY SEARCH (50 trees, 9 configs) ===")
rf_grid = list(itertools.product([None, 20, 30], [1, 3, 5]))
RF_FIXED = dict(max_features=0.7, bootstrap=True, random_state=42, n_jobs=-1)

best_rf_val  = np.inf
best_rf_cfg  = {}
t0 = time.time()
for i, (md, msl) in enumerate(rf_grid, 1):
    cfg = dict(max_depth=md, min_samples_leaf=msl)
    m   = RandomForestRegressor(n_estimators=PROXY_TREES, **RF_FIXED, **cfg)
    m.fit(X_tr, y_tr)
    _, rv, _ = score(y_val, m.predict(X_val))
    flag = " <--" if rv < best_rf_val else ""
    print(f"  [{i}/9] depth={str(md):4s} leaf={msl}  val_RMSE={rv:.1f}{flag}")
    all_rows.append({'model':'RF','max_depth':md,'min_samples_leaf':msl,
                     'learning_rate':None,'n_estimators_proxy':PROXY_TREES,
                     'val_rmse':round(rv,2),'test_mae':None,'test_rmse':None,'test_r2':None})
    if rv < best_rf_val:
        best_rf_val = rv
        best_rf_cfg = cfg
print(f"  Proxy search: {time.time()-t0:.0f}s | Best cfg: {best_rf_cfg}\n")

print(f"  Refitting RF with n_estimators=200, best cfg={best_rf_cfg} on FULL train...")
t0 = time.time()
rf_best = RandomForestRegressor(n_estimators=200, **RF_FIXED, **best_rf_cfg)
rf_best.fit(X_train_full, y_train_full)
print(f"  Refit done in {time.time()-t0:.0f}s")
mae_rf, rmse_rf, r2_rf = score(y_test, rf_best.predict(X_test))
print(f"  RF tuned test: MAE={mae_rf:.2f}  RMSE={rmse_rf:.2f}  R2={r2_rf:.4f}")
joblib.dump(rf_best, MODEL_DIR / 'rf_tuned.pkl')

# Patch test scores into RF rows
for row in all_rows:
    if row['model'] == 'RF' and row['max_depth'] == best_rf_cfg['max_depth'] \
       and row['min_samples_leaf'] == best_rf_cfg['min_samples_leaf']:
        row.update({'test_mae':round(mae_rf,2),'test_rmse':round(rmse_rf,2),'test_r2':round(r2_rf,4)})
        break

# ── GBM GRID (proxy) ──────────────────────────────────────────────────────────
print("\n=== GBM PROXY SEARCH (50 trees, 9 configs) ===")
gbm_grid = list(itertools.product([0.05, 0.08, 0.10], [4, 5, 6]))
GBM_FIXED = dict(min_samples_leaf=10, subsample=0.8, random_state=42)

best_gbm_val = np.inf
best_gbm_cfg = {}
t0 = time.time()
for i, (lr, md) in enumerate(gbm_grid, 1):
    cfg = dict(learning_rate=lr, max_depth=md)
    m   = GradientBoostingRegressor(n_estimators=PROXY_TREES, **GBM_FIXED, **cfg)
    m.fit(X_tr, y_tr)
    _, rv, _ = score(y_val, m.predict(X_val))
    flag = " <--" if rv < best_gbm_val else ""
    print(f"  [{i}/9] lr={lr}  depth={md}  val_RMSE={rv:.1f}{flag}")
    all_rows.append({'model':'GBM','max_depth':md,'min_samples_leaf':10,
                     'learning_rate':lr,'n_estimators_proxy':PROXY_TREES,
                     'val_rmse':round(rv,2),'test_mae':None,'test_rmse':None,'test_r2':None})
    if rv < best_gbm_val:
        best_gbm_val = rv
        best_gbm_cfg = cfg
print(f"  Proxy search: {time.time()-t0:.0f}s | Best cfg: {best_gbm_cfg}\n")

print(f"  Refitting GBM with n_estimators=300, best cfg={best_gbm_cfg} on FULL train...")
t0 = time.time()
gbm_best = GradientBoostingRegressor(n_estimators=300, **GBM_FIXED, **best_gbm_cfg)
gbm_best.fit(X_train_full, y_train_full)
print(f"  Refit done in {time.time()-t0:.0f}s")
mae_gbm, rmse_gbm, r2_gbm = score(y_test, gbm_best.predict(X_test))
print(f"  GBM tuned test: MAE={mae_gbm:.2f}  RMSE={rmse_gbm:.2f}  R2={r2_gbm:.4f}")
joblib.dump(gbm_best, MODEL_DIR / 'gbm_tuned.pkl')

for row in all_rows:
    if row['model'] == 'GBM' and row['learning_rate'] == best_gbm_cfg['learning_rate'] \
       and row['max_depth'] == best_gbm_cfg['max_depth']:
        row.update({'test_mae':round(mae_gbm,2),'test_rmse':round(rmse_gbm,2),'test_r2':round(r2_gbm,4)})
        break

# ── SAVE TUNING RESULTS ───────────────────────────────────────────────────────
tuning_df = pd.DataFrame(all_rows).sort_values('val_rmse').reset_index(drop=True)
tuning_df.to_csv(REPORT_DIR / 'tuning_results.csv', index=False)
print(f"\nSaved -> reports/tuning_results.csv ({len(tuning_df)} configs)")

# ── FINAL COMPARISON ─────────────────────────────────────────────────────────
comparison = pd.DataFrame([
    {'Model':'RF  - Baseline', 'MAE':round(mae_rfb,2),  'RMSE':round(rmse_rfb,2),  'R2':round(r2_rfb,4)},
    {'Model':'GBM - Baseline', 'MAE':round(mae_gbmb,2), 'RMSE':round(rmse_gbmb,2), 'R2':round(r2_gbmb,4)},
    {'Model':'RF  - Tuned',    'MAE':round(mae_rf,2),   'RMSE':round(rmse_rf,2),   'R2':round(r2_rf,4)},
    {'Model':'GBM - Tuned',    'MAE':round(mae_gbm,2),  'RMSE':round(rmse_gbm,2),  'R2':round(r2_gbm,4)},
]).sort_values('RMSE').reset_index(drop=True)
comparison['delta_RMSE'] = (comparison['RMSE'] - rmse_rfb).round(2)
print("\n=== FINAL COMPARISON (test set 2018) ===")
print(comparison.to_string(index=False))
comparison.to_csv(REPORT_DIR / 'final_model_comparison.csv', index=False)
print("Saved -> reports/final_model_comparison.csv")

# ── SELECT FINAL MODEL ────────────────────────────────────────────────────────
best_row = comparison.iloc[0]
if 'RF' in best_row['Model']:
    final_model = rf_best
    final_label = 'Random Forest (Tuned)'
    final_params = {**RF_FIXED, **best_rf_cfg, 'n_estimators': 200}
    final_params.pop('n_jobs', None)
else:
    final_model = gbm_best
    final_label = 'Gradient Boosting (Tuned)'
    final_params = {**GBM_FIXED, **best_gbm_cfg, 'n_estimators': 300}

joblib.dump(final_model, MODEL_DIR / 'final_model.pkl')

meta = {
    'model_type':   final_label,
    'best_params':  {k: (None if v is None else
                         int(v) if isinstance(v,(np.integer,int)) else
                         float(v) if isinstance(v,(np.floating,float)) else v)
                     for k,v in final_params.items()},
    'test_mae':     float(best_row['MAE']),
    'test_rmse':    float(best_row['RMSE']),
    'test_r2':      float(best_row['R2']),
    'feature_set':  'tree',
    'n_features':   len(TREE_FEATURES),
    'train_rows':   int(len(X_train_full)),
    'test_rows':    int(len(X_test)),
    'val_strategy': 'chronological 80/20 inner split; proxy search with 50 trees then full refit',
}
with open(MODEL_DIR / 'final_model_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

fi = pd.DataFrame({'feature': TREE_FEATURES,
                   'importance': final_model.feature_importances_}
                  ).sort_values('importance', ascending=False)
fi.to_csv(REPORT_DIR / 'feature_importance_final.csv', index=False)

print(f"\n{'='*55}")
print(f"FINAL MODEL : {final_label}")
print(f"  MAE  : {best_row['MAE']:,.2f}")
print(f"  RMSE : {best_row['RMSE']:,.2f}")
print(f"  R2   : {best_row['R2']:.4f}")
print(f"Saved -> models/final_model.pkl + final_model_meta.json")
print(f"{'='*55}")
print(f"\nTop 15 feature importances:")
print(fi.head(15).to_string(index=False))
print(f"\nSaved -> reports/feature_importance_final.csv")
