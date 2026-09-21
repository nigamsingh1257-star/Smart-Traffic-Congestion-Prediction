"""
Phase 3 — Feature Engineering
Metro Interstate Traffic Volume
Produces two datasets:
  - data/processed/features_linear.csv  (no weather_description OHE)
  - data/processed/features_tree.csv    (with weather_description OHE)
Both include: X_train, X_test, y_train, y_test saved as separate CSVs.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
CLEAN_PATH = Path(r'C:\Users\nigam\OneDrive\Desktop\Smart-Traffic-Congestion-Prediction\data\processed\traffic_cleaned.csv')
OUT_DIR    = Path(r'C:\Users\nigam\OneDrive\Desktop\Smart-Traffic-Congestion-Prediction\data\processed')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & SORT
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(CLEAN_PATH, parse_dates=['date_time'])
df = df.sort_values('date_time').reset_index(drop=True)
print(f"Loaded: {df.shape[0]:,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Raw time extractions
# ─────────────────────────────────────────────────────────────────────────────
df['hour']        = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.dayofweek     # 0=Mon, 6=Sun
df['month']       = df['date_time'].dt.month
df['year']        = df['date_time'].dt.year
print("Step 1 — Raw time features extracted")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Binary indicators
# ─────────────────────────────────────────────────────────────────────────────
df['is_weekend']   = (df['day_of_week'] >= 5).astype(int)
df['is_rush_hour'] = (
    (df['hour'].isin([7, 8, 9, 16, 17, 18])) & (df['is_weekend'] == 0)
).astype(int)
print("Step 2 — is_weekend, is_rush_hour created")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Cyclical encodings
# ─────────────────────────────────────────────────────────────────────────────
df['hour_sin']  = np.sin(2 * np.pi * df['hour']        / 24)
df['hour_cos']  = np.cos(2 * np.pi * df['hour']        / 24)
df['dow_sin']   = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos']   = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
print("Step 3 — Cyclical encodings (hour, dow, month) created")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Weather: merge rare categories, one-hot encode weather_main
# ─────────────────────────────────────────────────────────────────────────────
# Merge smoke (2) and squall (1) into 'other'
df['weather_main_grouped'] = df['weather_main'].replace({'smoke': 'other', 'squall': 'other'})

wm_dummies = pd.get_dummies(df['weather_main_grouped'], prefix='wm', drop_first=False, dtype=int)
# Drop 'wm_clear' as reference category (for linear models; trees ignore this)
ref_col = 'wm_clear'
wm_cols_full = list(wm_dummies.columns)
wm_cols_drop1 = [c for c in wm_cols_full if c != ref_col]  # 9 cols (drop reference)

df = pd.concat([df, wm_dummies], axis=1)
print(f"Step 4 — weather_main OHE: {len(wm_cols_full)} categories → columns: {wm_cols_full}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — One-hot encode weather_description (tree models only)
# ─────────────────────────────────────────────────────────────────────────────
wd_dummies = pd.get_dummies(df['weather_description'], prefix='wd', drop_first=False, dtype=int)
wd_cols = list(wd_dummies.columns)
df = pd.concat([df, wd_dummies], axis=1)
print(f"Step 5 — weather_description OHE: {len(wd_cols)} categories → {len(wd_cols)} columns")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Gap-aware lag features
# A lag value is only valid if the previous row is exactly 1 hour prior.
# Invalid (gap) lags are set to NaN here; filled post-split using train medians.
# ─────────────────────────────────────────────────────────────────────────────
tv = df['traffic_volume']
dt = df['date_time']

def gap_aware_lag(series, datetimes, hours):
    """Return series.shift(hours) but set NaN wherever the gap != `hours` h."""
    shifted_val  = series.shift(hours)
    shifted_time = datetimes.shift(hours)
    gap_hours    = (datetimes - shifted_time).dt.total_seconds() / 3600
    valid        = (gap_hours == hours)
    result       = shifted_val.where(valid, other=np.nan)
    return result

df['traffic_lag_1h']   = gap_aware_lag(tv, dt, 1)
df['traffic_lag_24h']  = gap_aware_lag(tv, dt, 24)
df['traffic_lag_168h'] = gap_aware_lag(tv, dt, 168)

lag_nan_1h   = df['traffic_lag_1h'].isna().sum()
lag_nan_24h  = df['traffic_lag_24h'].isna().sum()
lag_nan_168h = df['traffic_lag_168h'].isna().sum()
print(f"Step 6 — Lag features created (NaN due to gaps):")
print(f"  traffic_lag_1h   NaN : {lag_nan_1h:,}  ({lag_nan_1h/len(df)*100:.1f}%)")
print(f"  traffic_lag_24h  NaN : {lag_nan_24h:,}  ({lag_nan_24h/len(df)*100:.1f}%)")
print(f"  traffic_lag_168h NaN : {lag_nan_168h:,}  ({lag_nan_168h/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Gap-aware rolling mean features
# Rolling window uses only values from strictly before the current row,
# and only within contiguous hourly runs (gap breaks the window).
# ─────────────────────────────────────────────────────────────────────────────
def gap_aware_rolling_mean(series, datetimes, window_hours):
    """
    Rolling mean over [t-window_hours, t-1] (exclusive of current row).
    Restarts at each gap: any NaN within the window propagates → NaN result.
    """
    results = np.full(len(series), np.nan)
    vals    = series.values
    times   = datetimes.values.astype('datetime64[h]')

    for i in range(1, len(series)):
        # Collect up to window_hours previous rows if they are all consecutive
        window_vals = []
        for j in range(i - 1, max(i - window_hours - 1, -1), -1):
            expected_gap = int((times[i] - times[j]) / np.timedelta64(1, 'h'))
            if expected_gap != (i - j):   # gap detected
                break
            if pd.isna(vals[j]):
                break
            window_vals.append(vals[j])
        if len(window_vals) == window_hours:
            results[i] = np.mean(window_vals)
    return results

print("Step 7 — Computing rolling means (this may take ~30s)...")
df['traffic_roll_mean_3h']  = gap_aware_rolling_mean(tv, dt, 3)
df['traffic_roll_mean_24h'] = gap_aware_rolling_mean(tv, dt, 24)

roll_nan_3h  = pd.isna(df['traffic_roll_mean_3h']).sum()
roll_nan_24h = pd.isna(df['traffic_roll_mean_24h']).sum()
print(f"  traffic_roll_mean_3h  NaN : {roll_nan_3h:,}  ({roll_nan_3h/len(df)*100:.1f}%)")
print(f"  traffic_roll_mean_24h NaN : {roll_nan_24h:,}  ({roll_nan_24h/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Train / Test split (chronological: train 2012-2017, test 2018)
# ─────────────────────────────────────────────────────────────────────────────
train_mask = df['year'] < 2018
df_train = df[train_mask].copy()
df_test  = df[~train_mask].copy()
print(f"\nStep 8 — Train/test split:")
print(f"  Train : {len(df_train):,} rows  ({df_train['date_time'].min().date()} → {df_train['date_time'].max().date()})")
print(f"  Test  : {len(df_test):,}  rows  ({df_test['date_time'].min().date()} → {df_test['date_time'].max().date()})")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Fill NaN lags/rolling with TRAINING-SET hour-of-day medians
#           (computed from train only, applied to both train and test)
# ─────────────────────────────────────────────────────────────────────────────
lag_roll_cols = [
    'traffic_lag_1h', 'traffic_lag_24h', 'traffic_lag_168h',
    'traffic_roll_mean_3h', 'traffic_roll_mean_24h'
]

# Compute hour-of-day medians from train set
hour_medians = {}
for col in lag_roll_cols:
    hour_medians[col] = df_train.groupby('hour')[col].median()

# Fill NaN using the per-hour median from training set
# `medians` is a pd.Series indexed by hour (0-23) — map directly, do NOT subscript it
def fill_with_hour_median(frame, col, medians):
    frame = frame.copy()
    mask = frame[col].isna()
    frame.loc[mask, col] = frame.loc[mask, 'hour'].map(medians)
    return frame

for col in lag_roll_cols:
    df_train = fill_with_hour_median(df_train, col, hour_medians[col])
    df_test  = fill_with_hour_median(df_test,  col, hour_medians[col])

# Verify
remaining_nulls_train = df_train[lag_roll_cols].isna().sum().sum()
remaining_nulls_test  = df_test[lag_roll_cols].isna().sum().sum()
print(f"\nStep 9 — NaN filling with train hour medians:")
print(f"  Remaining NaN in train lag/roll cols : {remaining_nulls_train}")
print(f"  Remaining NaN in test  lag/roll cols : {remaining_nulls_test}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Define final feature sets and save
# ─────────────────────────────────────────────────────────────────────────────
TARGET = 'traffic_volume'

# Time features (keep raw AND cyclical — trees use raw, linear models use cyclical)
TIME_FEATURES = [
    'hour', 'day_of_week', 'month', 'year',
    'hour_sin', 'hour_cos',
    'dow_sin',  'dow_cos',
    'month_sin','month_cos',
    'is_weekend', 'is_rush_hour',
]

# Weather numeric
WEATHER_NUMERIC = ['temp', 'rain_1h', 'snow_1h', 'clouds_all']

# weather_main OHE — all 10 cols (linear models drop ref col at model time)
WM_OHE = wm_cols_full   # 10 cols including reference

# Lag + rolling
LAG_ROLL = lag_roll_cols

# Event
EVENT = ['is_holiday']

# ── Linear-model feature set (no weather_description) ──
FEATURES_LINEAR = TIME_FEATURES + WEATHER_NUMERIC + WM_OHE + EVENT + LAG_ROLL
# ── Tree-model feature set (includes weather_description OHE) ──
FEATURES_TREE   = TIME_FEATURES + WEATHER_NUMERIC + WM_OHE + wd_cols + EVENT + LAG_ROLL

print(f"\nStep 10 — Final feature sets:")
print(f"  Linear feature count : {len(FEATURES_LINEAR)}")
print(f"  Tree   feature count : {len(FEATURES_TREE)}")

# Confirm no NaN in any feature column
for label, feats, tr, te in [
    ('Linear', FEATURES_LINEAR, df_train, df_test),
    ('Tree',   FEATURES_TREE,   df_train, df_test),
]:
    null_train = tr[feats].isna().sum().sum()
    null_test  = te[feats].isna().sum().sum()
    print(f"  {label} — NaN in train features: {null_train} | test features: {null_test}")

# ── Save linear sets ──
df_train[FEATURES_LINEAR + [TARGET]].to_csv(OUT_DIR / 'train_linear.csv', index=False)
df_test [FEATURES_LINEAR + [TARGET]].to_csv(OUT_DIR / 'test_linear.csv',  index=False)

# ── Save tree sets ──
df_train[FEATURES_TREE + [TARGET]].to_csv(OUT_DIR / 'train_tree.csv', index=False)
df_test [FEATURES_TREE + [TARGET]].to_csv(OUT_DIR / 'test_tree.csv',  index=False)

# ── Save feature name lists ──
import json
with open(OUT_DIR / 'feature_names.json', 'w') as f:
    json.dump({'linear': FEATURES_LINEAR, 'tree': FEATURES_TREE, 'target': TARGET}, f, indent=2)

print(f"\n=== SAVED FILES ===")
for fn in ['train_linear.csv','test_linear.csv','train_tree.csv','test_tree.csv','feature_names.json']:
    p = OUT_DIR / fn
    print(f"  {fn:30s} {p.stat().st_size/1024:.1f} KB")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 3 FEATURE ENGINEERING COMPLETE")
print("="*60)
print(f"Total rows        : {len(df):,}")
print(f"Train rows        : {len(df_train):,}")
print(f"Test rows         : {len(df_test):,}")
print(f"Linear features   : {len(FEATURES_LINEAR)}")
print(f"Tree features     : {len(FEATURES_TREE)}")
print(f"Target            : {TARGET}")
print(f"All NaN resolved  : True")

print("\n--- LINEAR FEATURE LIST ---")
for i, f in enumerate(FEATURES_LINEAR, 1):
    print(f"  {i:2d}. {f}")

print("\n--- TREE-ONLY ADDITIONAL FEATURES (weather_description) ---")
print(f"  +{len(wd_cols)} one-hot columns from weather_description")
