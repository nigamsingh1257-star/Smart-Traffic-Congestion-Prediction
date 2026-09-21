"""
Phase 2 — Data Cleaning
Metro Interstate Traffic Volume Dataset
Decisions: A, A, A
  1. Multi-label dedup → keep most-severe weather label per timestamp
  2. rain_1h cap      → 99.9th percentile of non-zero rain
  3. holiday          → binary is_holiday flag only
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path(r'C:\Users\nigam\OneDrive\Desktop\Smart-Traffic-Congestion-Prediction\data\raw\Metro_Interstate_Traffic_Volume.csv')
OUT_PATH = Path(r'C:\Users\nigam\OneDrive\Desktop\Smart-Traffic-Congestion-Prediction\data\processed\traffic_cleaned.csv')
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_PATH)
df['date_time'] = pd.to_datetime(df['date_time'])
print(f"Loaded:  {df.shape[0]:,} rows × {df.shape[1]} cols")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Drop complete duplicate rows (all 9 columns identical)
# ─────────────────────────────────────────────────────────────────────────────
before = len(df)
df = df.drop_duplicates(keep='first')
dropped_step1 = before - len(df)
print(f"\nStep 1 — Complete duplicates dropped : {dropped_step1:,}  →  {len(df):,} rows remain")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Normalize weather_main and weather_description to lowercase/stripped
#           (must happen BEFORE severity sort in Step 3)
# ─────────────────────────────────────────────────────────────────────────────
df['weather_main']        = df['weather_main'].str.strip().str.lower()
df['weather_description'] = df['weather_description'].str.strip().str.lower()

wm_unique_before = 11
wm_unique_after  = df['weather_main'].nunique()
wd_unique_before = 38
wd_unique_after  = df['weather_description'].nunique()
print(f"\nStep 2 — Case normalization")
print(f"  weather_main unique        : {wm_unique_before} → {wm_unique_after}")
print(f"  weather_description unique : {wd_unique_before} → {wd_unique_after}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Deduplicate date_time rows by keeping most-severe weather label
# ─────────────────────────────────────────────────────────────────────────────
SEVERITY_ORDER = [
    'thunderstorm', 'snow', 'rain', 'drizzle',
    'fog', 'mist', 'haze', 'smoke', 'squall', 'clouds', 'clear'
]
severity_map = {v: i for i, v in enumerate(SEVERITY_ORDER)}

before = len(df)
df['_severity'] = df['weather_main'].map(severity_map).fillna(len(SEVERITY_ORDER))  # unknown → lowest priority
df = (df
      .sort_values(['date_time', '_severity'])
      .drop_duplicates(subset='date_time', keep='first')
      .drop(columns=['_severity']))
dropped_step3 = before - len(df)
print(f"\nStep 3 — Multi-label weather dedup   : {dropped_step3:,} dropped  →  {len(df):,} rows remain")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Reset index (required for clean interpolation in Step 5)
# ─────────────────────────────────────────────────────────────────────────────
df = df.reset_index(drop=True)
print(f"\nStep 4 — Index reset ✓")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Impute temp == 0 via linear interpolation
# ─────────────────────────────────────────────────────────────────────────────
zero_temp_count = (df['temp'] == 0).sum()
df['temp'] = df['temp'].replace(0, np.nan)
df['temp'] = df['temp'].interpolate(method='linear')
remaining_nan  = df['temp'].isna().sum()
print(f"\nStep 5 — temp == 0 imputation")
print(f"  Rows with temp == 0 replaced : {zero_temp_count}")
print(f"  NaN remaining after interp   : {remaining_nan}  (should be 0)")

# Quick sanity: show imputed values
imputed_idx = [11898, 11899, 11900, 11901, 11946, 11947, 11948, 11949, 11950, 11951]
# After dedup & reset, these indices may have shifted — locate by date_time instead
block_dates = pd.to_datetime([
    '2014-01-31 03:00', '2014-01-31 04:00', '2014-01-31 05:00', '2014-01-31 06:00',
    '2014-02-02 03:00', '2014-02-02 04:00', '2014-02-02 05:00',
    '2014-02-02 06:00', '2014-02-02 07:00', '2014-02-02 08:00',
])
imputed_rows = df[df['date_time'].isin(block_dates)][['date_time', 'temp', 'traffic_volume']]
print(f"  Imputed temp values:\n{imputed_rows.to_string(index=False)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Cap rain_1h at 99.9th percentile of non-zero rain
# ─────────────────────────────────────────────────────────────────────────────
nonzero_rain = df[df['rain_1h'] > 0]['rain_1h']
cap_value = nonzero_rain.quantile(0.999)
rows_capped = (df['rain_1h'] > cap_value).sum()
df['rain_1h'] = df['rain_1h'].clip(upper=cap_value)
print(f"\nStep 6 — rain_1h outlier cap")
print(f"  99.9th-percentile cap value  : {cap_value:.4f} mm/h")
print(f"  Rows capped                  : {rows_capped}")
print(f"  New max rain_1h              : {df['rain_1h'].max():.4f} mm/h")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Encode holiday as binary is_holiday flag, drop original column
# ─────────────────────────────────────────────────────────────────────────────
df['is_holiday'] = df['holiday'].notna().astype(int)
df = df.drop(columns=['holiday'])
holiday_count = df['is_holiday'].sum()
print(f"\nStep 7 — Holiday encoding")
print(f"  is_holiday == 1 rows : {holiday_count}")
print(f"  Columns now          : {list(df.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL CHECKS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL DATASET SUMMARY")
print("=" * 60)
print(f"Shape            : {df.shape}")
print(f"Duplicate rows   : {df.duplicated().sum()}")
print(f"Dup date_times   : {df['date_time'].duplicated().sum()}")
print(f"Nulls per column :\n{df.isnull().sum()}")
print(f"\ntraffic_volume stats:\n{df['traffic_volume'].describe()}")
print(f"\ntemp stats:\n{df['temp'].describe()}")
print(f"\nrain_1h stats:\n{df['rain_1h'].describe()}")
print(f"\nweather_main value counts:\n{df['weather_main'].value_counts()}")
print(f"\nweather_description unique count : {df['weather_description'].nunique()}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
df.to_csv(OUT_PATH, index=False)
print(f"\n✅ Cleaned dataset saved → {OUT_PATH}")
print(f"   File size: {OUT_PATH.stat().st_size / 1024:.1f} KB")
