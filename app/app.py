"""
Smart Traffic Congestion Prediction — Streamlit App
Final model: Gradient Boosting (Tuned)  |  66 tree features
Feature engineering exactly mirrors src/feature_engineering.py
"""

import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from datetime import datetime, date, time as dtime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "final_model.pkl"
META_PATH  = ROOT / "models" / "final_model_meta.json"
FEAT_PATH  = ROOT / "data" / "processed" / "feature_names.json"

# ── Compatibility shim for scikit-learn cross-version unpickling ─────────────
import sys
try:
    import sklearn._loss._loss as _loss_module
    sys.modules.setdefault('_loss', _loss_module)
except (ImportError, AttributeError):
    pass

# ── Load assets (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_meta():
    with open(META_PATH) as f:
        return json.load(f)

@st.cache_data
def load_feature_names():
    with open(FEAT_PATH) as f:
        return json.load(f)["tree"]

model        = load_model()
meta         = load_meta()
TREE_FEATURES = load_feature_names()

# ── Constants (from feature_engineering.py) ──────────────────────────────────
WEATHER_MAIN_CATS = ["clear", "clouds", "drizzle", "fog", "haze",
                     "mist", "other", "rain", "snow", "thunderstorm"]

WEATHER_DESC_CATS = [
    "broken clouds", "drizzle", "few clouds", "fog", "haze",
    "heavy intensity drizzle", "heavy intensity rain", "heavy snow",
    "light intensity drizzle", "light intensity shower rain", "light rain",
    "light rain and snow", "light shower snow", "light snow", "mist",
    "moderate rain", "overcast clouds", "proximity shower rain",
    "proximity thunderstorm", "proximity thunderstorm with drizzle",
    "proximity thunderstorm with rain", "scattered clouds", "sky is clear",
    "sleet", "smoke", "snow", "squalls", "thunderstorm",
    "thunderstorm with drizzle", "thunderstorm with heavy rain",
    "thunderstorm with light drizzle", "thunderstorm with light rain",
    "thunderstorm with rain", "very heavy rain",
]

# Mapping from weather_main display label → internal (lowercased, smoke/squall→other)
WM_DISPLAY = {
    "Clear":        "clear",
    "Clouds":       "clouds",
    "Drizzle":      "drizzle",
    "Fog":          "fog",
    "Haze":         "haze",
    "Mist":         "mist",
    "Rain":         "rain",
    "Snow":         "snow",
    "Thunderstorm": "thunderstorm",
    "Smoke / Squall (Other)": "other",
}

RUSH_HOURS = {7, 8, 9, 16, 17, 18}

# ── Feature builder ───────────────────────────────────────────────────────────
def build_feature_row(
    dt: datetime,
    temp_k: float,
    rain_1h: float,
    snow_1h: float,
    clouds_all: int,
    weather_main: str,       # internal lowercase
    weather_desc: str,       # exact string from WEATHER_DESC_CATS
    is_holiday: int,
    lag_1h: float,
    lag_24h: float,
    lag_168h: float,
    roll_3h: float,
    roll_24h: float,
) -> pd.DataFrame:
    """
    Mirrors feature_engineering.py steps 1–5 + lag/rolling.
    Returns a single-row DataFrame with columns in TREE_FEATURES order.
    """
    hour       = dt.hour
    dow        = dt.weekday()        # 0=Mon, 6=Sun  (matches dt.dayofweek)
    month      = dt.month
    year       = dt.year
    is_weekend  = int(dow >= 5)
    is_rush     = int(hour in RUSH_HOURS and is_weekend == 0)

    # Cyclical encodings — exactly as in feature_engineering.py
    hour_sin  = np.sin(2 * np.pi * hour       / 24)
    hour_cos  = np.cos(2 * np.pi * hour       / 24)
    dow_sin   = np.sin(2 * np.pi * dow        / 7)
    dow_cos   = np.cos(2 * np.pi * dow        / 7)
    month_sin = np.sin(2 * np.pi * (month-1)  / 12)
    month_cos = np.cos(2 * np.pi * (month-1)  / 12)

    # weather_main OHE (10 categories, all kept — matches tree feature set)
    wm_row = {f"wm_{cat}": 0 for cat in WEATHER_MAIN_CATS}
    if weather_main in WEATHER_MAIN_CATS:
        wm_row[f"wm_{weather_main}"] = 1

    # weather_description OHE (34 categories)
    wd_row = {f"wd_{cat}": 0 for cat in WEATHER_DESC_CATS}
    if weather_desc in WEATHER_DESC_CATS:
        wd_row[f"wd_{weather_desc}"] = 1

    row = {
        "hour":                 hour,
        "day_of_week":          dow,
        "month":                month,
        "year":                 year,
        "hour_sin":             hour_sin,
        "hour_cos":             hour_cos,
        "dow_sin":              dow_sin,
        "dow_cos":              dow_cos,
        "month_sin":            month_sin,
        "month_cos":            month_cos,
        "is_weekend":           is_weekend,
        "is_rush_hour":         is_rush,
        "temp":                 temp_k,
        "rain_1h":              rain_1h,
        "snow_1h":              snow_1h,
        "clouds_all":           clouds_all,
        **wm_row,
        **wd_row,
        "is_holiday":           is_holiday,
        "traffic_lag_1h":       lag_1h,
        "traffic_lag_24h":      lag_24h,
        "traffic_lag_168h":     lag_168h,
        "traffic_roll_mean_3h": roll_3h,
        "traffic_roll_mean_24h": roll_24h,
    }

    df = pd.DataFrame([row])[TREE_FEATURES]  # enforce exact column order
    return df


def interpret(volume: float) -> tuple[str, str]:
    """Return (label, colour_hex) for a given traffic volume."""
    if volume < 1000:
        return "Very Low Traffic", "#27ae60"
    elif volume < 2500:
        return "Low Traffic", "#2ecc71"
    elif volume < 4000:
        return "Moderate Traffic", "#f39c12"
    elif volume < 5500:
        return "Heavy Traffic", "#e67e22"
    else:
        return "Severe Congestion", "#e74c3c"


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Traffic Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-title { font-size: 2.4rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
    .sub-title  { font-size: 1.05rem; color: #555; margin-top: 0.2rem; margin-bottom: 1.5rem; }
    .metric-box { background: #f0f4ff; border-left: 5px solid #3b82f6; border-radius: 8px;
                  padding: 1.2rem 1.5rem; margin-top: 1rem; }
    .metric-value { font-size: 3rem; font-weight: 800; color: #1a1a2e; line-height: 1.1; }
    .metric-label { font-size: 0.9rem; color: #888; text-transform: uppercase; letter-spacing: 0.08em; }
    .interpret-chip { display: inline-block; padding: 0.4rem 1rem; border-radius: 20px;
                      font-weight: 600; font-size: 1rem; margin-top: 0.6rem; color: white; }
    .section-header { font-size: 1.1rem; font-weight: 600; color: #1a1a2e;
                      border-bottom: 2px solid #e5e7eb; padding-bottom: 0.4rem; margin-bottom: 1rem; }
    .info-note { background: #fffbeb; border-left: 4px solid #f59e0b;
                 padding: 0.7rem 1rem; border-radius: 4px; font-size: 0.88rem; color: #78350f; }
    .model-badge { background: #f0fdf4; border: 1px solid #86efac; border-radius: 6px;
                   padding: 0.3rem 0.8rem; font-size: 0.83rem; color: #166534; display: inline-block; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="main-title">🚦 Smart Traffic Congestion Prediction</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Metro Interstate I-94 · Twin Cities, MN · '
    'Predict hourly traffic volume from date, time, and weather conditions.</p>',
    unsafe_allow_html=True
)

col_badge1, col_badge2, col_badge3 = st.columns([1, 1, 4])
with col_badge1:
    st.markdown(
        f'<span class="model-badge">🤖 {meta["model_type"]}</span>',
        unsafe_allow_html=True
    )
with col_badge2:
    st.markdown(
        f'<span class="model-badge">📊 R² = {meta["test_r2"]:.4f} | RMSE = {meta["test_rmse"]:.0f} veh/h</span>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1.1, 1], gap="large")

# ── LEFT: Date / Time / Holiday ───────────────────────────────────────────────
with left:
    st.markdown('<p class="section-header">📅 Date & Time</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        input_date = st.date_input("Date", value=date.today(), min_value=date(2012, 10, 2))
    with c2:
        input_hour = st.selectbox(
            "Hour of day",
            options=list(range(24)),
            index=8,
            format_func=lambda h: f"{h:02d}:00  {'🌙' if h < 6 or h >= 22 else '☀️' if 6 <= h < 18 else '🌆'}"
        )

    is_holiday_flag = st.checkbox(
        "Is this a public holiday?",
        value=False,
        help="Columbus Day, Christmas Day, Independence Day, Labor Day, Martin Luther King Jr Day, "
             "Memorial Day, New Years Day, State Fair, Thanksgiving Day, Veterans Day, Washington's Birthday"
    )

    # Show derived time info
    dt_obj = datetime.combine(input_date, dtime(hour=input_hour))
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    rush_flag = input_hour in RUSH_HOURS and dt_obj.weekday() < 5
    st.caption(
        f"**{dow_names[dt_obj.weekday()]}** &nbsp;·&nbsp; "
        f"{'🏖️ Weekend' if dt_obj.weekday() >= 5 else '🏢 Weekday'} &nbsp;·&nbsp; "
        f"{'🚗 Rush hour' if rush_flag else 'Off-peak'}"
    )

    st.markdown('<p class="section-header" style="margin-top:1.2rem">🌤️ Weather Conditions</p>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        wm_display = st.selectbox(
            "Weather (main)",
            options=list(WM_DISPLAY.keys()),
            index=0,
        )
        temp_c = st.number_input(
            "Temperature (°C)", min_value=-40.0, max_value=50.0,
            value=15.0, step=0.5,
            help="Will be converted to Kelvin for the model (training used Kelvin)"
        )
        rain_1h_input = st.number_input(
            "Rain last hour (mm)", min_value=0.0, max_value=43.8,
            value=0.0, step=0.5,
            help="Capped at 43.8 mm/h (99.9th percentile after data cleaning)"
        )
    with c4:
        weather_desc_input = st.selectbox(
            "Weather description",
            options=WEATHER_DESC_CATS,
            index=WEATHER_DESC_CATS.index("sky is clear"),
        )
        clouds_all_input = st.slider(
            "Cloud cover (%)", min_value=0, max_value=100, value=0, step=5
        )
        snow_1h_input = st.number_input(
            "Snow last hour (mm)", min_value=0.0, max_value=10.0,
            value=0.0, step=0.1
        )

# ── RIGHT: Traffic Lag Inputs ─────────────────────────────────────────────────
with right:
    st.markdown('<p class="section-header">🕐 Historical Traffic (Lag Features)</p>', unsafe_allow_html=True)

    st.markdown(
        '<div class="info-note">'
        '<b>Why are these required?</b><br>'
        'The model was trained with lag features that represent recent traffic observations. '
        'These three values collectively explain <b>~82%</b> of prediction variance. '
        'Enter the actual traffic count recorded at those past timestamps on this road.'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    lag_1h_input = st.number_input(
        "Traffic volume — 1 hour ago (vehicles/hour)",
        min_value=0, max_value=7300, value=3000, step=50,
        help=f"Traffic count exactly 1 hour before the prediction hour. "
             f"Training data range: 0–7,280 veh/h, mean ≈ 3,260 veh/h."
    )
    lag_24h_input = st.number_input(
        "Traffic volume — 24 hours ago (vehicles/hour)",
        min_value=0, max_value=7300, value=3000, step=50,
        help="Traffic count at the same hour yesterday."
    )
    lag_168h_input = st.number_input(
        "Traffic volume — 1 week ago (vehicles/hour)",
        min_value=0, max_value=7300, value=3000, step=50,
        help="Traffic count at the same hour last week (most predictive lag feature, ~23% importance)."
    )

    st.markdown('<p class="section-header" style="margin-top:1rem">📈 Rolling Averages</p>', unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        roll_3h_input = st.number_input(
            "3-hour rolling avg (vehicles/hour)",
            min_value=0, max_value=7300, value=3000, step=50,
            help="Mean traffic over the 3 hours immediately before this hour."
        )
    with c6:
        roll_24h_input = st.number_input(
            "24-hour rolling avg (vehicles/hour)",
            min_value=0, max_value=7300, value=3000, step=50,
            help="Mean traffic over the 24 hours immediately before this hour."
        )

    st.markdown("")
    st.caption(
        "💡 **Tip:** If you don't have exact historical data, use the same value for all lag inputs "
        "as a rough baseline. The prediction will still be directionally meaningful."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICT BUTTON
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
predict_col, _ = st.columns([1, 2])
with predict_col:
    predict_btn = st.button("🔮  Predict Traffic Volume", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════
if predict_btn:
    try:
        # Convert inputs to model format
        temp_kelvin  = temp_c + 273.15   # training data used Kelvin
        weather_main = WM_DISPLAY[wm_display]
        is_hol       = int(is_holiday_flag)

        X = build_feature_row(
            dt           = dt_obj,
            temp_k       = temp_kelvin,
            rain_1h      = min(rain_1h_input, 43.78),   # honour training cap
            snow_1h      = snow_1h_input,
            clouds_all   = clouds_all_input,
            weather_main = weather_main,
            weather_desc = weather_desc_input,
            is_holiday   = is_hol,
            lag_1h       = float(lag_1h_input),
            lag_24h      = float(lag_24h_input),
            lag_168h     = float(lag_168h_input),
            roll_3h      = float(roll_3h_input),
            roll_24h     = float(roll_24h_input),
        )

        # Verify feature count before prediction
        assert X.shape == (1, 66), f"Feature shape mismatch: {X.shape}"
        assert list(X.columns) == TREE_FEATURES, "Feature order mismatch"

        pred = float(model.predict(X.values)[0])
        pred = max(0.0, round(pred, 1))

        label, colour = interpret(pred)

        # ── Display ──────────────────────────────────────────────────────────
        res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1])

        with res_col1:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-label">Predicted Traffic Volume</div>'
                f'<div class="metric-value">{pred:,.0f}</div>'
                f'<div style="color:#555; font-size:0.9rem;">vehicles / hour</div>'
                f'<span class="interpret-chip" style="background:{colour};">{label}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        with res_col2:
            st.markdown("**📋 Prediction Summary**")
            st.write(f"**Date/Time:** {dt_obj.strftime('%a %b %d, %Y — %H:00')}")
            st.write(f"**Weather:** {wm_display} / {weather_desc_input}")
            st.write(f"**Temperature:** {temp_c:.1f}°C  ({temp_kelvin:.2f} K)")
            st.write(f"**Holiday:** {'Yes ✅' if is_hol else 'No'}")
            st.write(f"**Rush hour:** {'Yes 🚗' if rush_flag else 'No'}")

        with res_col3:
            st.markdown("**📊 Congestion Scale**")
            scale = {
                "Very Low  ( < 1,000)":     "#27ae60",
                "Low       (1,000–2,499)":  "#2ecc71",
                "Moderate  (2,500–3,999)":  "#f39c12",
                "Heavy     (4,000–5,499)":  "#e67e22",
                "Severe    (≥ 5,500)":      "#e74c3c",
            }
            for lvl, col in scale.items():
                is_current = lvl.split()[0].lower() in label.lower()
                weight = "bold" if is_current else "normal"
                arrow = " ← You are here" if is_current else ""
                st.markdown(
                    f'<span style="color:{col}; font-weight:{weight};">● {lvl}{arrow}</span>',
                    unsafe_allow_html=True
                )

        # Model confidence note
        st.markdown("")
        st.caption(
            f"Model: {meta['model_type']} · "
            f"Test MAE ≈ ±{meta['test_mae']:.0f} veh/h · "
            f"Test RMSE ≈ {meta['test_rmse']:.0f} veh/h · "
            f"R² = {meta['test_r2']:.4f} · "
            f"Trained on 2012–2017, validated on 2018 I-94 data"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — About
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📖 About this App")
    st.markdown(
        "This app predicts hourly traffic volume on the **Metro Interstate I-94** "
        "in the Twin Cities, Minnesota using a machine learning model trained on "
        "data from October 2012 to September 2018."
    )
    st.markdown("---")
    st.markdown("### 🤖 Model Details")
    st.write(f"**Type:** {meta['model_type']}")
    st.write(f"**n_estimators:** {meta['best_params'].get('n_estimators', 'N/A')}")
    st.write(f"**max_depth:** {meta['best_params'].get('max_depth', 'N/A')}")
    st.write(f"**learning_rate:** {meta['best_params'].get('learning_rate', 'N/A')}")
    st.write(f"**Features:** {meta['n_features']}")
    st.write(f"**Train rows:** {meta['train_rows']:,}")
    st.write(f"**Test rows:** {meta['test_rows']:,}")
    st.markdown("---")
    st.markdown("### 📈 Performance (Test 2018)")
    st.metric("MAE",  f"{meta['test_mae']:.2f} veh/h")
    st.metric("RMSE", f"{meta['test_rmse']:.2f} veh/h")
    st.metric("R²",   f"{meta['test_r2']:.4f}")
    st.markdown("---")
    st.markdown("### 🏗️ Pipeline")
    st.markdown(
        "1. **EDA** — Phase 1\n"
        "2. **Cleaning** — Phase 2\n"
        "3. **Feature Engineering** — Phase 3\n"
        "4. **Baseline Models** — Phase 4\n"
        "5. **Hyperparameter Tuning** — Phase 5\n"
        "6. **Streamlit App** — Phase 6 *(you are here)*"
    )
    st.markdown("---")
    st.caption("Dataset: UCI Metro Interstate Traffic Volume · © 2024 Portfolio Project")
