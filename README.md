# 🚦 Smart Traffic Congestion Prediction

An end-to-end machine learning system to predict hourly interstate traffic volume on the **I-94 Eastbound freeway (Minneapolis–St. Paul, MN)** using chronological feature engineering, gradient boosting regression, and an interactive Streamlit application.

---

## 1. Project Overview

Traffic congestion causes severe economic loss, excess fuel consumption, and transit delays. This project implements a production-ready machine learning pipeline that forecasts hourly traffic flow ($0 \text{ to } 7,280\text{ vehicles/hour}$) based on temporal dynamics, environmental weather conditions, and gap-aware historical traffic patterns.

The pipeline emphasizes **rigorous time-series validation**, **strict temporal leakage prevention**, and **deployable model artifact serving**.

---

## 2. Problem Statement

* **Task**: Hourly traffic volume forecasting on interstate highway I-94.
* **Problem Type**: Supervised Regression.
* **Target Variable**: `traffic_volume` (continuous integer count representing westbound/eastbound interstate throughput past the MN DoT ATR 301 station).
* **Objective**: Minimize prediction error, specifically Root Mean Squared Error ($\text{RMSE}$) and Mean Absolute Error ($\text{MAE}$), with particular emphasis on peak rush-hour volume accuracy.

---

## 3. Dataset

* **Source**: UCI Machine Learning Repository — [Metro Interstate Traffic Volume Dataset](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume).
* **Location**: Interstate 94, Minneapolis–St. Paul, MN (Eastbound).
* **Time Span**: October 2, 2012 to September 30, 2018 (~6 years of hourly records).
* **Raw Size**: 48,204 records across 9 attributes:
  * `holiday`: Categorical indicator of US national holidays.
  * `temp`: Average temperature in Kelvin.
  * `rain_1h`: Amount of rain in millimeters per hour.
  * `snow_1h`: Amount of snow in millimeters per hour.
  * `clouds_all`: Percentage of cloud cover (0–100%).
  * `weather_main`: Short textual summary of current weather (11 raw categories).
  * `weather_description`: Granular description of weather conditions (38 raw categories).
  * `date_time`: Hourly timestamp (`YYYY-MM-DD HH:00:00`).
  * `traffic_volume`: Hourly traffic volume count (Target).

---

## 4. Data Cleaning Pipeline

The raw dataset contained key operational anomalies that were addressed in [`src/clean_data.py`](src/clean_data.py):

1. **Complete Duplicate Elimination**:
   * Removed 17 completely identical rows across all 9 attributes ($48,204 \to 48,187$).
2. **Text Normalization**:
   * Lowercased and trimmed `weather_main` and `weather_description` strings, resolving case inconsistencies.
3. **Multi-Label Weather Deduplication**:
   * Root Cause: OpenWeatherMap API returns multiple weather records for a single timestamp if multiple weather conditions occur simultaneously (e.g., Rain + Mist), while `traffic_volume` is identical across these duplicates.
   * Resolution: Rather than arbitrary deletion, records were sorted by an established **meteorological severity hierarchy**:
     $$\text{Thunderstorm} > \text{Snow} > \text{Rain} > \text{Drizzle} > \text{Fog} > \text{Mist} > \text{Haze} > \text{Smoke} > \text{Squall} > \text{Clouds} > \text{Clear}$$
   * The most severe weather condition was retained per hourly timestamp, eliminating 7,612 redundant rows ($48,187 \to 40,575$ unique timestamps).
4. **Sensor Dropout Interpolation (`temp == 0`)**:
   * Identified 10 erroneous sensor zero-readings ($0\text{ K} \approx -273.15^\circ\text{C}$) in Jan/Feb 2014. Replaced with linear interpolation from neighboring hourly readings ($255.4\text{ K} \text{ to } 255.9\text{ K}$).
5. **Rain Sensor Outlier Capping**:
   * Observed an erroneous extreme value of $9,831.3\text{ mm/h}$ (greater than global record rainfall). Capped at the 99.9th percentile of non-zero rainfall ($43.78\text{ mm/h}$), preserving valid heavy rain storms while stabilizing tree splits.
6. **Holiday Feature Binary Encoding**:
   * Converted sparse text entries (99.87% null/None) into a robust binary indicator `is_holiday` (53 holiday timestamps retained post-deduplication).

*Output cleaned dataset*: [`data/processed/traffic_cleaned.csv`](data/processed/traffic_cleaned.csv) (40,575 rows, 0 nulls).

---

## 5. Feature Engineering

Engineered 66 total tree features in [`src/feature_engineering.py`](src/feature_engineering.py):

1. **Calendar & Temporal Extraction**:
   * `hour` (0–23), `day_of_week` (0=Mon, 6=Sun), `month` (1–12), `year` (2012–2018).
   * Binary flags: `is_weekend` and `is_rush_hour` (hours 7–9 and 16–18 on weekdays).
2. **Cyclical Encoding**:
   * Transformed periodic features into continuous circular coordinates using sine and cosine:
     $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
     $$\text{dow\_sin} = \sin\left(\frac{2\pi \cdot \text{dow}}{7}\right), \quad \text{dow\_cos} = \cos\left(\frac{2\pi \cdot \text{dow}}{7}\right)$$
     $$\text{month\_sin} = \sin\left(\frac{2\pi \cdot (\text{month}-1)}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi \cdot (\text{month}-1)}{12}\right)$$
3. **Categorical Weather One-Hot Encoding**:
   * `weather_main`: 10 dummy features (`wm_*`), grouping rare categories (`smoke`, `squall`) into `wm_other`.
   * `weather_description`: 34 dummy features (`wd_*`).
4. **Gap-Aware Lag Features**:
   * The dataset has intermittent hourly missing periods (22.8% missing hour slots). A naive `.shift()` would silently pull traffic from non-consecutive hours.
   * Lags were constructed with explicit time-delta validation:
     * `traffic_lag_1h`: Traffic recorded exactly 1 hour prior.
     * `traffic_lag_24h`: Traffic recorded at the exact same hour yesterday.
     * `traffic_lag_168h`: Traffic recorded at the exact same hour last week.
   * Any gap-broken lags were filled strictly using **training-set hour-of-day medians** to prevent future lookahead leakage.
5. **Gap-Aware Rolling Statistics**:
   * `traffic_roll_mean_3h`: 3-hour moving average over $[t-3, t-1]$.
   * `traffic_roll_mean_24h`: 24-hour moving average over $[t-24, t-1]$.
   * Current row $t$ is strictly excluded from all window calculations.

---

## 6. Time-Based Train / Test Split

Random splitting violates temporal causality in time-series forecasting. Data was partitioned using a strict **chronological cutoff**:

* **Training Set**: October 2, 2012 – December 31, 2017 (34,042 rows, ~84%)
* **Test Set**: January 1, 2018 – September 30, 2018 (6,533 rows, ~16%)
* **Overlap**: 0 rows. Test data represents an unseen future 9-month operational horizon.

---

## 7. Baseline Models Evaluated

Three baseline regressors were evaluated on the test set in [`src/train_models.py`](src/train_models.py):

| Rank | Model Architecture | Test MAE (veh/h) | Test RMSE (veh/h) | Test $R^2$ | Fit Time |
|:---:|:---|:---:|:---:|:---:|:---:|
| 🥇 | **Random Forest Regressor** (100 trees) | **147.21** | **231.94** | **0.9862** | 34.6s |
| 🥈 | **Gradient Boosting Regressor** (200 trees, lr=0.1) | 151.18 | 233.10 | 0.9861 | 56.6s |
| 🥉 | **Linear Regression** (OLS baseline) | 330.54 | 448.37 | 0.9484 | 0.1s |

*Key Takeaways*:
* All models achieved $R^2 \ge 0.94$, highlighting the predictive power of the gap-aware lag features.
* Tree ensembles outperformed linear regression by $\sim 216\text{ veh/h}$ in RMSE, capturing complex non-linear interactions between weather events and commute rush hours.

---

## 8. Hyperparameter Tuning Approach

Implemented in [`src/tune_models.py`](src/tune_models.py) using a chronological validation strategy:
* **Validation Split**: The training dataset (2012–2017) was chronologically partitioned 80/20 into an inner train set (27,233 rows) and an inner validation set (6,809 rows).
* **Two-Stage Proxy Search**:
  1. Evaluated candidate configurations using 50-tree proxy models on the inner validation split to rank configurations efficiently without nested parallel serialization bottlenecks.
  2. Refit top-ranked parameter sets on the full training set (34,042 rows) at full tree count (200 trees for RF, 300 trees for GBM).
* **Tuned Hyperparameters**:
  * Random Forest: `max_depth` $\in [\text{None}, 20, 30]$, `min_samples_leaf` $\in [1, 3, 5]$, `max_features` fixed at 0.7.
  * Gradient Boosting: `learning_rate` $\in [0.05, 0.08, 0.10]$, `max_depth` $\in [4, 5, 6]$, `n_estimators`=300.

---

## 9. Final Model Selection & Justification

| Model Variation | Test MAE (veh/h) | Test RMSE (veh/h) | Test $R^2$ | $\Delta\text{RMSE vs Baseline}$ |
|:---|:---:|:---:|:---:|:---:|
| 🏆 **Gradient Boosting (Tuned)** | **145.18** | **223.67** | **0.9872** | **-8.27** |
| Random Forest (Tuned) | 143.54 | 226.03 | 0.9869 | -5.91 |
| Random Forest (Baseline) | 147.21 | 231.94 | 0.9862 | Baseline |
| Gradient Boosting (Baseline) | 151.18 | 233.10 | 0.9861 | +1.16 |

### Why Gradient Boosting (Tuned) Was Selected
1. **Lowest RMSE ($223.67\text{ veh/h}$)**: RMSE squares individual errors, penalizing severe traffic mispredictions during peak hours. In intelligent transportation systems, missing a rush-hour jam by 1,000 vehicles has far greater operational cost than minor off-peak drift.
2. **Iterative Residual Correction**: By boosting shallow trees (`max_depth=6`, `learning_rate=0.10`), the model systematically corrects non-linear residual errors at transition hours (e.g., Sunday night to Monday morning rush).
3. **Compact Serialization**: The final GBM model (`~2.2 MB`) loads and evaluates significantly faster at inference time than the unpruned Random Forest forest (`~130 MB`), making it ideal for web app serving.

---

## 10. Final Test Metrics

Evaluated on 6,533 unseen hourly observations from calendar year 2018:

$$\text{MAE} = 145.18 \text{ vehicles/hour} \quad (\approx 4.4\% \text{ relative error on mean 3,290})$$
$$\text{RMSE} = 223.67 \text{ vehicles/hour}$$
$$R^2 = 0.9872 \quad (98.72\% \text{ variance explained})$$

### Top Feature Importances (Final Model)
1. `traffic_lag_1h` (66.9%): Primary trend anchor.
2. `traffic_lag_168h` (13.6%): Same-hour last-week recurring commute pattern.
3. `hour_cos` (13.6%): Diurnal cycle modulation.
4. `hour` (1.3%): Hourly progression.
5. `hour_sin` (0.8%): Diurnal cycle phase.
6. `day_of_week` / `is_weekend` (1.4%): Weekly traffic profile shift.

---

## 11. Leakage Verification

Automated audit in [`scratch/verify_leakage.py`](scratch/verify_leakage.py) verified 7 strict criteria:

* [x] **Check 1**: Zero year overlap between train ($\le 2017$) and test ($= 2018$).
* [x] **Check 2**: `traffic_lag_1h` matches $t-1$ exactly across 37,986 non-gap rows (0 mismatches).
* [x] **Check 3**: `traffic_lag_24h` matches $t-24$ across 28,871 valid rows (0 mismatches).
* [x] **Check 4**: `traffic_lag_168h` matches $t-168$ across 16,909 valid rows (0 mismatches).
* [x] **Check 5**: Rolling average loop starts strictly at $j = i - 1$ (row $i$ never included).
* [x] **Check 6**: Target column `traffic_volume` absent from both feature dictionaries.
* [x] **Check 7**: No training lag references row index $\ge 34,042$ (test boundary).

---

## 12. Streamlit Application

The interactive web dashboard ([`app/app.py`](app/app.py)) provides real-time traffic volume forecasting:

* **Interactive Controls**:
  * Date picker and hour slider with visual day/night and rush-hour indicators.
  * Public holiday toggle.
  * Meteorological parameters: Weather group, weather description, temperature in Celsius (automatically converted to Kelvin), rainfall, snowfall, and cloud cover percentage.
  * Historical traffic inputs: 1h, 24h, 168h lags and rolling averages with contextual defaults.
* **Output Displays**:
  * Prominent predicted volume metric with unit labels.
  * Dynamic Congestion Categorization chip:
    * Very Low Traffic ($< 1,000$)
    * Low Traffic ($1,000 - 2,499$)
    * Moderate Traffic ($2,500 - 3,999$)
    * Heavy Traffic ($4,000 - 5,499$)
    * Severe Congestion ($\ge 5,500$)
  * Sidebar model card reporting live metadata, hyperparameters, and test benchmarks.
* **Cross-Version Resilience**: Includes an automated unpickling compatibility shim for seamless execution across scikit-learn environments.

---

## 13. How to Run the Project Locally

### Prerequisites
* Python 3.10 – 3.14
* Git

### Step-by-Step Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/Smart-Traffic-Congestion-Prediction.git
   cd Smart-Traffic-Congestion-Prediction
   ```

2. **Create and Activate Virtual Environment**:
   * **Windows (PowerShell)**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   * **Linux / macOS**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit Dashboard**:
   ```bash
   streamlit run app/app.py --server.port 8501
   ```
   Open [http://localhost:8501](http://localhost:8501) in your browser.

5. **(Optional) Re-execute Pipeline Steps**:
   ```bash
   # Step 1: Clean raw data
   python src/clean_data.py

   # Step 2: Generate features
   python src/feature_engineering.py

   # Step 3: Train baseline models
   python src/train_models.py

   # Step 4: Run hyperparameter tuning
   python src/tune_models.py
   ```

---

## 14. Project Folder Structure

```text
Smart-Traffic-Congestion-Prediction/
├── app/
│   └── app.py                        # Streamlit web application
├── data/
│   ├── raw/
│   │   └── Metro_Interstate_Traffic_Volume.csv  # Raw UCI dataset
│   └── processed/
│       ├── feature_names.json        # Tree & linear feature lists
│       ├── test_linear.csv           # Test set for linear models
│       ├── test_tree.csv             # Test set for tree models
│       ├── traffic_cleaned.csv       # Cleaned & deduplicated data
│       ├── train_linear.csv          # Train set for linear models
│       └── train_tree.csv            # Train set for tree models
├── models/
│   ├── final_model.pkl               # Production Gradient Boosting model
│   ├── final_model_meta.json         # Architecture parameters & test metrics
│   ├── gbm_tuned.pkl                 # Tuned Gradient Boosting artifact
│   ├── gradient_boosting.pkl         # Baseline Gradient Boosting artifact
│   ├── linear_regression.pkl         # Baseline Linear Regression artifact
│   ├── random_forest.pkl             # Baseline Random Forest artifact
│   └── rf_tuned.pkl                  # Tuned Random Forest artifact
├── reports/
│   ├── feature_coef_lr.csv           # Linear Regression feature weights
│   ├── feature_importance_final.csv  # Final model feature importances
│   ├── feature_importance_gbm.csv    # Baseline GBM feature importances
│   ├── feature_importance_rf.csv     # Baseline RF feature importances
│   ├── final_model_comparison.csv    # Baseline vs. Tuned comparison table
│   ├── model_comparison.csv          # Phase 4 baseline evaluation table
│   ├── test_predictions.csv          # Predictions & residuals on 2018 test data
│   └── tuning_results.csv            # Grid search validation records
├── src/
│   ├── clean_data.py                 # Data cleaning & deduplication
│   ├── feature_engineering.py        # Gap-aware lags, rolling windows, cyclical features
│   ├── train_models.py               # Baseline training (LR, RF, GBM)
│   └── tune_models.py                # Validation-driven hyperparameter tuning
├── .gitignore                        # Git exclusion rules
├── README.md                         # Comprehensive project documentation
└── requirements.txt                  # Python dependencies
```

---

## 15. Key Interview Talking Points

1. **Why Regression, Not Classification?**
   * Traffic volume is inherently continuous. Imposing arbitrary thresholds (e.g., "high" vs. "low") discards granularity and creates boundary error spikes. Predicting continuous volume enables downstream transportation systems to set customizable, dynamic thresholds.
2. **Preventing Time-Series Data Leakage**:
   * Standard random K-Fold cross-validation leaks future traffic trends into the past. We strictly enforced chronological training ($2012–2017$) and testing ($2018$).
   * Missing periods in the raw series were handled via **gap-aware lags**: a 1-hour lag was only created if $(t - t_{\text{prev}}) = 1\text{ hour}$. Unfillable gap points were imputed using **training-set hour-of-day medians only**.
3. **Handling Multi-Label Weather Timestamps**:
   * Duplicate timestamps in the raw data were not corrupted sensor rows; they resulted from multiple weather observations logged for the same hour. We established a meteorological severity hierarchy to select the dominant weather condition without losing valid timestamps.
4. **Metric Prioritization: RMSE over MAE**:
   * While Random Forest achieved slightly lower MAE ($143.54$ vs. $145.18$), Gradient Boosting achieved lower RMSE ($223.67$ vs. $226.03$). For transit operations, extreme prediction errors during peak rush hours are costlier than minor off-peak variance, making RMSE the decisive criterion.
5. **Production Deployment Engineering**:
   * Gradient Boosting produced a lightweight serial file ($\sim 2.2\text{ MB}$) compared to Random Forest ($\sim 130\text{ MB}$), decreasing application cold-start time and memory footprint while providing equal or superior accuracy.
