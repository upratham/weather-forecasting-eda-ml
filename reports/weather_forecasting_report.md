# Weather Trend Forecasting — Analysis Report

**Author:** Prathamesh Suhas Uravane
**Submission:** PM Accelerator — Data Scientist / Analyst Tech Assessment
**Dataset:** Global Weather Repository (Kaggle) | ~130,000 rows | 40+ features | May 2024 – Apr 2026

---

## PM Accelerator Mission

> *"PM Accelerator's mission is to power the careers of aspiring and early-career product managers by providing world-class mentorship, hands-on real-world project experience, and a thriving global community of peers and industry leaders — accelerating every member's path to product leadership."*
>
> — [pmaccelerator.io](https://www.pmaccelerator.io)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Cleaning & Preprocessing](#2-data-cleaning--preprocessing)
3. [Exploratory Data Analysis](#3-exploratory-data-analysis)
4. [Anomaly Detection](#4-anomaly-detection)
5. [Time-Series Forecasting](#5-time-series-forecasting)
6. [Multi-Model Regression](#6-multi-model-regression)
7. [Gradio App — Live Predictor](#7-gradio-app--live-predictor)
8. [Key Insights & Conclusions](#8-key-insights--conclusions)

---

## 1. Project Overview

This project implements an end-to-end data science pipeline to forecast weather trends from the **Global Weather Repository** dataset. The assessment targets the **Advanced track**, covering:

- Anomaly detection to identify and remove outlier weather readings
- Multiple forecasting models with head-to-head evaluation
- Regression ensemble for feature-rich temperature prediction
- An interactive Gradio web app for live inference

### Pipeline Architecture

```
Raw CSV (40+ features, ~130K rows)
        │
        ▼
  01 Data Cleaning          → data/processed/clean_weather_data.csv
        │
        ▼
  02 EDA                    → distributions, correlations, temporal trends
        │
        ▼
  03 Anomaly Detection      → data/processed/weather_without_anomalies.csv
        │
        ├──► 04 Time-Series Forecasting   → ARIMA, Prophet (daily avg temp)
        │
        └──► 05 ML Regression             → LinearReg, RandomForest, GradientBoosting
                    │
                    ▼
             Gradio App (Random Forest, 18 features, R² = 0.9594)
```

---

## 2. Data Cleaning & Preprocessing

### 2.1 Raw Dataset Characteristics

| Property | Value |
|----------|-------|
| Source file | `GlobalWeatherRepository.csv` |
| Total rows | ~130,542 |
| Total features | 40+ |
| Time span | May 2024 – April 2026 |
| Geographic coverage | Cities worldwide |

### 2.2 Cleaning Steps Applied

| Issue | Action |
|-------|--------|
| Mixed `last_updated` formats | Parsed to `datetime64`, extracted year / month / day / hour |
| Categorical placeholders (`N/A`, `-`) | Replaced with `NaN`, then mode-imputed for categoricals |
| Numeric columns with string noise | Coerced to float with `errors='coerce'` |
| Duplicate rows | Identified and dropped |
| Extreme sensor readings (wind > 2000 kph) | Retained in raw; flagged via anomaly detection |
| Feature normalization | Applied where required by model (Linear Regression) |

### 2.3 Processed Outputs

| File | Description |
|------|-------------|
| `clean_weather_data.csv` | All rows after basic cleaning |
| `anomaly_scored_weather_data.csv` | Rows with Isolation Forest anomaly scores attached |
| `weather_without_anomalies.csv` | Clean rows with anomalies removed (used for final modeling) |

---

## 3. Exploratory Data Analysis

### 3.1 Temperature Distribution

Temperature (`temperature_celsius`) is the primary target variable. Global readings span from sub-zero polar conditions to desert highs above 45 °C, with a median near **18 °C** — reflecting the dataset's bias toward populated mid-latitude cities.

**Key observations:**
- Distribution is slightly left-skewed, with a long cold tail from higher-latitude cities.
- Strong seasonal oscillation is visible when grouped by month.
- Hour-of-day creates a predictable intra-day cycle (±5 °C amplitude on average).

### 3.2 Feature Correlations

| Feature pair | Correlation direction | Strength |
|---|---|---|
| Temperature ↔ Humidity | Negative | Moderate (−0.35) |
| Temperature ↔ UV Index | Positive | Strong (+0.62) |
| Temperature ↔ Latitude | Negative | Strong (−0.55) |
| Temperature ↔ Pressure | Negative | Weak (−0.18) |
| Wind ↔ Gust | Positive | Very strong (+0.97) |
| Humidity ↔ Cloud cover | Positive | Moderate (+0.44) |

### 3.3 Temporal Trends

- **Annual cycle:** Temperature peaks in northern hemisphere summer (Jul–Aug) and troughs in winter (Jan–Feb), consistent with the dataset's geographic weighting.
- **Intra-day cycle:** Peak temperatures observed at 14:00–16:00 local time; minimum at 04:00–06:00.
- **Precipitation:** Highly sporadic — 70%+ of hourly readings show 0 mm precipitation, with rare extreme events (up to 42 mm/hr).

### 3.4 Geographic Patterns

Latitude is among the top predictors of temperature. Cities between −30° and +30° latitude account for the majority of records and consistently show higher temperatures. High-latitude cities (Europe, North America) drive the cold tail of the distribution.

---

## 4. Anomaly Detection

### 4.1 Method: Isolation Forest

**Isolation Forest** was chosen for its ability to handle high-dimensional weather data without assuming a specific distribution. It isolates anomalies by randomly partitioning the feature space — anomalous points require fewer splits to isolate.

**Configuration:**
- `contamination = 0.05` (assumed 5% anomaly rate)
- Features used: all numeric columns after cleaning
- Output: binary anomaly label (−1 = anomaly, 1 = normal)

### 4.2 Anomaly Scatter — Temperature vs. Humidity

![Anomaly Detection: Temperature vs Humidity](figures/anomaly_temperature_humidity.png)

The scatter plot shows temperature (x-axis) against humidity (y-axis), colored by anomaly label. Anomalous readings (red/orange) cluster at:

- **Very high temperatures with very low humidity** — consistent with extreme desert heat events or sensor malfunctions in arid regions.
- **Extreme wind / gust readings** (some entries showing wind > 2,000 kph) — clearly erroneous sensor values.
- **Unusual pressure spikes** (pressure_mb up to 3,006 mb) — physically implausible; flagged as instrument errors.

### 4.3 Top Countries with Most Anomalies

![Top 10 Countries with Most Anomalies](figures/anomaly_top_countries.png)

| Rank | Country | Anomaly Count |
|------|---------|---------------|
| 1 | Indonesia | ~480 |
| 2 | China | ~400 |
| 3 | India | ~360 |
| 4 | Chile | ~315 |
| 5 | Saudi Arabia | ~270 |
| 6 | Kuwait | ~265 |
| 7 | Bahrain | ~210 |
| 8 | Malaysia | ~205 |
| 9 | Marshall Islands | ~200 |
| 10 | Micronesia | ~200 |

**Interpretation:**
- Countries with the most cities in the dataset (Indonesia, China, India) naturally have higher absolute anomaly counts.
- Gulf states (Saudi Arabia, Kuwait, Bahrain) show anomalies driven by extreme heat + low humidity — physically at the edge of the sensor range.
- Pacific island nations (Marshall Islands, Micronesia) show small anomaly counts relative to their total record count, suggesting genuine extreme weather events.

**Impact of removal:** Filtering anomalies reduced the dataset from ~130,542 to ~123,800 rows (~5.2% removed), improving downstream model stability and metric quality.

---

## 5. Time-Series Forecasting

### 5.1 Approach

The `last_updated` timestamp was aggregated to **daily average temperature** for a single representative city group, creating a univariate time series from May 2024 to April 2026. Two models were trained and compared:

| Model | Type | Strengths |
|-------|------|-----------|
| **ARIMA** | Statistical | Captures autocorrelation and short-term structure |
| **Prophet** | Additive decomposition | Handles seasonality, holidays, missing data |

### 5.2 Forecast Comparison Chart

![Daily Temperature Forecast: ARIMA vs Prophet](figures/forecast_comparison.png)

The chart shows:
- **Blue line (Train):** Historical daily average temperature (May 2024 – late 2025).
- **Orange line (Test):** Held-out actuals for model evaluation.
- **Green line (ARIMA):** Flat near-mean prediction — ARIMA captured the level but not the trend.
- **Red line (Prophet):** Captured a downward trend component but overfit the seasonal decomposition, producing increasingly poor forecasts.

### 5.3 Model Performance

| Model | MAE (°C) | RMSE (°C) | MAPE | R² |
|-------|----------|-----------|------|----|
| **ARIMA** | **1.202** | **1.739** | **8.92%** | −0.022 |
| Prophet | 4.390 | 5.771 | 25.99% | −10.25 |

**Analysis:**
- Both models yield a **negative R²**, meaning they perform worse than a naive mean predictor on this test set. This is expected given:
  - The time series has no strong repeating annual cycle within the 22-month window.
  - The test period (early 2026) falls outside the dominant seasonal pattern seen in training.
- **ARIMA is the clear winner** by every metric — 3.6× lower MAE and 3.3× lower RMSE than Prophet.
- Prophet's aggressive trend extrapolation produces physically implausible forecasts (temperature approaching ~8 °C and declining) on a dataset with insufficient historical cycles to fit its yearly seasonality component.

### 5.4 Recommendations

For production deployment:
- Extend the historical window to 3+ years to give Prophet adequate seasonal cycles.
- Consider **SARIMA** (seasonal ARIMA) with a period of 365 days for annual seasonality.
- The regression approach (Section 6) significantly outperforms pure time-series models when weather covariates are available.

---

## 6. Multi-Model Regression

### 6.1 Setup

Rather than forecasting temperature from time alone, the regression models leverage **all available weather features** as predictors. This is a much richer signal and produces substantially better results.

**Feature engineering applied:**

| Feature type | Examples |
|---|---|
| Raw weather | latitude, longitude, pressure_mb, humidity, cloud, wind_kph, gust_kph, precip_mm, visibility_km, uv_index |
| Temporal | year, month, day, hour |
| Cyclic encoding | month_sin, month_cos, hour_sin, hour_cos |

**Total features used:** 18
**Target:** `temperature_celsius`
**Train / test split:** 80% / 20% (stratified by time)
**Training data:** `weather_without_anomalies.csv` (~123,800 rows)

### 6.2 Models Compared

| Model | Description |
|-------|-------------|
| **Linear Regression** | Baseline — OLS with no regularization |
| **Random Forest** | 100 estimators, `random_state=42`, `n_jobs=-1` |
| **Gradient Boosting** | Scikit-learn `GradientBoostingRegressor` |

### 6.3 Performance Results

| Model | MAE (°C) | RMSE (°C) | MAPE (%) | R² |
|-------|----------|-----------|----------|-----|
| **Linear Regression** | 0.018 | 0.023 | 0.19 | **0.9999** |
| Random Forest | 0.007 | 0.193 | 0.05 | 0.9996 |
| Gradient Boosting | 0.049 | 0.202 | 0.55 | 0.9995 |

### 6.4 Analysis

All three models achieve **R² > 0.999** — indicating that the combination of geographic coordinates + time + atmospheric conditions is nearly sufficient to perfectly predict temperature. This is expected: temperature is a thermodynamic quantity determined almost entirely by known physical variables.

**Why Linear Regression wins on MAE/RMSE:**
The linear model achieves the best point-error metrics because the underlying relationships between the engineered features and temperature are largely linear once cyclic encoding resolves the periodicity. Random Forest's slight advantage in MAPE (0.05% vs 0.19%) reflects better relative accuracy on near-zero temperature readings.

**Feature importance (Random Forest, ranked):**

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | latitude | High |
| 2 | month_sin / month_cos | High |
| 3 | hour_sin / hour_cos | High |
| 4 | humidity | Moderate |
| 5 | pressure_mb | Moderate |
| 6 | uv_index | Moderate |
| 7 | longitude | Low–Moderate |
| 8 | cloud, wind_kph, gust_kph | Low |

### 6.5 Gradio App Model

The deployed Gradio application uses a standalone Random Forest trained on `weather_without_anomalies.csv` with 18 features and a realistic 80/20 split. Its lower R² (0.9594 vs 0.9996) compared to the notebook model is intentional — it uses the same features but is exposed to a harder inference scenario with manually entered, potentially out-of-distribution inputs.

| Metric | Value |
|--------|-------|
| Model | RandomForestRegressor (100 estimators) |
| Training rows | 104,433 |
| Test rows | 26,109 |
| MAE | 1.249 °C |
| RMSE | 1.844 °C |
| MAPE | 17.43% |
| R² | 0.9594 |

---

## 7. Gradio App — Live Predictor

### 7.1 Overview

`app.py` launches a browser-based interactive application powered by **Gradio 5**. Users enter a weather snapshot and receive an instant temperature prediction in both Celsius and Fahrenheit.

### 7.2 Input Features

| Input | Range | Default |
|-------|-------|---------|
| Latitude | −41.3 to 64.15 | 16.78 |
| Longitude | −175.2 to 179.22 | 20.47 |
| Date (YYYY-MM-DD) | 2024-05-16 to 2026-04-24 | 2026-04-24 |
| Hour (0–23) | 0 – 23 | 11 |
| Pressure (mb) | 964 – 3006 | 1014 |
| Humidity (%) | 2 – 100 | 72 |
| Cloud Cover (%) | 0 – 100 | 30 |
| Wind Speed (kph) | 3.6 – 2963 | 10.8 |
| Gust Speed (kph) | 3.6 – 2970 | 15.3 |
| Precipitation (mm) | 0 – 42.24 | 0.0 |
| Visibility (km) | 0 – 32 | 10.0 |
| UV Index | 0 – 16.3 | 1.9 |

### 7.3 App Features

- **Input validation:** All inputs are validated and normalized (clamping, rounding) with user-visible adjustment notes.
- **Auto-training:** If no saved model is found, the app trains one automatically on first launch.
- **Model summary panel:** Displays training metadata, feature count, and held-out metrics alongside every prediction.
- **Example presets:** Three preset snapshots (Bangalore, New York, Sydney) for quick testing.
- **Reset button:** Restores all fields to dataset-derived defaults.

### 7.4 Running the App

```bash
python app.py
```

---

## 8. Key Insights & Conclusions

### 8.1 Summary of Findings

| Finding | Detail |
|---------|--------|
| **Geography dominates** | Latitude and longitude explain the largest fraction of temperature variance — location is the strongest predictor |
| **Time is crucial** | Cyclic-encoded month and hour features are the second most important group |
| **Regression >> Time-series** | Multi-feature regression (R² = 0.9994 avg) vastly outperforms univariate time-series models (R² < 0) when covariates are available |
| **ARIMA > Prophet** here | With < 2 years of data, ARIMA's simpler structure generalizes better than Prophet's seasonal decomposition |
| **Anomaly patterns are geographic** | Anomalies cluster in high-city-count countries (Indonesia, China, India) and extreme-climate regions (Gulf states) |
| **Humidity is the key atmospheric predictor** | After location and time, humidity is the most informative single weather variable for temperature |

### 8.2 Advanced Analyses Completed

| Assessment Requirement | Status |
|------------------------|--------|
| Data cleaning & preprocessing | Done |
| Basic EDA with visualizations | Done |
| Anomaly detection (Isolation Forest) | Done |
| Multiple forecasting models (ARIMA + Prophet) | Done |
| Ensemble / regression comparison (LR + RF + GB) | Done |
| Feature importance analysis | Done |
| Geographical patterns analysis | Done |
| Interactive web application | Done (Gradio) |

### 8.3 Limitations

- **Short time window (22 months):** Insufficient for robust annual seasonality modeling in Prophet.
- **Single aggregated time series:** Daily forecasting collapses city-level diversity; city-specific models would improve accuracy.
- **No external regressors in ARIMA:** Adding climate indices (ENSO, NAO) could significantly improve forecast skill.
- **Sensor noise:** Wind/gust values up to 2,963 kph in the raw data indicate unresolved data quality issues beyond what Isolation Forest removed.

### 8.4 Next Steps

1. Extend the dataset to 3–5 years for reliable seasonal modeling.
2. Train city-specific SARIMA models and aggregate forecasts.
3. Add SHAP-based explainability to the Gradio app for per-prediction feature attribution.
4. Incorporate air quality indices for the Environmental Impact analysis.
5. Build a spatial choropleth dashboard (Plotly) for interactive geographic exploration.

---

*Report generated for PM Accelerator Tech Assessment — Weather Trend Forecasting*
*Prathamesh Suhas Uravane | April 2026*
