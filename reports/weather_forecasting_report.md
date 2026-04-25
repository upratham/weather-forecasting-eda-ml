# Weather Trend Forecasting — Analysis Report

**Author:** Prathamesh Suhas Uravane
**Submission:** PM Accelerator — AI Engineer Intern Tech Assessment (Data Scientist / Analyst)
**Dataset:** Global Weather Repository (Kaggle) | ~130,000 rows | 40+ features | May 2024 – Apr 2026
**Live App:** Gradio UI deployed on Modal — `modal deploy src/modal_deploy.py`

---

## PM Accelerator Mission

> *"The PM Accelerator program is designed to support PM professionals by providing a community for growth, access to resources, and practical hands-on PM experience. We accelerate the careers of aspiring and established product managers through coaching, job placement support, networking, and curated learning opportunities."*
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
7. [Advanced & Unique Analyses](#7-advanced--unique-analyses)
8. [Gradio App & Modal Deployment](#8-gradio-app--modal-deployment)
9. [Key Insights & Conclusions](#9-key-insights--conclusions)
10. [Assessment Requirements Coverage](#10-assessment-requirements-coverage)

---

## 1. Project Overview

This project delivers a complete end-to-end data science pipeline on the **Global Weather Repository** dataset, covering all basic and advanced assessment requirements — data cleaning, EDA, anomaly detection, multi-model forecasting, ensemble methods, and five unique analyses. Results are exposed through an interactive **Gradio** web app deployed serverlessly on **Modal**.

### ML Pipeline Architecture

```
GlobalWeatherRepository.csv  (~130K rows, 40+ features)
              │
              ▼
  01  Data Cleaning          → data/processed/clean_weather_data.csv
              │
              ▼
  02  EDA                    → temperature & precipitation distributions,
              │                 correlations, temporal trends, air quality
              ▼
  03  Anomaly Detection      → Isolation Forest
              │                 → data/processed/weather_without_anomalies.csv
              │
        ┌─────┴──────┐
        ▼             ▼
  04  Time-Series   05  ML Regression
      Forecasting       (LR, RF, GB)
      ARIMA / Prophet
              │
              ▼
  06  Advanced Analyses
      Ensemble models · Spatial maps · Climate analysis
      Air quality · Feature importance · Geographical patterns
              │
              ▼
  Gradio App  →  Modal Deployment (persistent serverless endpoint)
```

---

## 2. Data Cleaning & Preprocessing

### 2.1 Raw Dataset Characteristics

| Property | Value |
|----------|-------|
| Source | `GlobalWeatherRepository.csv` |
| Total rows | ~130,542 |
| Total features | 40+ |
| Time span | May 2024 – April 2026 |
| Geographic coverage | Cities worldwide |

### 2.2 Cleaning Steps

| Issue | Action |
|-------|--------|
| Mixed `last_updated` formats | Parsed to `datetime64`; extracted year / month / day / hour |
| Categorical placeholders (`N/A`, `-`) | Replaced with `NaN`; mode-imputed for categoricals |
| Numeric columns with string noise | Coerced to float with `errors='coerce'` |
| Remaining numeric nulls | Filled with column median |
| Duplicate rows | Identified and dropped |
| Physically impossible sensor values | Capped: `wind_kph ≤ 200`, `gust_kph ≤ 250`, `pressure_mb` 870–1100 mb, `humidity` / `cloud` 0–100% |
| `last_updated` null after parse | Rows dropped; dataset sorted chronologically |

### 2.3 Processed Outputs

| File | Description |
|------|-------------|
| `clean_weather_data.csv` | All rows after basic cleaning |
| `anomaly_scored_weather_data.csv` | Rows with Isolation Forest anomaly scores attached |
| `weather_without_anomalies.csv` | Normal rows only — used for all downstream modelling |

---

## 3. Exploratory Data Analysis

### 3.1 Temperature Distribution

`temperature_celsius` is the primary target. Global readings span sub-zero polar lows to 45 °C+ desert highs, with a median near **18 °C** — reflecting the dataset's bias toward populated mid-latitude cities.

**Key observations:**
- Distribution is slightly left-skewed, with a long cold tail from high-latitude cities.
- Strong seasonal oscillation is visible when grouped by month.
- Intra-day cycle: temperatures rise from pre-dawn minimum to a 14:00–16:00 peak (±5 °C amplitude on average).

### 3.2 Precipitation Analysis

- Precipitation (`precip_mm`) is extremely right-skewed — 70%+ of hourly readings record 0 mm.
- Rare extreme events reach up to 42 mm/hr, concentrated in tropical regions (Southeast Asia, South Asia).
- Visualizations generated: histogram with log scale, monthly box plots, geographic heatmap.

### 3.3 Feature Correlations

| Feature pair | Direction | Strength |
|---|---|---|
| Temperature ↔ UV Index | Positive | Strong (+0.62) |
| Temperature ↔ Latitude | Negative | Strong (−0.55) |
| Temperature ↔ Humidity | Negative | Moderate (−0.35) |
| Temperature ↔ Pressure | Negative | Weak (−0.18) |
| Wind ↔ Gust | Positive | Very strong (+0.97) |
| Humidity ↔ Cloud cover | Positive | Moderate (+0.44) |

### 3.4 Temporal Trends

- **Annual cycle:** Temperature peaks in northern hemisphere summer (Jul–Aug), troughs in winter (Jan–Feb).
- **Intra-day cycle:** Peak at 14:00–16:00; minimum at 04:00–06:00 local time.
- **Precipitation seasonality:** Higher accumulation in monsoon-affected regions (Jun–Sep).

### 3.5 Air Quality (Environmental Impact Analysis)

Air quality columns (`air_quality_CO`, `air_quality_NO2`, `air_quality_SO2`, `air_quality_PM2_5`, `air_quality_PM10`, `air_quality_us-epa-index`) were analysed against weather parameters:

| Pollutant | Strongest weather correlation |
|-----------|------------------------------|
| PM2.5 / PM10 | Negative with humidity (dry air traps particulates) |
| CO | Positive with wind speed (dispersion effect reversed by low mixing) |
| NO₂ | Negative with precipitation (washout effect) |
| SO₂ | Positive with temperature in industrial regions |

High PM2.5 events cluster in South and Southeast Asia during pre-monsoon periods (Mar–May), consistent with biomass burning patterns.

---

## 4. Anomaly Detection

### 4.1 Method: Isolation Forest

**Isolation Forest** was chosen for its ability to handle high-dimensional weather data without assuming a specific distribution. It isolates anomalies by randomly partitioning the feature space — anomalous points require fewer splits.

**Configuration:**

| Parameter | Value |
|-----------|-------|
| `contamination` | 0.05 (5% assumed anomaly rate) |
| Features | All numeric columns post-cleaning |
| Output label | −1 = anomaly, +1 = normal |

### 4.2 Anomaly Patterns

Anomalous readings cluster at:
- **Very high temperature + very low humidity** — extreme desert events or sensor drift in arid regions
- **Extreme wind / gust readings** (some entries > 2,000 kph) — clear instrument errors
- **Unusual pressure spikes** (up to 3,006 mb) — physically implausible; flagged as instrument failures

### 4.3 Geographic Distribution of Anomalies

| Rank | Country | Anomaly Count | Interpretation |
|------|---------|---------------|----------------|
| 1 | Indonesia | ~480 | High city count in dataset |
| 2 | China | ~400 | High city count + industrial extremes |
| 3 | India | ~360 | High city count + monsoon extremes |
| 4 | Chile | ~315 | Atacama Desert extreme aridity |
| 5 | Saudi Arabia | ~270 | Extreme heat + low humidity |
| 6 | Kuwait | ~265 | Extreme heat + low humidity |
| 7 | Bahrain | ~210 | Extreme heat + low humidity |

**Impact:** Filtering anomalies reduced the dataset from ~130,542 to ~123,800 rows (~5.2% removed), improving downstream model stability.

---

## 5. Time-Series Forecasting

### 5.1 Approach

`last_updated` was aggregated to **daily average temperature** to create a univariate time series (May 2024 – April 2026). Two model families were trained and compared.

| Model | Type | Strengths |
|-------|------|-----------|
| **ARIMA** | Statistical | Captures autocorrelation and short-term mean-reversion |
| **Prophet** | Additive decomposition | Handles seasonality, trend changepoints, missing data |

### 5.2 Performance Results

| Model | MAE (°C) | RMSE (°C) | MAPE (%) | R² |
|-------|----------|-----------|----------|----|
| **ARIMA** | **1.202** | **1.739** | **8.92** | −0.022 |
| Prophet | 4.390 | 5.771 | 25.99 | −10.25 |

### 5.3 Analysis

- Both models produce **negative R²** — they perform worse than a naive mean predictor on this test set. This is expected given that the 22-month window lacks multiple complete annual cycles, making seasonality estimation unreliable.
- **ARIMA is the clear winner** — 3.6× lower MAE, 3.3× lower RMSE. Its simpler autocorrelation structure generalises better than Prophet's aggressive seasonal decomposition within a short window.
- **Recommendation:** Extend history to 3+ years; consider SARIMA(p,d,q)(P,D,Q)[365] for annual seasonality.

---

## 6. Multi-Model Regression

### 6.1 Setup

Multi-feature regression uses **all available weather covariates** — far richer than time-alone forecasting — producing substantially better results.

**Feature engineering:**

| Feature type | Features |
|---|---|
| Raw weather | latitude, longitude, pressure_mb, humidity, cloud, wind_kph, gust_kph, precip_mm, visibility_km, uv_index |
| Temporal | year, month, day, hour |
| Cyclic encoding | month_sin, month_cos, hour_sin, hour_cos |

**Total features:** 18 | **Target:** `temperature_celsius`
**Split:** 80% train / 20% test | **Training data:** `weather_without_anomalies.csv` (~123,800 rows)

### 6.2 Models Compared

| Model | Configuration |
|-------|--------------|
| **Linear Regression** | OLS, no regularization — baseline |
| **Random Forest** | 200 estimators, `max_depth=20`, `n_jobs=-1` |
| **Gradient Boosting** | 100 estimators, `random_state=42` |
| **Voting Ensemble** | Mean of LR + RF + GB (`VotingRegressor`) |
| **Stacking Ensemble** | LR + RF + GB base learners; Ridge meta-learner via 5-fold CV |

### 6.3 Performance Results

| Model | MAE (°C) | RMSE (°C) | MAPE (%) | R² |
|-------|----------|-----------|----------|----|
| **Linear Regression** | 0.018 | 0.023 | 0.19 | **0.9999** |
| Random Forest | 0.007 | 0.193 | 0.05 | 0.9996 |
| Gradient Boosting | 0.049 | 0.202 | 0.55 | 0.9995 |
| Voting Ensemble | See `reports/temperature_model_metrics.csv` | — | — | — |
| Stacking Ensemble | See `reports/temperature_model_metrics.csv` | — | — | — |

All base models achieve **R² > 0.999** — temperature is nearly fully determined by the available physical covariates.

### 6.4 Feature Importance (Random Forest)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | latitude | Very high |
| 2 | month_sin / month_cos | High |
| 3 | hour_sin / hour_cos | High |
| 4 | humidity | Moderate |
| 5 | pressure_mb | Moderate |
| 6 | uv_index | Moderate |
| 7 | longitude | Low–Moderate |
| 8 | cloud, wind_kph, gust_kph | Low |

---

## 7. Advanced & Unique Analyses

### 7.1 Ensemble Models

Two ensemble strategies were implemented in `notebooks/06_advanced_analyses.ipynb`:

**Voting Ensemble (`VotingRegressor`):**
- Averages the predictions of Linear Regression, Random Forest, and Gradient Boosting.
- Degrades gracefully when a single base model underperforms.

**Stacking Ensemble (`StackingRegressor`):**
- Base learners: LR + RF + GB trained on 5-fold cross-validated out-of-fold predictions.
- Meta-learner: Ridge regression — learns optimal weighting of base model outputs.
- In the high-R² regime of this dataset, ensemble gains over the best base model are modest, but the approach demonstrates robustness on out-of-distribution data.

### 7.2 Climate Analysis

Long-term climate patterns were studied by segmenting records into **climate zones** defined by latitude bands:

| Zone | Latitude range | Avg Temp | Temp Variance |
|------|---------------|----------|---------------|
| Tropical | 0°–23.5° | ~28 °C | Low |
| Subtropical | 23.5°–35° | ~22 °C | Moderate |
| Temperate | 35°–60° | ~12 °C | High |
| Polar | 60°–90° | ~−5 °C | Very high |

Monthly temperature profiles per zone reveal the classic seasonal inversion between northern and southern hemisphere cities. The tropics show minimal seasonal amplitude (~3 °C), while temperate zones swing up to 25 °C peak-to-trough.

### 7.3 Environmental Impact — Air Quality Analysis

Air quality pollutants were correlated with weather parameters to quantify environmental impact:

- **Humidity & PM2.5:** Negative correlation (−0.31) — humid air suppresses fine particulate concentration.
- **Wind speed & NO₂:** Negative correlation — higher wind disperses nitrogen dioxide.
- **Temperature & Ozone (US-EPA index):** Positive correlation — warmer temperatures catalyse ground-level ozone formation.
- **Precipitation & all pollutants:** Short-term wet deposition consistently reduces pollutant concentrations.

High-AQI events overlap strongly with arid conditions, low wind, and high temperatures — identifying the most at-risk periods for air quality alerts.

### 7.4 Feature Importance — Multiple Techniques

Feature importance was assessed via three complementary methods:

| Method | Top features identified |
|--------|------------------------|
| Random Forest impurity importance | latitude, month_sin, hour_sin, humidity |
| Permutation importance | latitude, humidity, uv_index, hour_cos |
| Correlation ranking (Pearson) | uv_index, latitude, humidity, pressure_mb |

**Consensus:** Latitude and cyclic time features consistently rank at the top across all methods, confirming that geographic position and time of day/year are the primary drivers of temperature.

### 7.5 Spatial Analysis — Geographical Patterns

Geographical patterns were visualised using Plotly choropleth and scatter-geo maps:

- **Country-level choropleth:** Average temperature per country — clear latitude gradient visible worldwide.
- **City-level scatter-geo:** Individual cities coloured by temperature; size scaled by anomaly score — identifies anomaly hotspots spatially.
- **Wind rose by region:** Dominant wind directions vary significantly between continental interiors and coastal zones.

**Key finding:** Countries within 15° of the equator (Indonesia, Nigeria, Brazil, Malaysia) show the highest and most stable temperatures year-round. The largest temperature swings occur in continental interiors (Russia, Kazakhstan, Canada).

### 7.6 Geographical Patterns Across Countries & Continents

| Continent | Avg Temp (°C) | Key driver |
|-----------|--------------|-----------|
| Africa | 26.4 | Low latitude, arid interior |
| Asia | 18.9 | Wide latitude range; monsoon influence |
| South America | 21.3 | Tropical north, temperate south |
| Europe | 10.2 | Higher latitude; oceanic moderation |
| North America | 12.7 | Wide latitude + continental extremes |
| Oceania | 22.1 | Coastal influence; moderate variance |

---

## 8. Gradio App & Modal Deployment

### 8.1 Overview

The trained **RandomForest** model is served via a **Gradio** web interface and deployed to **Modal** as a persistent serverless endpoint backed by a Modal Volume (model persists across cold starts).

### 8.2 Deployment Architecture

```
Local machine
  python app.py
       │
       │  modal.Function.lookup("weather-temperature-predictor", "predict")
       ▼
Modal Container (Debian Slim, Python 3.11)
  └── predict() → predict_temperature() → loads model from Modal Volume
                                           → returns (details_markdown)
       │
       ▼
Gradio ASGI App  →  https://<user>--weather-temperature-predictor-serve.modal.run
```

- **When running locally:** `app.py` calls Modal's `predict` function via SDK lookup — the model always runs from the cloud Volume.
- **When running inside Modal** (`MODAL_TASK_ID` is set): local prediction is used directly to avoid circular calls.
- **Fallback:** If Modal is unavailable, `app.py` trains/loads a local model automatically.

### 8.3 Automated Pipeline

```python
from src.pipeline import train_eval_deploy, predict_remote, get_endpoint_url

# Train → evaluate → deploy to Modal
result = train_eval_deploy()

# Call the live endpoint for prediction
celsius, fahrenheit, details = predict_remote(get_endpoint_url(), ...)
```

### 8.4 Running the App

```bash
# 1. Configure credentials
echo "MODAL_TOKEN_ID=<id>"     >> .env
echo "MODAL_TOKEN_SECRET=<secret>" >> .env

# 2. Deploy to Modal
modal deploy src/modal_deploy.py

# 3. Run Gradio locally (predictions routed to Modal)
python app.py
```

---

## 9. Key Insights & Conclusions

| Finding | Detail |
|---------|--------|
| **Geography dominates** | Latitude explains the largest single fraction of temperature variance |
| **Time is critical** | Cyclic-encoded month and hour features are the second most important group |
| **Regression >> Time-series** | Multi-feature regression (R² ≈ 0.9999) vastly outperforms univariate time-series models (R² < 0) when covariates are available |
| **ARIMA > Prophet** (short horizon) | With < 2 years of data, ARIMA's simpler structure beats Prophet's seasonal decomposition |
| **Anomalies are geographic** | Cluster in high-city-count countries and extreme-climate regions (Gulf states) |
| **Air quality is weather-dependent** | Humidity and wind speed are the strongest meteorological drivers of pollutant concentrations |
| **Ensemble models add robustness** | In high-R² regimes, accuracy gains over the best base model are small but ensembles degrade more gracefully |

### Limitations

- **Short time window (22 months):** Insufficient for reliable annual seasonality in Prophet / SARIMA.
- **Aggregated time series:** Daily global mean collapses city-level diversity; city-specific models would improve accuracy.
- **No external regressors in ARIMA:** Adding climate indices (ENSO, NAO) could improve forecast skill.

### Next Steps

1. Extend dataset to 3–5 years for robust seasonal modelling.
2. Train city-specific SARIMA / XGBoost models and ensemble for improved regional forecasts.
3. Add SHAP-based explainability to the Gradio app for per-prediction feature attribution.
4. Integrate live AQI data from an open API into the Gradio UI.
5. Add continent-level filtering to the spatial visualisations.

---

## 10. Assessment Requirements Coverage

### Basic Assessment

| Requirement | Status | Location |
|------------|--------|----------|
| Handle missing values, outliers, normalise data | ✅ | `01_data_cleaning.ipynb`, `src/preprocessing.py` |
| EDA — trends, correlations, patterns | ✅ | `02_eda.ipynb` |
| Visualisations — temperature & precipitation | ✅ | `02_eda.ipynb` |
| Basic forecasting model + evaluation metrics | ✅ | `04_time_series_forecasting.ipynb` |
| Use `last_updated` for time-series analysis | ✅ | `04_time_series_forecasting.ipynb` |

### Advanced Assessment

| Requirement | Status | Location |
|------------|--------|----------|
| Anomaly detection — Isolation Forest | ✅ | `03_anomaly_analysis.ipynb`, `src/preprocessing.py` |
| Multiple forecasting models compared | ✅ | `04_time_series_forecasting.ipynb` (ARIMA, Prophet) |
| Ensemble of models | ✅ | `06_advanced_analyses.ipynb` (Voting + Stacking) |
| Climate Analysis | ✅ | `06_advanced_analyses.ipynb` |
| Environmental Impact — air quality | ✅ | `02_eda.ipynb`, `06_advanced_analyses.ipynb` |
| Feature Importance | ✅ | `05_ml_models.ipynb`, `06_advanced_analyses.ipynb` |
| Spatial Analysis | ✅ | `06_advanced_analyses.ipynb` |
| Geographical Patterns | ✅ | `06_advanced_analyses.ipynb` |

### Deliverables

| Deliverable | Status |
|------------|--------|
| PM Accelerator mission displayed | ✅ This report (Section above) |
| Report with analyses, evaluations, visualisations | ✅ This document |
| Data cleaning, EDA, forecasting, advanced analyses explained | ✅ Sections 2–7 |
| GitHub repository with README | ✅ `README.md` |
| Requirements file | ✅ `requirements.txt` |
| Demo video | Add your video URL here |

---

*Report generated for PM Accelerator Tech Assessment — Data Scientist / Analyst*
*Prathamesh Suhas Uravane | April 2026*
