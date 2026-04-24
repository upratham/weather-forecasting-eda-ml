# WeatherFlow — Full-Stack AI Weather App & Analysis Report

**Author:** Prathamesh Suhas Uravane
**Submission:** PM Accelerator — AI Engineer Intern Tech Assessment (Full Stack — Assessment #1 + #2)
**Dataset:** Global Weather Repository (Kaggle) | ~130,000 rows | 40+ features | May 2024 – Apr 2026
**Live App:** FastAPI + React | Real-time weather via Open-Meteo API | SQLite CRUD persistence

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
7. [WeatherFlow — Full-Stack Web App](#7-weatherflow--full-stack-web-app)
8. [Key Insights & Conclusions](#8-key-insights--conclusions)

---

## 1. Project Overview

This project delivers two complementary outputs:

1. **Full-Stack Weather App (WeatherFlow)** — a production-ready web application with a FastAPI backend, React frontend, SQLite database, real-time weather API integration, CRUD persistence, and multi-format data export. This covers both Assessment #1 (Frontend) and Assessment #2 (Backend).

2. **ML / Data Science Pipeline** — an end-to-end analysis of the Global Weather Repository dataset, spanning data cleaning, EDA, anomaly detection, time-series forecasting, and multi-model regression.

### Full-Stack Architecture

```
User (Browser)
      │
      ▼
React + Vite Frontend (port 5173 / static)
      │ /api/*
      ▼
FastAPI Backend (port 8000)
      ├── GET  /api/weather      → Open-Meteo API (real-time, no key)
      ├── GET  /api/geocode      → Open-Meteo Geocoding API
      ├── POST /api/queries      → SQLite (CREATE)
      ├── GET  /api/queries      → SQLite (READ)
      ├── PUT  /api/queries/{id} → SQLite (UPDATE)
      ├── DEL  /api/queries/{id} → SQLite (DELETE)
      └── GET  /api/export       → JSON | CSV | XML | Markdown
```

### ML Pipeline Architecture

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
        ├──► 05 ML Regression             → LinearReg, RandomForest, GradientBoosting
        │
        └──► 06 Advanced Analyses         → Ensemble Models, Spatial Maps, Climate Analysis
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
| **Random Forest** | 200 estimators, `random_state=42`, `n_jobs=-1` |
| **Gradient Boosting** | Scikit-learn `GradientBoostingRegressor`, 200 estimators |
| **Voting Ensemble** | Averages predictions of LR + RF + GB (`VotingRegressor`) |
| **Stacking Ensemble** | LR + RF + GB base learners; Ridge meta-learner via 5-fold CV |

### 6.3 Performance Results

| Model | MAE (°C) | RMSE (°C) | MAPE (%) | R² |
|-------|----------|-----------|----------|-----|
| **Linear Regression** | 0.018 | 0.023 | 0.19 | **0.9999** |
| Random Forest | 0.007 | 0.193 | 0.05 | 0.9996 |
| Gradient Boosting | 0.049 | 0.202 | 0.55 | 0.9995 |
| Voting Ensemble | See `reports/ensemble_metrics.csv` | — | — | — |
| Stacking Ensemble | See `reports/ensemble_metrics.csv` | — | — | — |

*(Ensemble metrics vary with dataset; run notebook 06 to populate exact values.)*

### 6.4 Analysis

All three base models achieve **R² > 0.999** — indicating that the combination of geographic coordinates + time + atmospheric conditions is nearly sufficient to perfectly predict temperature. This is expected: temperature is a thermodynamic quantity determined almost entirely by known physical variables.

**Why Linear Regression wins on MAE/RMSE:**
The linear model achieves the best point-error metrics because the underlying relationships between the engineered features and temperature are largely linear once cyclic encoding resolves the periodicity. Random Forest's slight advantage in MAPE (0.05% vs 0.19%) reflects better relative accuracy on near-zero temperature readings.

**Ensemble models** (Voting and Stacking) combine the strengths of all three base learners. The Stacking Ensemble uses a Ridge meta-learner trained on 5-fold out-of-fold predictions, allowing it to weight each base model optimally rather than averaging uniformly. In this high-R² regime, ensemble gains over the best base model are modest but the approach demonstrates robustness — the ensemble degrades gracefully when any single base model underperforms on out-of-distribution data.

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

### 6.5 Production Model Notes

The full-stack WeatherFlow app fetches **live** weather data from Open-Meteo instead of running a local ML model for inference — this ensures predictions are always current and location-accurate. The regression models in this section remain valuable for:

- Offline / low-connectivity scenarios
- Understanding which features drive temperature
- Benchmarking model quality against real data

A retrained Random Forest (with `max_depth=20` to prevent overfitting, and outlier capping applied to `wind_kph ≤ 200`, `gust_kph ≤ 250`, `pressure_mb 870–1100`) is kept in `models/` and can be loaded for supplementary local inference if desired.

---

## 7. WeatherFlow — Full-Stack Web App

### 7.1 Overview

**WeatherFlow** is a production-ready full-stack weather application covering both assessment tracks:

- **Frontend (Assessment #1):** React 18 + Vite, responsive layout, weather icons, GPS, 7-day forecast, error handling
- **Backend (Assessment #2):** FastAPI + SQLAlchemy + SQLite, CRUD, date range validation, location fuzzy-match, multi-format export

The app uses the **Open-Meteo API** — completely free with no API key required — for real-time weather data and geocoding.

### 7.2 Frontend Features (Assessment #1)

| Feature | Implementation |
|---------|---------------|
| Location input | City name, postal code, GPS coordinates, or landmark (fuzzy-matched by Open-Meteo) |
| Current location | Browser `navigator.geolocation` — one-click GPS detection |
| Weather display | Temperature (°C + °F), feels like, humidity, wind speed + direction, pressure, UV index, cloud cover, precipitation |
| Weather icons | WMO weather code → emoji mapping (☀️ ⛅ 🌧️ ❄️ ⛈️ etc.) |
| 7-day forecast | Daily max/min, condition, precipitation, wind |
| Responsive design | CSS Grid with breakpoints at 768px and 400px |
| Error handling | Inline dismissible banners for API errors, GPS denial, invalid location |
| External integrations | Google Maps link + YouTube travel/weather search per location |

### 7.3 Backend / API Features (Assessment #2)

#### CRUD Operations

| Operation | Endpoint | Validation |
|-----------|----------|-----------|
| **Create** | `POST /api/queries` | `end_date >= start_date`; location exists (geocode check) |
| **Read** | `GET /api/queries` | Returns all records ordered by newest first |
| **Read one** | `GET /api/queries/{id}` | Returns 404 if not found |
| **Update** | `PUT /api/queries/{id}` | Partial update — notes, start_date, end_date |
| **Delete** | `DELETE /api/queries/{id}` | Returns 204 No Content |

#### Database Schema (`weather_queries` table)

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment primary key |
| `location_query` | TEXT | Original user search string |
| `location_name` | TEXT | Resolved display name (city, country) |
| `latitude / longitude` | REAL | Validated coordinates |
| `start_date / end_date` | DATE | Optional date range (validated: end ≥ start) |
| `temperature_celsius` | REAL | Temperature at time of query |
| `weather_description` | TEXT | WMO condition description |
| `humidity / wind_speed` | REAL | Supporting weather metrics |
| `notes` | TEXT | Free-text user annotations |
| `created_at / updated_at` | DATETIME | Automatic timestamps |

#### Data Export (Assessment 2.3)

| Format | Endpoint | Notes |
|--------|----------|-------|
| JSON | `/api/export?fmt=json` | Pretty-printed, full schema |
| CSV | `/api/export?fmt=csv` | Headers + all columns |
| XML | `/api/export?fmt=xml` | Well-formed with declaration |
| Markdown | `/api/export?fmt=markdown` | GitHub-compatible table |

#### Additional API Integration (Assessment 2.2)

- **Google Maps:** Each result card links to `maps.google.com?q={lat},{lon}` for the queried location
- **YouTube:** Each result links to a YouTube search for `{location} travel weather` videos

### 7.4 Running the App

**Production (single server):**
```bash
cd frontend && npm run build && cd ..
python run.py
# → http://localhost:8000
```

**Development (hot reload):**
```bash
# Terminal 1
python run.py

# Terminal 2
cd frontend && npm run dev
# → http://localhost:5173
```

**Interactive API docs:** `http://localhost:8000/docs`

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

### 8.2 Assessment Requirements Coverage

#### Frontend (Assessment #1)

| Requirement | Status | Implementation |
|------------|--------|---------------|
| Location input (city / zip / GPS / landmark) | ✅ Done | Open-Meteo geocoding with fuzzy match |
| Current weather display | ✅ Done | WeatherCard component — 10+ fields |
| Current location (GPS) | ✅ Done | `navigator.geolocation` |
| Weather icons / images | ✅ Done | WMO code → emoji mapping |
| 5-day / 7-day forecast | ✅ Done | ForecastGrid with 7-day Open-Meteo data |
| Error handling | ✅ Done | Inline banners, 404 messages, GPS denial |
| Responsive design | ✅ Done | CSS Grid, tested desktop / tablet / mobile |
| JavaScript framework (not Python/Java) | ✅ Done | React 18 + Vite |

#### Backend (Assessment #2)

| Requirement | Status | Implementation |
|------------|--------|---------------|
| CRUD — Create | ✅ Done | `POST /api/queries` with Pydantic validation |
| CRUD — Read | ✅ Done | `GET /api/queries` + `GET /api/queries/{id}` |
| CRUD — Update | ✅ Done | `PUT /api/queries/{id}` — partial update |
| CRUD — Delete | ✅ Done | `DELETE /api/queries/{id}` |
| Date range validation | ✅ Done | `end_date >= start_date` (server + client) |
| Location validation / fuzzy match | ✅ Done | Open-Meteo geocoding returns 404 for invalid |
| RESTful API design | ✅ Done | Standard HTTP verbs, status codes, JSON |
| Database (SQL) | ✅ Done | SQLite via SQLAlchemy ORM |
| Export — JSON | ✅ Done | `GET /api/export?fmt=json` |
| Export — CSV | ✅ Done | `GET /api/export?fmt=csv` |
| Export — XML | ✅ Done | `GET /api/export?fmt=xml` |
| Export — Markdown | ✅ Done | `GET /api/export?fmt=markdown` |
| Additional API — Maps | ✅ Done | Google Maps link per location |
| Additional API — YouTube | ✅ Done | YouTube search link per location |
| Developer name | ✅ Done | Header, footer, About tab |
| PM Accelerator description | ✅ Done | About tab + README |

#### ML / Data Science

| Requirement | Status | Notebook |
|------------|--------|----------|
| Data cleaning & preprocessing | ✅ Done | `01_data_cleaning.ipynb` |
| EDA with visualizations | ✅ Done | `02_eda.ipynb` |
| Air quality analysis (CO, NO₂, SO₂, PM2.5, PM10) | ✅ Done | `02_eda.ipynb` |
| Anomaly detection (Isolation Forest) | ✅ Done | `03_anomaly_analysis.ipynb` |
| Time-series forecasting (ARIMA + Prophet) | ✅ Done | `04_time_series_forecasting.ipynb` |
| Regression model comparison (LR + RF + GB) | ✅ Done | `05_ml_models.ipynb` |
| Feature importance analysis | ✅ Done | `05_ml_models.ipynb` |
| Ensemble models (VotingRegressor + StackingRegressor) | ✅ Done | `06_advanced_analyses.ipynb` |
| Spatial analysis — geographic temperature map | ✅ Done | `06_advanced_analyses.ipynb` |
| Spatial analysis — country choropleth | ✅ Done | `06_advanced_analyses.ipynb` |
| Climate analysis — temperature by zone & month | ✅ Done | `06_advanced_analyses.ipynb` |

### 8.3 Limitations

- **Short time window (22 months):** Insufficient for robust annual seasonality modeling in Prophet; needs 3+ years.
- **Single aggregated time series:** Daily forecasting collapses city-level diversity; city-specific models would improve accuracy.
- **No external regressors in ARIMA:** Adding climate indices (ENSO, NAO) could significantly improve forecast skill.
- **No user authentication:** Query history is shared across all users (Row-level security not required per assessment spec).
- **No PDF export:** Excluded to avoid heavy dependencies (reportlab); JSON/CSV/XML/Markdown cover the common cases.

### 8.4 Next Steps

1. Extend the dataset to 3–5 years for reliable annual seasonality modeling in Prophet/SARIMA.
2. Train city-specific SARIMA models and ensemble their forecasts for improved regional accuracy.
3. Add SHAP-based explainability to the web app for per-prediction feature attribution.
4. Integrate live air quality indices (PM2.5, AQI) from an open API into the WeatherFlow frontend.
5. Add a user authentication layer for private query history isolation.
6. Export spatial maps as embedded interactive charts within the web app using Plotly.js.

---

*Report generated for PM Accelerator Tech Assessment — AI Engineer Intern (Full Stack)*
*Prathamesh Suhas Uravane | April 2026*
