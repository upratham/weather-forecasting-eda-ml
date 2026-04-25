# Weather Trend Forecasting

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-5.0%2B-FF7C00?style=flat&logo=gradio&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

> **Tech Assessment Submission — PM Accelerator | Data Scientist / Analyst Track**

---

## PM Accelerator Mission

> *"PM Accelerator's mission is to power the careers of aspiring and early-career product managers by providing world-class mentorship, hands-on real-world project experience, and a thriving global community of peers and industry leaders — accelerating every member's path to product leadership."*
>
> — [pmaccelerator.io](https://www.pmaccelerator.io)

---

## Overview

An end-to-end weather trend forecasting pipeline built on the **Global Weather Repository** (Kaggle, ~130K observations, 40+ features). The project covers every stage of the data science lifecycle — from raw data ingestion and cleaning through anomaly detection, time-series forecasting, multi-model regression, and a deployed interactive web application.

### What's Inside

| Stage | Technique |
|-------|-----------|
| Data Cleaning | Missing-value imputation, outlier handling, type normalization |
| EDA | Distribution analysis, correlation heatmaps, temporal trends |
| Anomaly Detection | Isolation Forest (unsupervised) |
| Time-Series Forecasting | ARIMA, Facebook Prophet |
| Regression Modeling | Linear Regression, Random Forest, Gradient Boosting |
| Ensemble Inference | Random Forest on anomaly-cleaned data (live app) |
| Web Application | Gradio interactive temperature predictor |

---

## Dataset

| Detail | Value |
|--------|-------|
| Source | [Kaggle — Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/code) |
| Rows (raw) | ~130,000+ |
| Features | 40+ |
| Time span | May 2024 – April 2026 |
| Key target | `temperature_celsius` |
| Geography | Cities worldwide |

---

## Results at a Glance

### Regression Models (test set, anomaly-cleaned data)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 0.018 °C | 0.023 °C | **0.9999** |
| Random Forest | 0.007 °C | 0.193 °C | 0.9996 |
| Gradient Boosting | 0.049 °C | 0.202 °C | 0.9995 |
| Gradio App RF | 1.249 °C | 1.844 °C | 0.9594 |

### Time-Series Models (daily temperature forecast)

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| ARIMA | 1.202 °C | 1.739 °C | 8.92% |
| Prophet | 4.390 °C | 5.771 °C | 25.99% |

---

## Project Structure

```
weather-forecasting-eda-ml/
├── data/
│   ├── raw/                               # GlobalWeatherRepository.csv (~35 MB)
│   ├── processed/                         # Cleaned, scored, and filtered datasets
│   └── cleaned/                           # weather_cleaned.csv
├── notebooks/
│   ├── 01_data_cleaning.ipynb             # Missing values, outliers, type fixes
│   ├── 02_eda.ipynb                       # Distributions, correlations, trends
│   ├── 03_anomaly_analysis.ipynb          # Isolation Forest anomaly detection
│   ├── 04_time_series_forecasting.ipynb   # ARIMA + Prophet daily forecasting
│   └── 05_ml_models.ipynb                 # Regression model comparison
├── src/
│   ├── preprocessing.py                   # Data loading & anomaly scoring
│   ├── features.py                        # Feature engineering & app columns
│   ├── train.py                           # Model training helpers
│   ├── eval.py                            # Regression & forecast metrics
│   └── visualize.py                       # Plotting utilities
├── models/                                # Persisted .joblib model files
├── reports/
│   ├── figures/                           # PNG charts from analyses
│   │   ├── anomaly_temperature_humidity.png
│   │   ├── anomaly_top_countries.png
│   │   └── forecast_comparison.png
│   ├── forecast_metrics.csv               # ARIMA / Prophet test metrics
│   ├── temperature_model_metrics.csv      # Regression model metrics
│   └── weather_forecasting_report.md      # Full analysis report (this project)
├── app.py                                 # Gradio web application
├── requirements.txt
└── pyproject.toml
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/upratham/weather-forecasting-eda-ml.git
cd weather-forecasting-eda-ml
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Place `GlobalWeatherRepository.csv` in `data/raw/`. Download from Kaggle:

```bash
kaggle datasets download -d nelgiriyewithana/global-weather-repository -p data/raw/ --unzip
```

---

## Running the Notebooks

Run notebooks in order for the full pipeline:

```bash
jupyter notebook
```

Open in sequence:

| Notebook | What it does |
|----------|-------------|
| `01_data_cleaning.ipynb` | Cleans raw CSV, handles missing values and type errors |
| `02_eda.ipynb` | Distributions, correlation heatmaps, temporal trends |
| `03_anomaly_analysis.ipynb` | Isolation Forest — flags and removes anomalous readings |
| `04_time_series_forecasting.ipynb` | Trains and compares ARIMA vs Prophet |
| `05_ml_models.ipynb` | Trains Linear Regression, Random Forest, Gradient Boosting |

---

## Running the Gradio App

```bash
python app.py
```

The app launches in your browser. On first run it trains a Random Forest model automatically, then serves an interactive form where you can enter any weather snapshot (latitude, longitude, pressure, humidity, cloud cover, wind, precipitation, visibility, UV index) and receive an instant temperature prediction in both **Celsius** and **Fahrenheit** with full input validation and model metadata.

---

## Key Findings

- **Location is the strongest predictor.** Latitude and longitude account for the largest share of temperature variance across all models.
- **Random Forest achieves near-perfect R² (0.9996)** on the multi-feature regression task against held-out data.
- **ARIMA outperforms Prophet** on the daily time-series task (MAPE 8.92% vs 25.99%), as the 22-month window is too short for Prophet's seasonal decomposition to converge reliably.
- **Anomaly detection flagged ~5.2% of records** — concentrated in Indonesia, China, India, and Gulf states — driven by sensor malfunctions and extreme climate conditions.
- **Humidity and UV index are the strongest non-spatial predictors** of temperature after location and time.

---

## Full Report

A detailed written report with embedded visualizations, methodology explanations, model evaluation tables, and findings is available at:

**[`reports/weather_forecasting_report.md`](reports/weather_forecasting_report.md)**

---

## Requirements

Key packages (see [`requirements.txt`](requirements.txt) for full list):

```
pandas>=2.0.0        numpy>=1.24.0
scikit-learn>=1.3.0  statsmodels>=0.14.0
prophet>=1.1.0       gradio>=5.0.0
matplotlib>=3.7.0    seaborn>=0.12.0
plotly>=5.15.0       xgboost>=1.7.0
joblib>=1.3.0        jupyter>=1.0.0
```

---

## Demo Video

*A 1–2 minute screen recording walking through the notebooks and the live Gradio app:*
`[Add your Google Drive / YouTube / Vimeo link here]`

---

## Author

**Prathamesh Suhas Uravane**
Submitted for PM Accelerator — Data Scientist / Analyst Tech Assessment
