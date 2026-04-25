# WeatherFlow — AI Weather Temperature Predictor

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat&logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-FF7C00?style=flat&logo=gradio&logoColor=white)
![Modal](https://img.shields.io/badge/Modal-deployed-6C47FF?style=flat&logo=modal&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

> **Tech Assessment Submission — PM Accelerator | AI Engineer Intern (Full Stack)**

---

## PM Accelerator

**Product Manager Accelerator (PMA)** is a leading community-driven program designed to help professionals break into and advance within product management. Through mentorship from experienced PMs, structured coaching, hands-on projects, and a thriving peer community, PMA accelerates careers in product management across tech, fintech, healthtech, and beyond.

> [linkedin.com/company/product-manager-accelerator](https://www.linkedin.com/company/product-manager-accelerator)

---

## Overview

**WeatherFlow** is an end-to-end AI weather intelligence application. It trains a **RandomForest** temperature predictor on the Global Weather Repository, exposes it through a **Gradio** web interface, and deploys it serverlessly to **Modal** — all from a single automated pipeline.

### Key Features

- **Gradio UI** — interactive weather input form with live temperature predictions (°C + °F)
- **Automated pipeline** — `train_eval_deploy()` trains the model, prints metrics, and deploys to Modal in one call
- **Modal deployment** — persistent serverless endpoint with a Modal Volume to cache the trained model across cold starts
- **Remote prediction** — `predict_remote()` calls the live Modal endpoint from notebooks or scripts
- **Structured logging** — all operations (data loading, training, deployment, inference) emit timestamped log lines
- **Full ML notebook suite** — EDA, anomaly detection, time-series forecasting, and ensemble models

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Gradio 4.0+ (glassmorphism theme) |
| ML | scikit-learn RandomForestRegressor |
| Deployment | Modal (serverless ASGI) |
| Remote client | gradio_client |
| Data | [Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository) |
| Containerisation | Docker (Debian Slim, Python 3.11) |
| Notebooks | Jupyter, pandas, numpy, statsmodels, XGBoost, LightGBM |

---

## Project Structure

```
weather-forecasting-eda-ml/
├── app.py                             # Gradio app — train, predict, UI
├── src/
│   ├── pipeline.py                    # Train → eval → Modal deploy pipeline
│   ├── modal_deploy.py                # Modal app definition (ASGI)
│   ├── preprocessing.py               # Data loading, cleaning, anomaly detection
│   ├── features.py                    # Feature engineering helpers
│   ├── train.py                       # Model training utilities
│   ├── eval.py                        # Regression + forecast metrics
│   └── visualize.py                   # Plotting utilities
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_anomaly_analysis.ipynb
│   ├── 04_time_series_forecasting.ipynb
│   ├── 05_ml_models.ipynb
│   └── 06_advanced_analyses.ipynb     # Ensemble, spatial maps, climate analysis
├── data/
│   ├── raw/                           # GlobalWeatherRepository.csv (~35 MB)
│   └── processed/                     # Cleaned + anomaly-filtered datasets
├── models/                            # Saved model + metadata (auto-created)
│   ├── gradio_temperature_model.joblib
│   ├── gradio_temperature_model_meta.json
│   └── endpoint_url.txt               # Cached Modal endpoint URL after deploy
├── reports/
│   ├── weather_forecasting_report.md
│   └── temperature_model_metrics.csv
├── Dockerfile
├── .env                               # Modal credentials (gitignored)
├── requirements.txt
└── pyproject.toml
```

---

## Getting Started

### 1. Clone

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

```bash
kaggle datasets download -d nelgiriyewithana/global-weather-repository -p data/raw/ --unzip
```

### 5. Configure Modal credentials

Create a `.env` file in the project root (get your token from [modal.com/settings/tokens](https://modal.com/settings/tokens)):

```
MODAL_TOKEN_ID=your_token_id_here
MODAL_TOKEN_SECRET=your_token_secret_here
```

---

## Running the App

### Local Gradio server

```bash
python app.py
```

Opens at **http://localhost:7860**. The model trains automatically on first launch (~30 s) and is cached to `models/` for subsequent runs.

### Docker

```bash
docker build -t weatherflow .
docker run -p 7860:7860 weatherflow
```

---

## Modal Deployment

### One-off commands

```bash
# Hot-reload dev server (temporary URL, useful for testing)
modal serve src/modal_deploy.py

# Permanent deploy to Modal cloud
modal deploy src/modal_deploy.py
```

### Automated pipeline (train → eval → deploy)

From a notebook or script:

```python
from src.pipeline import train_eval_deploy, predict_remote, get_endpoint_url

# Train the model, print metrics, deploy to Modal, cache the endpoint URL
result = train_eval_deploy()
print(result["endpoint_url"])
# → https://<user>--weather-temperature-predictor-serve.modal.run
```

The endpoint URL is saved to `models/endpoint_url.txt` and reused by `get_endpoint_url()` without redeploying.

To force a retrain and redeploy:

```python
result = train_eval_deploy(force_retrain=True)
```

### Remote prediction

```python
from src.pipeline import predict_remote, get_endpoint_url

celsius, fahrenheit, details = predict_remote(
    get_endpoint_url(),
    latitude=12.97,   longitude=77.59,
    date_value="2026-04-25", hour=9,
    pressure_mb=1012.0, humidity=68.0,  cloud=35.0,
    wind_kph=14.0,    gust_kph=22.0,   precip_mm=0.0,
    visibility_km=10.0, uv_index=5.5,
)
print(f"{celsius:.2f} °C / {fahrenheit:.2f} °F")
```

---

## Logging

All modules emit structured log lines via Python's standard `logging`. Run locally:

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
```

`python app.py` configures this automatically. Sample output:

```
10:32:01  INFO      app                 Loading model from models/gradio_temperature_model.joblib
10:32:01  INFO      app                 Model loaded (trained at 2026-04-25T10:00:00)
10:32:04  INFO      src.pipeline        Deploying to Modal…
10:32:18  INFO      src.pipeline        Deployed. Endpoint: https://user--weather-temperature-predictor-serve.modal.run
10:32:20  DEBUG     app                 Prediction: 24.31 °C / 75.76 °F
```

| Module | What it logs |
|--------|-------------|
| `app` | Model load/train, metrics after save, each prediction (DEBUG) |
| `src.pipeline` | Deploy start/result/error, remote prediction call + result |
| `src.preprocessing` | Row counts on load, duplicates dropped, anomaly count + % |
| `src.train` | Each model fit with sample count, metrics table |

---

## ML Notebooks

Run in order for the complete pipeline:

```bash
jupyter notebook
```

| Notebook | Description |
|----------|-------------|
| `01_data_cleaning.ipynb` | Missing values, outliers, type normalisation |
| `02_eda.ipynb` | Distributions, correlations, temporal trends, air quality |
| `03_anomaly_analysis.ipynb` | Isolation Forest anomaly detection |
| `04_time_series_forecasting.ipynb` | ARIMA vs Prophet comparison |
| `05_ml_models.ipynb` | Linear Regression, Random Forest, Gradient Boosting |
| `06_advanced_analyses.ipynb` | Ensemble models, spatial maps, climate zone analysis |

---

## ML Results

### Regression Models (test set, anomaly-cleaned data)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 0.018 °C | 0.023 °C | **0.9999** |
| Random Forest | 0.007 °C | 0.193 °C | 0.9996 |
| Gradient Boosting | 0.049 °C | 0.202 °C | 0.9995 |
| Voting Ensemble (LR+RF+GB) | see `reports/temperature_model_metrics.csv` | — | — |
| Stacking Ensemble (Ridge meta) | see `reports/temperature_model_metrics.csv` | — | — |

### Time-Series Models (daily temperature forecast)

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| ARIMA | 1.202 °C | 1.739 °C | 8.92% |
| Prophet | 4.390 °C | 5.771 °C | 25.99% |

Full findings in **[`reports/weather_forecasting_report.md`](reports/weather_forecasting_report.md)**

---

## Author

**Prathamesh Suhas Uravane**
Submitted for PM Accelerator — AI Engineer Intern Technical Assessment (Full Stack)
