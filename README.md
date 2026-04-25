# WeatherFlow — Full-Stack AI Weather App

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?style=flat&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react&logoColor=black)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=flat&logo=sqlite&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat&logo=docker&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-deployed-000000?style=flat&logo=vercel&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

> **Tech Assessment Submission — PM Accelerator | AI Engineer Intern (Full Stack)**

---

## PM Accelerator

**Product Manager Accelerator (PMA)** is a leading community-driven program designed to help professionals break into and advance within product management. Through mentorship from experienced PMs, structured coaching, hands-on projects, and a thriving peer community, PMA accelerates careers in product management across tech, fintech, healthtech, and beyond. PMA offers bootcamps, 1-on-1 coaching, portfolio-building workshops, and job placement support — empowering aspiring and current PMs to land roles at top companies and build products that matter.

> [linkedin.com/company/product-manager-accelerator](https://www.linkedin.com/company/product-manager-accelerator)

---

## Overview

**WeatherFlow** is a full-stack weather intelligence application built for the PM Accelerator AI Engineer Intern assessment (both Tech Assessment #1 and #2). It combines a **FastAPI** backend with a **React + Vite** frontend to deliver real-time weather data, CRUD-persistent query history, multi-format data export, and external API integrations — all without requiring any API keys.

### Assessments Covered

| Requirement | Implementation |
|------------|---------------|
| **Frontend (Assessment 1)** | React 18 + Vite, responsive CSS Grid, weather icons, error handling |
| Location input (city / zip / GPS / landmark) | Open-Meteo geocoding with fuzzy match |
| Current weather display | Temperature (°C + °F), humidity, wind, UV, pressure, cloud, precipitation |
| Current location via GPS | Browser Geolocation API |
| 5-day / 7-day forecast | Open-Meteo forecast API, WMO weather code icons |
| Responsive design | CSS Grid + Media Queries (desktop, tablet, mobile) |
| Error handling | Inline banners with dismissal, validation on all inputs |
| **Backend (Assessment 2)** | FastAPI + SQLAlchemy + SQLite |
| CRUD — Create | `POST /api/queries` with date range and location validation |
| CRUD — Read | `GET /api/queries` + `GET /api/queries/{id}` |
| CRUD — Update | `PUT /api/queries/{id}` with inline edit UI |
| CRUD — Delete | `DELETE /api/queries/{id}` with confirmation |
| Date range validation | Server-side: `end_date >= start_date` check |
| Location validation | Open-Meteo geocoding — returns 404 if location doesn't exist |
| Data export | JSON, CSV, XML, Markdown via `GET /api/export?fmt=` |
| Additional API integration | Google Maps link + YouTube search per queried location |
| Developer name + PM Accelerator info | Header, footer, and dedicated About tab |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI 0.111, Python 3.9+ |
| Database | SQLite via SQLAlchemy 2.0 (ORM) |
| HTTP client | httpx (async) |
| Frontend | React 18, Vite 5, CSS Modules |
| Weather data | [Open-Meteo](https://open-meteo.com) — free, no API key required |
| Geocoding | Open-Meteo Geocoding API — fuzzy location matching |
| ML (notebooks) | scikit-learn, statsmodels, Prophet, XGBoost |
| Containerisation | Docker (multi-stage: Node 20 + Python 3.11-slim) |
| Serverless deploy | Vercel (Python runtime via Mangum + static React build) |

---

## Project Structure

```
weather-forecasting-eda-ml/
├── backend/                           # FastAPI application
│   ├── __init__.py
│   ├── main.py                        # All routes: weather, CRUD, export, static serving
│   ├── database.py                    # SQLAlchemy engine + session factory
│   ├── models.py                      # WeatherQuery ORM model
│   └── schemas.py                     # Pydantic request/response schemas
│
├── frontend/                          # React + Vite application
│   ├── package.json
│   ├── vite.config.js                 # Dev proxy → FastAPI on :8000
│   ├── index.html
│   └── src/
│       ├── App.jsx                    # Root component + state management
│       ├── App.module.css
│       ├── index.css                  # Global CSS variables + resets
│       ├── main.jsx
│       ├── services/
│       │   └── api.js                 # Fetch wrappers for every API endpoint
│       └── components/
│           ├── SearchBar.jsx          # Location search + GPS button
│           ├── WeatherCard.jsx        # Current weather display
│           ├── ForecastGrid.jsx       # 7-day forecast grid
│           ├── SaveQueryModal.jsx     # Date range + notes → POST /api/queries
│           ├── QueryHistory.jsx       # CRUD table with inline edit
│           ├── ExportPanel.jsx        # JSON / CSV / XML / Markdown download
│           └── AboutSection.jsx       # Developer + PM Accelerator info
│
├── data/
│   ├── raw/                           # GlobalWeatherRepository.csv (~35 MB)
│   ├── processed/                     # Cleaned + anomaly-filtered datasets
│   └── cleaned/
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_anomaly_analysis.ipynb
│   ├── 04_time_series_forecasting.ipynb
│   ├── 05_ml_models.ipynb
│   └── 06_advanced_analyses.ipynb            # Ensemble, spatial maps, climate analysis
├── src/
│   ├── preprocessing.py               # Data loading, cleaning, anomaly detection
│   ├── features.py                    # Feature engineering helpers
│   ├── train.py                       # Model training utilities
│   ├── eval.py                        # Regression + forecast metrics
│   └── visualize.py                   # Plotting utilities
├── reports/
│   ├── weather_forecasting_report.md  # Full analysis report
│   ├── forecast_metrics.csv
│   └── temperature_model_metrics.csv
├── api/
│   └── index.py                       # Vercel serverless entry point (Mangum adapter)
├── Dockerfile                         # Multi-stage Docker build (frontend + backend)
├── vercel.json                        # Vercel deployment configuration
├── requirements-vercel.txt            # Lightweight deps for Vercel (no heavy ML stack)
├── run.py                             # Entry point — starts uvicorn
├── requirements.txt
└── weather_app.db                     # SQLite database (auto-created on first run)
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/weather?location=London` | Real-time weather + 7-day forecast |
| `GET` | `/api/weather?lat=51.5&lon=-0.12` | Same, by coordinates |
| `GET` | `/api/geocode?query=New+York` | Validate and resolve a location |
| `POST` | `/api/queries` | Save a weather query to the database |
| `GET` | `/api/queries` | List all saved queries |
| `GET` | `/api/queries/{id}` | Get a single query |
| `PUT` | `/api/queries/{id}` | Update notes / date range |
| `DELETE` | `/api/queries/{id}` | Delete a query |
| `GET` | `/api/export?fmt=json` | Export all queries as JSON |
| `GET` | `/api/export?fmt=csv` | Export as CSV |
| `GET` | `/api/export?fmt=xml` | Export as XML |
| `GET` | `/api/export?fmt=markdown` | Export as Markdown table |

Interactive docs available at **`http://localhost:8000/docs`** (Swagger UI).

---

## Deployment

### Docker

```bash
# Build the image
docker build -t weatherflow .

# Run — database persists in a named volume
docker run -p 8000:8000 -v weatherflow_data:/app/data weatherflow
```

Open **http://localhost:8000**

### Vercel

The project includes a pre-configured [`vercel.json`](vercel.json). Deploy with:

```bash
vercel deploy
```

The Vercel build:
- Compiles the React frontend (`frontend/`) into static assets
- Wraps the FastAPI backend in a Python serverless function via [Mangum](https://mangum.faizanbashir.me/) (`api/index.py`)
- Routes `/api/*` to the serverless function and all other paths to the static React build

> **Note:** Vercel's serverless environment is ephemeral — the SQLite database will not persist between invocations. For production persistence, swap `DATABASE_URL` to a hosted database (PostgreSQL, PlanetScale, etc.).

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/upratham/weather-forecasting-eda-ml.git
cd weather-forecasting-eda-ml
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Running the App

### Option A — Production (single server on port 8000)

```bash
# Build the React frontend first
cd frontend && npm run build && cd ..

# Start the FastAPI backend — serves both API and built UI
python run.py
```

Open **http://localhost:8000**

### Option B — Development (hot reload on both sides)

```bash
# Terminal 1 — FastAPI backend
python run.py

# Terminal 2 — React dev server (proxies /api to :8000)
cd frontend && npm run dev
```

Open **http://localhost:5173**

> The SQLite database (`weather_app.db`) is created automatically in the project root on first startup. No database setup is required.

---

## ML Notebooks

The `notebooks/` folder contains the full data science pipeline used to analyze the Global Weather Repository dataset. Run them in order for the complete pipeline:

```bash
jupyter notebook
```

| Notebook | Description |
|----------|-------------|
| `01_data_cleaning.ipynb` | Missing values, outliers, type normalization |
| `02_eda.ipynb` | Distributions, correlations, temporal trends, air quality |
| `03_anomaly_analysis.ipynb` | Isolation Forest anomaly detection |
| `04_time_series_forecasting.ipynb` | ARIMA vs Prophet comparison |
| `05_ml_models.ipynb` | Linear Regression, Random Forest, Gradient Boosting |
| `06_advanced_analyses.ipynb` | Ensemble models, spatial maps, climate zone analysis |

### Dataset

Download `GlobalWeatherRepository.csv` (~35 MB) from Kaggle and place in `data/raw/`:

```bash
kaggle datasets download -d nelgiriyewithana/global-weather-repository -p data/raw/ --unzip
```

---

## ML Results at a Glance

### Regression Models (test set, anomaly-cleaned data)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 0.018 °C | 0.023 °C | **0.9999** |
| Random Forest | 0.007 °C | 0.193 °C | 0.9996 |
| Gradient Boosting | 0.049 °C | 0.202 °C | 0.9995 |
| Voting Ensemble (LR+RF+GB) | see `reports/ensemble_metrics.csv` | — | — |
| Stacking Ensemble (Ridge meta) | see `reports/ensemble_metrics.csv` | — | — |

### Time-Series Models (daily temperature forecast)

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| ARIMA | 1.202 °C | 1.739 °C | 8.92% |
| Prophet | 4.390 °C | 5.771 °C | 25.99% |

Full findings in **[`reports/weather_forecasting_report.md`](reports/weather_forecasting_report.md)**

---

## Requirements

Key packages (full list in [`requirements.txt`](requirements.txt)):

```
# Backend
fastapi>=0.111.0      uvicorn[standard]>=0.29.0
httpx>=0.27.0         sqlalchemy>=2.0.0
aiofiles>=23.2.0      python-multipart>=0.0.9

# ML / notebooks
pandas>=2.0.0         numpy>=1.24.0
scikit-learn>=1.3.0   statsmodels>=0.14.0
matplotlib>=3.7.0     plotly>=5.15.0
joblib>=1.3.0         jupyter>=1.0.0
```

Frontend: Node.js 18+ with npm.

---

## Demo Video

*A 1–2 minute screen recording walking through the app and key features:*

`[Add your Google Drive / YouTube / Vimeo link here]`

**Recording script outline:**
1. Start the app (`python run.py`) and open `http://localhost:8000`
2. Search a city — show current weather card and 7-day forecast
3. Click **Save Query** — fill in dates and notes, submit
4. Open **History** tab — show the saved entry, edit notes inline, then delete
5. Open **Export** tab — download CSV and show the file
6. Briefly open `/docs` (Swagger UI) to show the live API
7. Optional: show a notebook cell and one ML result chart

---

## Author

**Prathamesh Suhas Uravane**
Submitted for PM Accelerator — AI Engineer Intern Technical Assessment (Full Stack)
