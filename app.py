from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

# Load .env so Modal credentials are available when running locally.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ModuleNotFoundError:
    pass

logger = logging.getLogger(__name__)


def _is_on_modal() -> bool:
    """True when executing inside a Modal container."""
    return bool(os.environ.get("MODAL_TASK_ID"))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing.
    raise SystemExit(
        "Gradio is not installed in the active environment. "
        "Install dependencies first, then run `python app.py` again."
    ) from exc

from src.eval import evaluate_regression
from src.features import (
    APP_MODEL_FEATURE_COLUMNS,
    APP_WEATHER_INPUT_COLUMNS,
    build_app_prediction_frame,
    prepare_app_model_frame,
)
from src.preprocessing import load_processed_data

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
APP_MODEL_PATH = MODELS_DIR / "gradio_temperature_model.joblib"
APP_META_PATH = MODELS_DIR / "gradio_temperature_model_meta.json"

# Enhanced theme with more vibrant colors
APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    body_background_fill_dark="linear-gradient(135deg, #1e3a8a 0%, #312e81 100%)",
)

# Enhanced CSS with modern design and IMPROVED CONTRAST
APP_CSS = """
/* Global container styling */
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    min-height: 100vh;
}

/* Hero section with glassmorphism - IMPROVED CONTRAST */
.hero {
    padding: 2rem 2.5rem;
    margin: 1rem 0 2rem 0;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-radius: 24px;
    background: rgba(255, 255, 255, 0.25);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2), 
                0 0 0 1px rgba(255, 255, 255, 0.1) inset;
    animation: fadeInUp 0.6s ease-out;
}

.hero h1 {
    font-family: 'Inter', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    color: #1e293b;
    margin: 0 0 0.5rem 0;
    text-shadow: 0 2px 4px rgba(255, 255, 255, 0.5);
}

.hero p {
    color: #334155;
    font-size: 1.1rem;
    font-weight: 500;
    margin: 0;
    line-height: 1.6;
}

/* Panel styling with glassmorphism - IMPROVED CONTRAST */
.panel {
    padding: 1.5rem 1.8rem;
    margin: 0.5rem 0;
    border: 2px solid rgba(255, 255, 255, 0.25);
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15),
                0 0 0 1px rgba(255, 255, 255, 0.08) inset;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.panel:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2),
                0 0 0 1px rgba(255, 255, 255, 0.12) inset;
}

.panel h3 {
    font-family: 'Inter', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #1e293b;
    margin: 0 0 1rem 0;
    text-shadow: 0 1px 2px rgba(255, 255, 255, 0.3);
}

/* Input field styling */
input[type="number"], 
input[type="text"], 
.gr-text-input,
.gr-number-input {
    background: rgba(255, 255, 255, 0.95) !important;
    border: 2px solid rgba(255, 255, 255, 0.5) !important;
    border-radius: 12px !important;
    color: #1e293b !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

input[type="number"]:focus, 
input[type="text"]:focus {
    background: rgba(255, 255, 255, 1) !important;
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 4px rgba(96, 165, 250, 0.2) !important;
    transform: scale(1.01);
}

/* Button styling */
.gr-button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    border: none !important;
    color: white !important;
}

.gr-button-primary:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important;
}

.gr-button-secondary {
    background: rgba(255, 255, 255, 0.3) !important;
    border: 2px solid rgba(255, 255, 255, 0.5) !important;
    color: #1e293b !important;
    font-weight: 700 !important;
}

.gr-button-secondary:hover {
    background: rgba(255, 255, 255, 0.4) !important;
    transform: translateY(-2px);
}

/* Output styling - IMPROVED CONTRAST */
.gr-markdown {
    background: rgba(255, 255, 255, 0.2) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    color: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
}

.gr-markdown h3 {
    color: #0f172a !important;
    font-weight: 700 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.75rem !important;
}

.gr-markdown strong {
    color: #1d4ed8 !important;
    font-weight: 700 !important;
}

/* Label styling - IMPROVED CONTRAST */
label {
    color: #1e293b !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    text-shadow: 0 1px 2px rgba(255, 255, 255, 0.3);
}

/* Examples section - IMPROVED CONTRAST */
.gr-examples {
    background: rgba(255, 255, 255, 0.15) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    margin-top: 1.5rem !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
}

/* Animation keyframes */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.8;
    }
}

/* Column styling */
.gr-column {
    background: transparent !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.3);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.5);
}

/* Temperature display styling */
.temp-display {
    font-size: 2.5rem;
    font-weight: 800;
    color: #1e293b;
    text-align: center;
    padding: 1.5rem;
    margin: 1rem 0;
    background: rgba(255, 255, 255, 0.25);
    border-radius: 16px;
    border: 2px solid rgba(255, 255, 255, 0.4);
    text-shadow: 0 2px 4px rgba(255, 255, 255, 0.5);
}

/* Group styling */
.gr-group {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    margin: 0.5rem 0 !important;
}

/* Markdown text inside groups - IMPROVED CONTRAST */
.gr-group .gr-markdown {
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 0 !important;
    color: #1e293b !important;
    font-weight: 600 !important;
}
"""


def default_output_message() -> str:
    """Default placeholder message for the prediction output."""

    return (
        "### 🌟 Ready to predict!\n\n"
        "Fill in the weather details on the left and click **Predict Temperature** "
        "to see the AI-powered forecast.\n\n"
        "💡 Try the example presets below for a quick start!"
    )


def format_model_summary(meta: dict[str, Any]) -> str:
    """Render the static model information panel."""

    metrics = meta["metrics"]
    row_count = meta.get("row_count", meta.get("training_samples", 0))
    return (
        "### 📈 Model Performance\n\n"
        f"**Training Samples**: {row_count:,}  \n"
        f"**Test MAE**: {metrics['mae']:.4f}°C  \n"
        f"**Test RMSE**: {metrics['rmse']:.4f}°C  \n"
        f"**Test R²**: {metrics['r2']:.4f}  \n\n"
        f"*Model trained on {row_count:,} weather snapshots*"
    )


_APP_ARTIFACTS: dict[str, Any] | None = None


def _resolve_training_source() -> Path:
    for candidate in (
        PROCESSED_DIR / "weather_without_anomalies.csv",
        PROCESSED_DIR / "clean_weather_data.csv",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No processed dataset found. Expected weather_without_anomalies.csv or "
        "clean_weather_data.csv under data/processed/."
    )


def _train_and_save() -> dict[str, Any]:
    from src.features import APP_MODEL_FEATURE_COLUMNS, APP_WEATHER_INPUT_COLUMNS, prepare_app_model_frame
    from src.preprocessing import load_processed_data

    source = _resolve_training_source()
    logger.info("Training app model from %s", source)
    df = load_processed_data(source).copy()
    for col, lo, hi in [("wind_kph", 0, 200), ("gust_kph", 0, 250), ("pressure_mb", 870, 1100)]:
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    X, y = prepare_app_model_frame(df, target_column="temperature_celsius")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    from src.eval import evaluate_regression
    metrics = evaluate_regression(y_test, model.predict(X_test))

    # Build input_ranges for UI defaults
    input_ranges: dict[str, Any] = {}
    for col in APP_WEATHER_INPUT_COLUMNS:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if not s.empty:
            input_ranges[col] = {"min": round(float(s.min()), 3), "max": round(float(s.max()), 3), "default": round(float(s.median()), 3)}
    ts = pd.to_datetime(df["last_updated"], errors="coerce").dropna()
    if not ts.empty:
        input_ranges["date"] = {"min": str(ts.min().date()), "max": str(ts.max().date()), "default": str(ts.max().date())}
        input_ranges["hour"] = {"min": 0, "max": 23, "default": int(ts.dt.hour.median())}

    meta = {
        "model_name": type(model).__name__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "feature_columns": APP_MODEL_FEATURE_COLUMNS,
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "row_count": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "input_ranges": input_ranges,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, APP_MODEL_PATH)
    APP_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(
        "Model saved — MAE: %.4f °C  RMSE: %.4f °C  R²: %.4f  rows: %d",
        meta["metrics"]["mae"], meta["metrics"]["rmse"], meta["metrics"]["r2"], meta["row_count"],
    )
    return {"model": model, "meta": meta}


def load_app_artifacts(force_retrain: bool = False) -> dict[str, Any]:
    """Return cached artifacts, loading from disk or training if necessary."""

    global _APP_ARTIFACTS
    if _APP_ARTIFACTS is not None and not force_retrain:
        return _APP_ARTIFACTS

    if not force_retrain and APP_MODEL_PATH.exists() and APP_META_PATH.exists():
        logger.info("Loading model from %s", APP_MODEL_PATH)
        model = joblib.load(APP_MODEL_PATH)
        with open(APP_META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        _APP_ARTIFACTS = {"model": model, "meta": meta}
        logger.info("Model loaded (trained at %s)", meta.get("trained_at", "unknown"))
    else:
        _APP_ARTIFACTS = _train_and_save()

    return _APP_ARTIFACTS


def normalize_prediction_inputs(
    latitude: Any,
    longitude: Any,
    date_value: Any,
    hour: Any,
    pressure_mb: Any,
    humidity: Any,
    cloud: Any,
    wind_kph: Any,
    gust_kph: Any,
    precip_mm: Any,
    visibility_km: Any,
    uv_index: Any,
) -> tuple[dict[str, float | int | str], list[str]]:
    """Validate and normalize user input for prediction."""

    notes: list[str] = []

    # Convert and validate latitude
    try:
        lat = float(latitude)
        if not -90.0 <= lat <= 90.0:
            lat = max(-90.0, min(90.0, lat))
            notes.append(f"⚠️ Latitude clamped to valid range: {lat:.2f}")
    except (TypeError, ValueError):
        lat = 0.0
        notes.append("⚠️ Invalid latitude; defaulted to 0.0")

    # Convert and validate longitude
    try:
        lon = float(longitude)
        if not -180.0 <= lon <= 180.0:
            lon = ((lon + 180.0) % 360.0) - 180.0
            notes.append(f"⚠️ Longitude wrapped to valid range: {lon:.2f}")
    except (TypeError, ValueError):
        lon = 0.0
        notes.append("⚠️ Invalid longitude; defaulted to 0.0")

    # Validate date
    try:
        date_obj = datetime.strptime(str(date_value).strip(), "%Y-%m-%d")
        date_str = date_obj.strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        date_str = "2026-04-24"
        notes.append(f"⚠️ Invalid date format; defaulted to {date_str}")

    # Validate hour
    try:
        hr = int(hour)
        if not 0 <= hr <= 23:
            hr = max(0, min(23, hr))
            notes.append(f"⚠️ Hour clamped to 0-23: {hr}")
    except (TypeError, ValueError):
        hr = 12
        notes.append("⚠️ Invalid hour; defaulted to 12")

    # Helper function to validate numeric fields
    def validate_numeric(value: Any, name: str, default: float, min_val: float = 0.0) -> float:
        try:
            num = float(value)
            if num < min_val:
                notes.append(f"⚠️ {name} must be >= {min_val}; clamped to {min_val}")
                return min_val
            return num
        except (TypeError, ValueError):
            notes.append(f"⚠️ Invalid {name}; defaulted to {default}")
            return default

    pressure = validate_numeric(pressure_mb, "Pressure", 1013.0, 800.0)
    humid = validate_numeric(humidity, "Humidity", 50.0, 0.0)
    cld = validate_numeric(cloud, "Cloud", 0.0, 0.0)
    wind = validate_numeric(wind_kph, "Wind speed", 0.0, 0.0)
    gust = validate_numeric(gust_kph, "Gust speed", 0.0, 0.0)
    precip = validate_numeric(precip_mm, "Precipitation", 0.0, 0.0)
    vis = validate_numeric(visibility_km, "Visibility", 10.0, 0.0)
    uv = validate_numeric(uv_index, "UV index", 0.0, 0.0)

    # Clamp humidity and cloud to 100%
    if humid > 100.0:
        humid = 100.0
        notes.append("⚠️ Humidity clamped to 100%")
    if cld > 100.0:
        cld = 100.0
        notes.append("⚠️ Cloud cover clamped to 100%")

    return (
        {
            "latitude": lat,
            "longitude": lon,
            "date": date_str,
            "hour": hr,
            "pressure_mb": pressure,
            "humidity": humid,
            "cloud": cld,
            "wind_kph": wind,
            "gust_kph": gust,
            "precip_mm": precip,
            "visibility_km": vis,
            "uv_index": uv,
        },
        notes,
    )


def format_prediction_details(
    meta: dict[str, Any],
    normalized: dict[str, float | int | str],
    notes: list[str],
    celsius: float,
    fahrenheit: float,
) -> str:
    """Render the prediction explanation block."""

    metrics = meta["metrics"]
    
    # Temperature emoji based on celsius value
    temp_emoji = "🥶" if celsius < 0 else "❄️" if celsius < 10 else "🌤️" if celsius < 20 else "☀️" if celsius < 30 else "🔥"
    
    input_lines = [
        f"📍 **Latitude / Longitude**: {normalized['latitude']:.2f}, {normalized['longitude']:.2f}",
        f"📅 **Date / Hour**: {normalized['date']} at {normalized['hour']:02d}:00",
        f"🌡️ **Pressure**: {normalized['pressure_mb']:.1f} mb",
        f"💧 **Humidity / Cloud**: {normalized['humidity']:.1f}% / {normalized['cloud']:.1f}%",
        f"💨 **Wind / Gust**: {normalized['wind_kph']:.1f} / {normalized['gust_kph']:.1f} kph",
        f"🌧️ **Precipitation**: {normalized['precip_mm']:.2f} mm",
        f"👁️ **Visibility / UV**: {normalized['visibility_km']:.1f} km / {normalized['uv_index']:.1f}",
    ]
    note_lines = notes or ["✅ No input adjustments were needed."]
    note_markdown = "\n".join(f"- {note}" for note in note_lines)

    return (
        f"### {temp_emoji} Prediction Results\n"
        f"<div class='temp-display'>{celsius:.2f}°C / {fahrenheit:.2f}°F</div>\n\n"
        f"**Model**: {meta['model_name']}  \n"
        f"**Held-out MAE**: {metrics['mae']}°C  \n"
        f"**Held-out RMSE**: {metrics['rmse']}°C\n\n"
        "### 📝 Input Snapshot\n"
        f"{chr(10).join(input_lines)}\n\n"
        "### ⚙️ Input Adjustments\n"
        f"{note_markdown}"
    )


def predict_temperature(
    latitude: Any,
    longitude: Any,
    date_value: Any,
    hour: Any,
    pressure_mb: Any,
    humidity: Any,
    cloud: Any,
    wind_kph: Any,
    gust_kph: Any,
    precip_mm: Any,
    visibility_km: Any,
    uv_index: Any,
) -> str:
    """Predict temperature from a single weather snapshot.

    When running locally, delegates to the deployed Modal ``predict`` function
    so the model is always served from the Modal Volume.  Falls back to a local
    model if Modal is unavailable or we are already inside a Modal container.
    """
    if not _is_on_modal():
        try:
            import modal
            predict_fn = modal.Function.lookup("weather-temperature-predictor", "predict")
            logger.info("Routing prediction to Modal")
            return predict_fn.remote(
                float(latitude), float(longitude), str(date_value), int(hour),
                float(pressure_mb), float(humidity), float(cloud),
                float(wind_kph), float(gust_kph), float(precip_mm),
                float(visibility_km), float(uv_index),
            )
        except Exception as exc:
            logger.warning("Modal lookup failed (%s) — falling back to local model", exc)

    artifacts = load_app_artifacts()
    normalized, notes = normalize_prediction_inputs(
        latitude,
        longitude,
        date_value,
        hour,
        pressure_mb,
        humidity,
        cloud,
        wind_kph,
        gust_kph,
        precip_mm,
        visibility_km,
        uv_index,
    )
    frame = build_app_prediction_frame(
        latitude=normalized["latitude"],
        longitude=normalized["longitude"],
        date_value=normalized["date"],
        hour=int(normalized["hour"]),
        pressure_mb=normalized["pressure_mb"],
        humidity=normalized["humidity"],
        cloud=normalized["cloud"],
        wind_kph=normalized["wind_kph"],
        gust_kph=normalized["gust_kph"],
        precip_mm=normalized["precip_mm"],
        visibility_km=normalized["visibility_km"],
        uv_index=normalized["uv_index"],
    )

    celsius = float(artifacts["model"].predict(frame)[0])
    fahrenheit = (celsius * 9.0 / 5.0) + 32.0
    logger.debug("Prediction: %.2f °C / %.2f °F", celsius, fahrenheit)
    details = format_prediction_details(
        meta=artifacts["meta"],
        normalized=normalized,
        notes=notes,
        celsius=celsius,
        fahrenheit=fahrenheit,
    )
    return details


def build_example_rows(meta: dict[str, Any]) -> list[list[Any]]:
    """Create a few example presets for the Gradio UI."""

    default_date = meta["input_ranges"].get("date", {}).get("default", "2026-04-24")
    return [
        [12.97, 77.59, default_date, 9, 1012.0, 68, 35, 14.0, 22.0, 0.0, 10.0, 5.5],
        [40.71, -74.01, default_date, 14, 1008.0, 57, 45, 18.0, 28.0, 0.8, 12.0, 6.7],
        [-33.87, 151.21, default_date, 19, 1016.0, 74, 60, 21.0, 32.0, 1.2, 9.5, 3.8],
    ]


def build_reset_values(meta: dict[str, Any]) -> tuple[Any, ...]:
    """Return the default input/output state for the reset button."""

    ranges = meta["input_ranges"]
    return (
        ranges["latitude"]["default"],
        ranges["longitude"]["default"],
        ranges["date"]["default"],
        ranges["hour"]["default"],
        ranges["pressure_mb"]["default"],
        ranges["humidity"]["default"],
        ranges["cloud"]["default"],
        ranges["wind_kph"]["default"],
        ranges["gust_kph"]["default"],
        ranges["precip_mm"]["default"],
        ranges["visibility_km"]["default"],
        ranges["uv_index"]["default"],
        default_output_message(),
    )


def build_interface() -> gr.Blocks:
    """Construct the Gradio app."""

    artifacts = load_app_artifacts()
    meta = artifacts["meta"]
    ranges = meta["input_ranges"]

    with gr.Blocks(title="🌡️ Weather Temperature Predictor", theme=APP_THEME, css=APP_CSS) as demo:
        gr.Markdown(
            """
<div class="hero">
  <h1>🌡️ Weather Temperature Predictor</h1>
  <p>Enter a realistic weather snapshot and get an AI-powered temperature prediction using our trained RandomForest model</p>
</div>
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown('<div class="panel"><h3>🌍 Manual Input</h3></div>')
                
                with gr.Group():
                    gr.Markdown("**📍 Location**")
                    latitude = gr.Number(label="Latitude", value=ranges["latitude"]["default"])
                    longitude = gr.Number(label="Longitude", value=ranges["longitude"]["default"])
                
                with gr.Group():
                    gr.Markdown("**📅 Date & Time**")
                    date_value = gr.Textbox(
                        label="Date (YYYY-MM-DD)",
                        value=ranges["date"]["default"],
                    )
                    hour = gr.Number(label="Hour (0-23)", value=ranges["hour"]["default"], precision=0)
                
                with gr.Group():
                    gr.Markdown("**🌤️ Weather Conditions**")
                    pressure_mb = gr.Number(label="Pressure (mb)", value=ranges["pressure_mb"]["default"])
                    humidity = gr.Number(label="Humidity (%)", value=ranges["humidity"]["default"])
                    cloud = gr.Number(label="Cloud Cover (%)", value=ranges["cloud"]["default"])
                
                with gr.Group():
                    gr.Markdown("**💨 Wind & Precipitation**")
                    wind_kph = gr.Number(label="Wind Speed (kph)", value=ranges["wind_kph"]["default"])
                    gust_kph = gr.Number(label="Gust Speed (kph)", value=ranges["gust_kph"]["default"])
                    precip_mm = gr.Number(label="Precipitation (mm)", value=ranges["precip_mm"]["default"])
                
                with gr.Group():
                    gr.Markdown("**👁️ Visibility & UV**")
                    visibility_km = gr.Number(
                        label="Visibility (km)",
                        value=ranges["visibility_km"]["default"],
                    )
                    uv_index = gr.Number(label="UV Index", value=ranges["uv_index"]["default"])

                with gr.Row():
                    predict_button = gr.Button("🔮 Predict Temperature", variant="primary", size="lg")
                    reset_button = gr.Button("🔄 Reset Form", size="lg")

                gr.Examples(
                    examples=build_example_rows(meta),
                    inputs=[
                        latitude,
                        longitude,
                        date_value,
                        hour,
                        pressure_mb,
                        humidity,
                        cloud,
                        wind_kph,
                        gust_kph,
                        precip_mm,
                        visibility_km,
                        uv_index,
                    ],
                    cache_examples=False,
                    label="📋 Quick Examples",
                )

            with gr.Column(scale=1):
                gr.Markdown('<div class="panel"><h3>📊 Prediction Output</h3></div>')
                details_output = gr.Markdown(default_output_message())
                gr.Markdown(format_model_summary(meta))

        inputs = [
            latitude,
            longitude,
            date_value,
            hour,
            pressure_mb,
            humidity,
            cloud,
            wind_kph,
            gust_kph,
            precip_mm,
            visibility_km,
            uv_index,
        ]
        outputs = [details_output]

        predict_button.click(fn=predict_temperature, inputs=inputs, outputs=outputs)
        reset_button.click(
            fn=lambda: build_reset_values(meta),
            inputs=[],
            outputs=inputs + outputs,
        )

    return demo


def main() -> None:
    """Launch the Gradio application."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    demo = build_interface()
    demo.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()