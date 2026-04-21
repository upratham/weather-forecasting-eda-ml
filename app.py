from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

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
APP_THEME = gr.themes.Soft(primary_hue="emerald", neutral_hue="stone")

APP_CSS = """
.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(20, 184, 166, 0.18), transparent 32%),
        radial-gradient(circle at top right, rgba(14, 116, 144, 0.16), transparent 28%),
        linear-gradient(180deg, #f6fbfb 0%, #eef6f4 100%);
}
.hero {
    padding: 1.2rem 1.4rem;
    border: 1px solid rgba(15, 118, 110, 0.16);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.84);
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.06);
}
.hero h1, .panel h3 {
    font-family: Georgia, "Times New Roman", serif;
}
.panel {
    padding: 1rem 1.1rem;
    border: 1px solid rgba(15, 118, 110, 0.14);
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.86);
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
}
"""

_APP_ARTIFACTS: dict[str, Any] | None = None


def resolve_training_source() -> Path:
    """Return the best available processed dataset for the app model."""

    preferred = PROCESSED_DIR / "weather_without_anomalies.csv"
    fallback = PROCESSED_DIR / "clean_weather_data.csv"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        "No processed dataset found for app training. Expected "
        "`data/processed/weather_without_anomalies.csv` or `data/processed/clean_weather_data.csv`."
    )


def _round_number(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def build_input_ranges(df: pd.DataFrame) -> dict[str, dict[str, float | int | str]]:
    """Summarize app-friendly defaults and min/max ranges from the dataset."""

    input_ranges: dict[str, dict[str, float | int | str]] = {}
    for column in APP_WEATHER_INPUT_COLUMNS:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        input_ranges[column] = {
            "min": _round_number(series.min()),
            "max": _round_number(series.max()),
            "default": _round_number(series.median()),
        }

    timestamps = pd.to_datetime(df["last_updated"], errors="coerce").dropna()
    if not timestamps.empty:
        hours = timestamps.dt.hour
        input_ranges["date"] = {
            "min": str(timestamps.min().date()),
            "max": str(timestamps.max().date()),
            "default": str(timestamps.max().date()),
        }
        input_ranges["hour"] = {
            "min": 0,
            "max": 23,
            "default": int(hours.median()),
        }

    return input_ranges


def default_output_message() -> str:
    """Return the default helper text shown before a prediction is made."""

    return "Run a prediction to see the cleaned inputs, model metrics, and final estimate."


def train_app_model() -> dict[str, Any]:
    """Train, persist, and return the Gradio app model plus metadata."""

    source_path = resolve_training_source()
    df = load_processed_data(source_path)
    X, y = prepare_app_model_frame(df, target_column="temperature_celsius")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    metrics = evaluate_regression(y_test, model.predict(X_test))
    metadata = {
        "model_name": type(model).__name__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "source_path": str(source_path),
        "feature_columns": APP_MODEL_FEATURE_COLUMNS,
        "metrics": {name: _round_number(value, 4) for name, value in metrics.items()},
        "row_count": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "input_ranges": build_input_ranges(df),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, APP_MODEL_PATH)
    APP_META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"model": model, "meta": metadata}


def load_app_artifacts(force_retrain: bool = False) -> dict[str, Any]:
    """Load the app model if available; otherwise train it once."""

    global _APP_ARTIFACTS
    if _APP_ARTIFACTS is not None and not force_retrain:
        return _APP_ARTIFACTS

    if not force_retrain and APP_MODEL_PATH.exists() and APP_META_PATH.exists():
        _APP_ARTIFACTS = {
            "model": joblib.load(APP_MODEL_PATH),
            "meta": json.loads(APP_META_PATH.read_text(encoding="utf-8")),
        }
        return _APP_ARTIFACTS

    _APP_ARTIFACTS = train_app_model()
    return _APP_ARTIFACTS


def _as_float(value: Any, label: str) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise gr.Error(f"{label} is required.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise gr.Error(f"{label} must be numeric.") from exc


def _clamp(value: float, *, lower: float | None = None, upper: float | None = None) -> float:
    if lower is not None:
        value = max(lower, value)
    if upper is not None:
        value = min(upper, value)
    return value


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
    """Validate user input and return normalized values plus adjustment notes."""

    notes: list[str] = []
    normalized: dict[str, float | int | str] = {}

    latitude_value = _as_float(latitude, "Latitude")
    longitude_value = _as_float(longitude, "Longitude")
    if not -90 <= latitude_value <= 90:
        raise gr.Error("Latitude must be between -90 and 90.")
    if not -180 <= longitude_value <= 180:
        raise gr.Error("Longitude must be between -180 and 180.")
    normalized["latitude"] = latitude_value
    normalized["longitude"] = longitude_value

    try:
        timestamp = pd.Timestamp(date_value)
    except (TypeError, ValueError) as exc:
        raise gr.Error("Date must be in YYYY-MM-DD format.") from exc
    if pd.isna(timestamp):
        raise gr.Error("Date must be in YYYY-MM-DD format.")
    normalized["date"] = str(timestamp.date())

    hour_value = _as_float(hour, "Hour")
    rounded_hour = int(round(hour_value))
    clamped_hour = int(_clamp(rounded_hour, lower=0, upper=23))
    if rounded_hour != hour_value:
        notes.append(f"Hour rounded from {hour_value} to {rounded_hour}.")
    if clamped_hour != rounded_hour:
        notes.append(f"Hour clamped to {clamped_hour} to stay within 0-23.")
    normalized["hour"] = clamped_hour

    pressure_value = _as_float(pressure_mb, "Pressure (mb)")
    if pressure_value <= 0:
        raise gr.Error("Pressure must be greater than 0 mb.")
    normalized["pressure_mb"] = pressure_value

    humidity_value = _as_float(humidity, "Humidity")
    clamped_humidity = _clamp(humidity_value, lower=0, upper=100)
    if clamped_humidity != humidity_value:
        notes.append(f"Humidity clamped to {clamped_humidity}%.")
    normalized["humidity"] = clamped_humidity

    cloud_value = _as_float(cloud, "Cloud cover")
    clamped_cloud = _clamp(cloud_value, lower=0, upper=100)
    if clamped_cloud != cloud_value:
        notes.append(f"Cloud cover clamped to {clamped_cloud}%.")
    normalized["cloud"] = clamped_cloud

    wind_value = _clamp(_as_float(wind_kph, "Wind speed"), lower=0)
    gust_value = _clamp(_as_float(gust_kph, "Gust speed"), lower=0)
    precip_value = _clamp(_as_float(precip_mm, "Precipitation"), lower=0)
    visibility_value = _clamp(_as_float(visibility_km, "Visibility"), lower=0)
    uv_value = _clamp(_as_float(uv_index, "UV index"), lower=0)

    if wind_value != float(wind_kph):
        notes.append("Wind speed clamped to a non-negative value.")
    if gust_value != float(gust_kph):
        notes.append("Gust speed clamped to a non-negative value.")
    if precip_value != float(precip_mm):
        notes.append("Precipitation clamped to a non-negative value.")
    if visibility_value != float(visibility_km):
        notes.append("Visibility clamped to a non-negative value.")
    if uv_value != float(uv_index):
        notes.append("UV index clamped to a non-negative value.")

    normalized["wind_kph"] = wind_value
    normalized["gust_kph"] = gust_value
    normalized["precip_mm"] = precip_value
    normalized["visibility_km"] = visibility_value
    normalized["uv_index"] = uv_value
    return normalized, notes


def format_model_summary(meta: dict[str, Any]) -> str:
    """Render the static model summary shown in the app."""

    metrics = meta["metrics"]
    source_name = Path(meta["source_path"]).name
    feature_count = len(meta["feature_columns"])
    return (
        "### Model Snapshot\n"
        f"- **Model**: {meta['model_name']}\n"
        f"- **Training data**: `{source_name}`\n"
        f"- **Rows used**: {meta['row_count']:,}\n"
        f"- **Feature count**: {feature_count}\n"
        f"- **Held-out MAE**: {metrics['mae']}\n"
        f"- **Held-out RMSE**: {metrics['rmse']}\n"
        f"- **Held-out R2**: {metrics['r2']}\n"
        f"- **Trained at**: {meta['trained_at']}"
    )


def format_prediction_details(
    *,
    meta: dict[str, Any],
    normalized: dict[str, float | int | str],
    notes: list[str],
    celsius: float,
    fahrenheit: float,
) -> str:
    """Render the prediction explanation block."""

    metrics = meta["metrics"]
    input_lines = [
        f"- **Latitude / Longitude**: {normalized['latitude']:.2f}, {normalized['longitude']:.2f}",
        f"- **Date / Hour**: {normalized['date']} at {normalized['hour']:02d}:00",
        f"- **Pressure**: {normalized['pressure_mb']:.1f} mb",
        f"- **Humidity / Cloud**: {normalized['humidity']:.1f}% / {normalized['cloud']:.1f}%",
        f"- **Wind / Gust**: {normalized['wind_kph']:.1f} / {normalized['gust_kph']:.1f} kph",
        f"- **Precipitation**: {normalized['precip_mm']:.2f} mm",
        f"- **Visibility / UV**: {normalized['visibility_km']:.1f} km / {normalized['uv_index']:.1f}",
    ]
    note_lines = notes or ["No input adjustments were needed."]
    note_markdown = "\n".join(f"- {note}" for note in note_lines)

    return (
        "### Prediction Details\n"
        f"- **Predicted temperature**: {celsius:.2f} deg C / {fahrenheit:.2f} deg F\n"
        f"- **Model**: {meta['model_name']}\n"
        f"- **Held-out MAE**: {metrics['mae']}\n"
        f"- **Held-out RMSE**: {metrics['rmse']}\n\n"
        "### Input Snapshot\n"
        f"{chr(10).join(input_lines)}\n\n"
        "### Input Adjustments\n"
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
) -> tuple[float, float, str]:
    """Predict temperature from a single weather snapshot."""

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
    details = format_prediction_details(
        meta=artifacts["meta"],
        normalized=normalized,
        notes=notes,
        celsius=celsius,
        fahrenheit=fahrenheit,
    )
    return round(celsius, 2), round(fahrenheit, 2), details


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
        None,
        None,
        default_output_message(),
    )


def build_interface() -> gr.Blocks:
    """Construct the Gradio app."""

    artifacts = load_app_artifacts()
    meta = artifacts["meta"]
    ranges = meta["input_ranges"]

    with gr.Blocks(title="Weather Temperature Predictor") as demo:
        gr.Markdown(
            """
<div class="hero">
  <h1>Weather Temperature Predictor</h1>
  <p>Enter a realistic weather snapshot and get an app-specific temperature prediction from the clean Gradio model.</p>
</div>
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown('<div class="panel"><h3>Manual Input</h3></div>')
                latitude = gr.Number(label="Latitude", value=ranges["latitude"]["default"])
                longitude = gr.Number(label="Longitude", value=ranges["longitude"]["default"])
                date_value = gr.Textbox(
                    label="Date (YYYY-MM-DD)",
                    value=ranges["date"]["default"],
                )
                hour = gr.Number(label="Hour (0-23)", value=ranges["hour"]["default"], precision=0)
                pressure_mb = gr.Number(label="Pressure (mb)", value=ranges["pressure_mb"]["default"])
                humidity = gr.Number(label="Humidity (%)", value=ranges["humidity"]["default"])
                cloud = gr.Number(label="Cloud Cover (%)", value=ranges["cloud"]["default"])
                wind_kph = gr.Number(label="Wind Speed (kph)", value=ranges["wind_kph"]["default"])
                gust_kph = gr.Number(label="Gust Speed (kph)", value=ranges["gust_kph"]["default"])
                precip_mm = gr.Number(label="Precipitation (mm)", value=ranges["precip_mm"]["default"])
                visibility_km = gr.Number(
                    label="Visibility (km)",
                    value=ranges["visibility_km"]["default"],
                )
                uv_index = gr.Number(label="UV Index", value=ranges["uv_index"]["default"])

                with gr.Row():
                    predict_button = gr.Button("Predict Temperature", variant="primary")
                    reset_button = gr.Button("Reset Form")

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
                )

            with gr.Column(scale=1):
                gr.Markdown('<div class="panel"><h3>Prediction Output</h3></div>')
                celsius_output = gr.Number(label="Predicted Temperature (deg C)", precision=2)
                fahrenheit_output = gr.Number(label="Predicted Temperature (deg F)", precision=2)
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
        outputs = [celsius_output, fahrenheit_output, details_output]

        predict_button.click(fn=predict_temperature, inputs=inputs, outputs=outputs)
        reset_button.click(
            fn=lambda: build_reset_values(meta),
            inputs=[],
            outputs=inputs + outputs,
        )

    return demo


def main() -> None:
    """Launch the Gradio application."""

    demo = build_interface()
    demo.launch(theme=APP_THEME, css=APP_CSS)


if __name__ == "__main__":
    main()
