"""Deploy the Gradio weather-temperature predictor to Modal.

Usage
-----
# First-time setup — add your Modal token to .env in the project root:
    MODAL_TOKEN_ID=<id>
    MODAL_TOKEN_SECRET=<secret>

# Serve locally (hot-reload, useful for testing):
    modal serve src/modal_deploy.py

# Deploy to Modal cloud (persistent URL):
    modal deploy src/modal_deploy.py
"""
from __future__ import annotations

from pathlib import Path

# Load .env from the project root so MODAL_TOKEN_ID / MODAL_TOKEN_SECRET are set
# before the Modal SDK initialises.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")
except ModuleNotFoundError:
    pass  # python-dotenv not installed; rely on env vars being set externally

import modal

ROOT = Path(__file__).parent.parent  # project root

APP_NAME = "weather-temperature-predictor"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "gradio>=4.0.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "joblib>=1.3.0",
    )
    .add_local_python_source("src")
    .add_local_file(ROOT / "app.py", "/root/app.py")
    .add_local_dir(ROOT / "data" / "processed", "/root/data/processed")
)

# Persists the trained model across deployments so it isn't retrained on every cold start.
model_vol = modal.Volume.from_name("weather-model-vol", create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    volumes={"/root/models": model_vol},
    timeout=120,
)
@modal.concurrent(max_inputs=10)
def predict(
    latitude: float,
    longitude: float,
    date_value: str,
    hour: int,
    pressure_mb: float,
    humidity: float,
    cloud: float,
    wind_kph: float,
    gust_kph: float,
    precip_mm: float,
    visibility_km: float,
    uv_index: float,
) -> tuple[float, float, str]:
    """Directly callable prediction function.

    Called by app.py via ``modal.Function.lookup()`` when running locally,
    so the model always runs from the Modal Volume rather than the local filesystem.
    """
    from app import predict_temperature  # /root/app.py is in the image

    return predict_temperature(
        latitude, longitude, date_value, hour,
        pressure_mb, humidity, cloud,
        wind_kph, gust_kph, precip_mm,
        visibility_km, uv_index,
    )


@app.function(
    image=image,
    volumes={"/root/models": model_vol},
    timeout=600,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def serve():
    from app import build_interface  # resolves to /root/app.py; PROJECT_ROOT = /root

    demo = build_interface()
    return demo.app
