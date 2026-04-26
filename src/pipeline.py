"""Train → evaluate → deploy pipeline for the weather temperature predictor.

Typical usage
-------------
From a notebook or script:

    from src.pipeline import train_eval_deploy, predict_remote, get_endpoint_url

    # Full pipeline: train, evaluate, deploy to Modal, cache the URL
    result = train_eval_deploy()
    print(result["endpoint_url"])

    # Later — call the live Modal endpoint for a prediction
    celsius, fahrenheit, details = predict_remote(
        get_endpoint_url(),
        latitude=12.97, longitude=77.59,
        date_value="2026-04-25", hour=9,
        pressure_mb=1012.0, humidity=68.0, cloud=35.0,
        wind_kph=14.0, gust_kph=22.0, precip_mm=0.0,
        visibility_km=10.0, uv_index=5.5,
    )
"""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
ENDPOINT_URL_PATH = ROOT / "models" / "endpoint_url.txt"


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------

def deploy_to_modal(deploy_file: str = "src/modal_deploy.py") -> str:
    """Run ``modal deploy`` and return the served ASGI endpoint URL.

    The URL is also written to ``models/endpoint_url.txt`` so subsequent
    calls to :func:`get_endpoint_url` can retrieve it without redeploying.
    """
    logger.info("Running: modal deploy %s", deploy_file)
    result = subprocess.run(
        ["modal", "deploy", deploy_file],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        logger.error("modal deploy failed (exit %d):\n%s", result.returncode, output)
        raise RuntimeError(f"modal deploy failed:\n{output}")

    # Modal prints the ASGI URL like:
    #   https://<user>--weather-temperature-predictor-serve.modal.run
    match = re.search(r"https://\S+\.modal\.run", output)
    if not match:
        logger.error("Could not parse endpoint URL from modal output:\n%s", output)
        raise RuntimeError(
            f"Could not find endpoint URL in modal output:\n{output}"
        )
    url = match.group(0).rstrip("/")

    ENDPOINT_URL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENDPOINT_URL_PATH.write_text(url, encoding="utf-8")
    logger.info("Deployed. Endpoint: %s", url)
    return url


def get_endpoint_url() -> str | None:
    """Return the cached Modal endpoint URL, or ``None`` if not yet deployed."""
    if ENDPOINT_URL_PATH.exists():
        text = ENDPOINT_URL_PATH.read_text(encoding="utf-8").strip()
        return text or None
    return None


# ---------------------------------------------------------------------------
# Remote prediction
# ---------------------------------------------------------------------------

def predict_remote(
    endpoint_url: str,
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
    """Call the deployed Gradio endpoint and return ``(celsius, fahrenheit, details)``.

    Uses ``gradio_client`` which handles the Gradio protocol automatically.
    The endpoint must already be running (deploy with :func:`deploy_to_modal`
    or ``modal deploy src/modal_deploy.py``).
    """
    try:
        from gradio_client import Client
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gradio_client is required for remote prediction. "
            "Install it with: pip install gradio_client"
        ) from exc

    logger.debug("Calling remote endpoint: %s", endpoint_url)
    client = Client(endpoint_url)
    result = client.predict(
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
        api_name="/predict_temperature",
    )
    celsius, fahrenheit = float(result[0]), float(result[1])
    logger.debug("Remote prediction: %.2f °C / %.2f °F", celsius, fahrenheit)
    return celsius, fahrenheit, str(result[2])



def train_eval_deploy(force_retrain: bool = False) -> dict:
  
    from app import load_app_artifacts

    logger.info("Loading / training model (force_retrain=%s)", force_retrain)
    artifacts = load_app_artifacts(force_retrain=force_retrain)
    m = artifacts["meta"]["metrics"]
    logger.info(
        "Model ready — MAE: %.4f °C  RMSE: %.4f °C  R²: %.4f  samples: %d",
        m["mae"], m["rmse"], m["r2"], artifacts["meta"]["row_count"],
    )

    logger.info("Deploying to Modal…")
    url = deploy_to_modal()
    logger.info("Pipeline complete. Endpoint: %s", url)

    return {**artifacts, "endpoint_url": url}
