"""Feature engineering helpers for forecasting, ML notebooks, and app inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

APP_WEATHER_INPUT_COLUMNS = [
    "latitude",
    "longitude",
    "pressure_mb",
    "humidity",
    "cloud",
    "wind_kph",
    "gust_kph",
    "precip_mm",
    "visibility_km",
    "uv_index",
]

APP_MODEL_FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "pressure_mb",
    "humidity",
    "cloud",
    "wind_kph",
    "gust_kph",
    "precip_mm",
    "visibility_km",
    "uv_index",
    "year",
    "month",
    "day",
    "hour",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
]


def _build_time_features(timestamp: pd.Timestamp) -> dict[str, float | int]:
    """Create calendar and cyclic features from a timestamp."""

    return {
        "year": int(timestamp.year),
        "month": int(timestamp.month),
        "day": int(timestamp.day),
        "hour": int(timestamp.hour),
        "month_sin": float(np.sin(2 * np.pi * timestamp.month / 12.0)),
        "month_cos": float(np.cos(2 * np.pi * timestamp.month / 12.0)),
        "hour_sin": float(np.sin(2 * np.pi * timestamp.hour / 24.0)),
        "hour_cos": float(np.cos(2 * np.pi * timestamp.hour / 24.0)),
    }


def engineer_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-friendly weather features."""

    features = df.copy()
    if "last_updated" in features.columns:
        features["last_updated"] = pd.to_datetime(features["last_updated"], errors="coerce")
        features["year"] = features["last_updated"].dt.year
        features["month"] = features["last_updated"].dt.month
        features["day"] = features["last_updated"].dt.day
        features["day_of_week"] = features["last_updated"].dt.dayofweek
        features["day_of_year"] = features["last_updated"].dt.dayofyear
        features["hour"] = features["last_updated"].dt.hour
        features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12.0)
        features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12.0)
        features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24.0)
        features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24.0)

    if {"sunrise", "sunset"}.issubset(features.columns):
        sunrise = pd.to_datetime(features["sunrise"], format="%I:%M %p", errors="coerce")
        sunset = pd.to_datetime(features["sunset"], format="%I:%M %p", errors="coerce")
        daylight = (sunset - sunrise).dt.total_seconds().div(3600)
        features["daylight_hours"] = daylight.fillna(daylight.median())

    return features


def prepare_app_model_frame(
    df: pd.DataFrame,
    target_column: str = "temperature_celsius",
) -> tuple[pd.DataFrame, pd.Series]:
    """Create the leakage-free design matrix used by the Gradio app model."""

    engineered = engineer_weather_features(df)
    missing_columns = [column for column in APP_MODEL_FEATURE_COLUMNS if column not in engineered.columns]
    if missing_columns:
        raise KeyError(f"Missing required app model columns: {missing_columns}")

    X = engineered[APP_MODEL_FEATURE_COLUMNS].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    y = engineered[target_column].copy()
    return X, y


def build_app_prediction_frame(
    *,
    latitude: float,
    longitude: float,
    date_value: str | pd.Timestamp,
    hour: int,
    pressure_mb: float,
    humidity: float,
    cloud: float,
    wind_kph: float,
    gust_kph: float,
    precip_mm: float,
    visibility_km: float,
    uv_index: float,
) -> pd.DataFrame:
    """Build a single-row feature frame that matches the app model schema."""

    timestamp = pd.Timestamp(date_value)
    if pd.isna(timestamp):
        raise ValueError("A valid date is required to build prediction features.")
    timestamp = timestamp.normalize() + pd.Timedelta(hours=int(hour))

    feature_values = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "pressure_mb": float(pressure_mb),
        "humidity": float(humidity),
        "cloud": float(cloud),
        "wind_kph": float(wind_kph),
        "gust_kph": float(gust_kph),
        "precip_mm": float(precip_mm),
        "visibility_km": float(visibility_km),
        "uv_index": float(uv_index),
    }
    feature_values.update(_build_time_features(timestamp))
    return pd.DataFrame([feature_values], columns=APP_MODEL_FEATURE_COLUMNS)


def build_daily_temperature_series(
    df: pd.DataFrame,
    date_column: str = "last_updated",
    value_column: str = "temperature_celsius",
) -> pd.Series:
    """Aggregate the dataset into a daily average temperature series."""

    series_frame = df.copy()
    series_frame[date_column] = pd.to_datetime(series_frame[date_column], errors="coerce")
    series_frame = series_frame.dropna(subset=[date_column, value_column])
    series_frame = series_frame.set_index(date_column).sort_index()
    return series_frame[value_column].resample("D").mean().dropna()


def prepare_model_frame(
    df: pd.DataFrame,
    target_column: str = "temperature_celsius",
) -> tuple[pd.DataFrame, pd.Series]:
    """Create a numeric design matrix and target series."""

    engineered = engineer_weather_features(df)
    y = engineered[target_column].copy()
    drop_columns = {
        target_column,
        "last_updated",
        "country",
        "location_name",
        "timezone",
        "condition_text",
        "wind_direction",
        "sunrise",
        "sunset",
        "moonrise",
        "moonset",
        "moon_phase",
    }
    X = engineered.drop(columns=[column for column in drop_columns if column in engineered.columns])
    X = X.select_dtypes(include=["number"]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X, y
