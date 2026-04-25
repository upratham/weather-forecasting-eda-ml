"""Data loading, cleaning, and anomaly helpers for the weather project."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_RAW_DATA_PATH = Path("data/raw/GlobalWeatherRepository.csv")
DEFAULT_PROCESSED_DIR = Path("data/processed")

DATETIME_COLUMNS = ["last_updated"]
CATEGORICAL_COLUMNS = [
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
]


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve() if not Path(path).is_absolute() else Path(path)


def load_raw_weather_data(path: str | Path = DEFAULT_RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw weather CSV."""

    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def load_processed_data(path: str | Path) -> pd.DataFrame:
    """Load a processed CSV file."""

    logger.info("Loading processed data from %s", path)
    df = pd.read_csv(path, parse_dates=["last_updated"], low_memory=False)
    logger.info("Loaded %d rows", len(df))
    return df


def get_numeric_columns(
    df: pd.DataFrame,
    exclude: Iterable[str] | None = None,
) -> list[str]:
    """Return numeric columns after excluding requested fields."""

    excluded = set(exclude or [])
    return [column for column in df.select_dtypes(include=[np.number]).columns if column not in excluded]


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a consistent cleaning pipeline to the raw weather data."""

    logger.info("Cleaning %d rows", len(df))
    cleaned = df.copy()

    for column in DATETIME_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")

    numeric_candidates = [
        column
        for column in cleaned.columns
        if column not in CATEGORICAL_COLUMNS and column not in DATETIME_COLUMNS
    ]
    for column in numeric_candidates:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    dropped = before - len(cleaned)
    if dropped:
        logger.debug("Dropped %d duplicate rows", dropped)

    numeric_columns = get_numeric_columns(cleaned)
    for column in numeric_columns:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    for column in cleaned.columns:
        if column in numeric_columns or column in DATETIME_COLUMNS:
            continue
        mode = cleaned[column].mode(dropna=True)
        cleaned[column] = cleaned[column].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    if "last_updated" in cleaned.columns:
        cleaned = cleaned.dropna(subset=["last_updated"]).sort_values("last_updated").reset_index(drop=True)

    logger.info("Clean dataset: %d rows remaining", len(cleaned))

    if "humidity" in cleaned.columns:
        cleaned["humidity"] = cleaned["humidity"].clip(lower=0, upper=100)

    if "cloud" in cleaned.columns:
        cleaned["cloud"] = cleaned["cloud"].clip(lower=0, upper=100)

    if "moon_illumination" in cleaned.columns:
        cleaned["moon_illumination"] = cleaned["moon_illumination"].clip(lower=0, upper=100)

    # Cap physically impossible values (outliers that survive anomaly detection)
    if "wind_kph" in cleaned.columns:
        cleaned["wind_kph"] = cleaned["wind_kph"].clip(lower=0, upper=200)
    if "gust_kph" in cleaned.columns:
        cleaned["gust_kph"] = cleaned["gust_kph"].clip(lower=0, upper=250)
    if "pressure_mb" in cleaned.columns:
        cleaned["pressure_mb"] = cleaned["pressure_mb"].clip(lower=870, upper=1100)
    if "visibility_km" in cleaned.columns:
        cleaned["visibility_km"] = cleaned["visibility_km"].clip(lower=0, upper=50)

    return cleaned


def compute_zscore_outliers(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    threshold: float = 3.0,
) -> pd.Series:
    """Return a boolean series for rows with at least one extreme z-score."""

    numeric_columns = list(columns) if columns is not None else get_numeric_columns(df)
    if not numeric_columns:
        return pd.Series(False, index=df.index, name="zscore_outlier")

    z_scores = np.abs(stats.zscore(df[numeric_columns], nan_policy="omit"))
    if z_scores.ndim == 1:
        mask = z_scores > threshold
    else:
        mask = (z_scores > threshold).any(axis=1)
    return pd.Series(mask, index=df.index, name="zscore_outlier")


def add_isolation_forest_anomalies(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Add anomaly labels and scores using Isolation Forest."""

    numeric_columns = list(columns) if columns is not None else get_numeric_columns(df)
    logger.info("Running Isolation Forest on %d rows, %d features", len(df), len(numeric_columns))
    scored = df.copy()
    if not numeric_columns:
        logger.warning("No numeric columns found; skipping anomaly detection")
        scored["anomaly"] = 1
        scored["anomaly_score"] = 0.0
        return scored

    scaler = StandardScaler()
    values = scored[numeric_columns].fillna(0.0)
    scaled_values = scaler.fit_transform(values)

    detector = IsolationForest(contamination=contamination, random_state=random_state)
    scored["anomaly"] = detector.fit_predict(scaled_values)
    scored["anomaly_score"] = detector.decision_function(scaled_values)
    n_anomalies = int((scored["anomaly"] == -1).sum())
    logger.info("Anomaly detection complete: %d anomalies (%.1f%%)", n_anomalies, 100 * n_anomalies / len(df))
    return scored


def remove_anomalies(df: pd.DataFrame, anomaly_column: str = "anomaly") -> pd.DataFrame:
    """Keep only the rows considered normal by the anomaly detector."""

    if anomaly_column not in df.columns:
        logger.warning("Column '%s' not found; returning all rows", anomaly_column)
        return df.copy()
    result = df[df[anomaly_column] == 1].copy()
    logger.info("Removed %d anomalous rows; %d rows remaining", len(df) - len(result), len(result))
    return result


def save_processed_data(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist a processed dataframe and return the resolved path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)
    return output_path
