"""Training helpers for tabular weather models."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge

from .eval import evaluate_regression, summarize_results


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_voting_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> VotingRegressor:
    """Average predictions of LR + RF + GB (soft voting by mean)."""
    estimators = [
        ("lr", LinearRegression()),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)),
        ("gb", GradientBoostingRegressor(n_estimators=100, random_state=random_state)),
    ]
    model = VotingRegressor(estimators=estimators, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_stacking_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> StackingRegressor:
    """Stack LR + RF + GB with a Ridge meta-learner using 5-fold CV."""
    estimators = [
        ("lr", LinearRegression()),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)),
        ("gb", GradientBoostingRegressor(n_estimators=100, random_state=random_state)),
    ]
    model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(),
        cv=5,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compare_regression_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    include_ensemble: bool = True,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Train a model suite (base + ensemble) and return fitted models plus metrics."""

    models: dict[str, object] = {
        "linear_regression": train_linear_regression(X_train, y_train),
        "gradient_boosting": train_gradient_boosting_regressor(X_train, y_train),
        "random_forest": train_random_forest_regressor(X_train, y_train),
    }
    if include_ensemble:
        models["voting_ensemble"] = train_voting_ensemble(X_train, y_train)
        models["stacking_ensemble"] = train_stacking_ensemble(X_train, y_train)

    metrics = {
        name: evaluate_regression(y_test, model.predict(X_test))
        for name, model in models.items()
    }
    return models, summarize_results(metrics)


def save_model(model: object, path: str | Path) -> Path:
    """Persist a fitted model using joblib."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path
