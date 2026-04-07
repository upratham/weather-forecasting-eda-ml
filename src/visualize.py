"""Reusable plotting helpers for weather analysis notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def save_figure(fig: plt.Figure, path: str | Path) -> Path:
    """Save a matplotlib figure and return the saved path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    return output_path


def plot_missing_values_heatmap(df: pd.DataFrame, title: str = "Missing Values Heatmap") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    return fig


def plot_numeric_distributions(
    df: pd.DataFrame,
    columns: Iterable[str],
    bins: int = 30,
    title: str = "Numeric Feature Distributions",
) -> plt.Figure:
    columns = list(columns)
    fig, axes = plt.subplots(1, len(columns), figsize=(6 * len(columns), 4))
    if len(columns) == 1:
        axes = [axes]
    for ax, column in zip(axes, columns):
        sns.histplot(df[column].dropna(), kde=True, bins=bins, ax=ax)
        ax.set_title(column.replace("_", " ").title())
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_temperature_vs_feels_like(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="temperature_celsius",
        y="feels_like_celsius",
        alpha=0.4,
        ax=ax,
    )
    ax.set_title("Temperature vs Feels Like")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    title: str = "Correlation Heatmap",
) -> plt.Figure:
    subset = df[list(columns)] if columns is not None else df.select_dtypes(include=["number"])
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(subset.corr(numeric_only=True), cmap="coolwarm", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_air_quality_distributions(df: pd.DataFrame, columns: Iterable[str]) -> plt.Figure:
    columns = list(columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for ax, column in zip(axes, columns):
        sns.histplot(df[column].dropna(), bins=30, kde=True, ax=ax)
        ax.set_title(column.replace("_", " ").title())
    for ax in axes[len(columns) :]:
        ax.axis("off")
    fig.suptitle("Air Quality Distributions", y=1.02)
    fig.tight_layout()
    return fig


def plot_scatter_with_hue(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.6, ax=ax)
    ax.set_title(title or f"{x} vs {y}")
    fig.tight_layout()
    return fig


def plot_top_categories(
    series: pd.Series,
    top_n: int = 10,
    title: str = "Top Categories",
    color: str = "steelblue",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    series.value_counts().head(top_n).plot(kind="bar", color=color, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_kde_comparison(
    df: pd.DataFrame,
    value_column: str,
    group_column: str,
    label_map: dict[int, str],
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    for raw_value, label in label_map.items():
        subset = df[df[group_column] == raw_value][value_column].dropna()
        if subset.empty:
            continue
        sns.kdeplot(subset, label=label, ax=ax)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
