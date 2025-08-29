from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Tuple

import pandas as pd


@dataclass
class DatasetPaths:
    """Container for dataset-related paths.

    Attributes:
        root: Root directory where data files are stored.
        excel: Path to the Excel file containing the dataset.
    """

    root: Path
    excel: Path


def ensure_sample_excel(paths: DatasetPaths, days: int = 180) -> None:
    """Create a sample Excel dataset if it does not already exist.

    The dataset simulates a package size time series with a mild upward trend
    and weekly seasonality plus noise.

    Args:
        paths: DatasetPaths with `excel` location.
        days: Number of sequential days to generate.
    """
    paths.root.mkdir(parents=True, exist_ok=True)
    if paths.excel.exists():
        return

    start = date.today() - timedelta(days=days)
    records = []
    for i in range(days):
        d = start + timedelta(days=i)
        # Linear trend component (slow increase)
        trend = 0.15 * i
        # Weekly seasonality (sinusoidal) to reflect weekly release cycles
        weekly = 2.5 * math.sin(2 * math.pi * (i % 7) / 7)
        # Baseline size
        base = 100.0
        # Heteroscedastic noise, simple and small for reproducibility
        noise = 1.0 * math.sin(i * 0.3)
        size = base + trend + weekly + noise
        records.append({"date": pd.Timestamp(d), "package_size": round(size, 3)})

    df = pd.DataFrame.from_records(records)
    # Write as Excel using openpyxl engine if available; fallback to default
    try:
        df.to_excel(paths.excel, index=False)
    except ValueError:
        # Some environments require explicit engine; try openpyxl
        df.to_excel(paths.excel, index=False, engine="openpyxl")


def load_excel_as_xy(paths: DatasetPaths) -> Tuple[pd.Series, pd.Series]:
    """Load the Excel dataset and return features X and targets y.

    Features are encoded as a single numeric column representing the ordinal
    day index starting from 0. Targets are the `package_size` values.

    Args:
        paths: DatasetPaths with `excel` location.

    Returns:
        Tuple of (X_series, y_series) where X is `day_index` and y is `package_size`.
    """
    df = pd.read_excel(paths.excel)
    if "date" not in df.columns or "package_size" not in df.columns:
        raise ValueError("Excel must contain 'date' and 'package_size' columns")

    # Ensure date dtype and sort ascending
    df["date"] = pd.to_datetime(df["date"])  # type: ignore[assignment]
    df = df.sort_values("date").reset_index(drop=True)
    # Convert dates to an ordinal day index starting at 0
    df["day_index"] = (df["date"] - df["date"].min()).dt.days

    X = df["day_index"].astype(float)
    y = df["package_size"].astype(float)
    return X, y

