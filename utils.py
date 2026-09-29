from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from enhanced_model import add_derived_metadata, transform_features

ROOT_DIR = Path(__file__).resolve().parent
FEATURE_COLUMNS = [
    "rating_count_lag1",
    "rating_count_lag2",
    "rating_count_lag3",
    "avg_rating_lag1",
    "unique_users_lag1",
    "rating_count_3m",
    "rating_count_6m",
    "users_3m",
]


def get_asset_path(filename: str) -> Path:
    configured_artifact_dir = os.getenv("STREAMIQ_ARTIFACT_DIR")
    candidates = [
        ROOT_DIR / filename,
        ROOT_DIR / "notebooks" / filename,
        ROOT_DIR / "models" / filename,
        ROOT_DIR / "data" / "processed" / filename,
    ]
    if configured_artifact_dir:
        candidates.insert(0, Path(configured_artifact_dir) / filename)
    elif Path("D:/").exists():
        candidates.insert(0, Path("D:/streamiq_artifacts") / filename)
    for path in candidates:
        if path.exists():
            return path
    return ROOT_DIR / filename


def load_model_package() -> dict[str, Any]:
    model_path = get_asset_path("streamiq_xgboost_model.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        model_package = joblib.load(model_path)
    except Exception as exc:  # pragma: no cover - user-facing error path
        raise ValueError(f"Unable to load the saved model: {exc}") from exc

    if not isinstance(model_package, dict):
        raise ValueError("The saved model package is invalid or corrupted.")

    model = model_package.get("model")
    features = model_package.get("features")
    if model is None or features is None:
        raise ValueError("The saved model package is missing the trained model or feature list.")

    if not isinstance(features, list):
        raise ValueError("The saved model feature list must be a Python list.")

    missing = [col for col in FEATURE_COLUMNS if col not in features]
    if missing:
        raise ValueError(f"Saved model is missing required model features: {missing}")

    return model_package


def load_enhanced_package() -> dict[str, Any] | None:
    model_path = get_asset_path("streamiq_xgboost_enhanced.pkl")
    if not model_path.exists():
        return None
    package = joblib.load(model_path)
    if not isinstance(package, dict) or package.get("model") is None or package.get("schema") is None:
        raise ValueError("The enhanced model package is invalid or incomplete.")
    return package


def load_enhanced_metadata(package: dict[str, Any]) -> pd.DataFrame:
    metadata_path = Path(package.get("metadata_path", ""))
    if not metadata_path.exists():
        metadata_path = get_asset_path("movie_metadata_enriched.csv")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Enhanced metadata file not found: {metadata_path}")
    return pd.read_csv(metadata_path)


def load_streamiq_data() -> pd.DataFrame:
    csv_path = get_asset_path("streamiq_streamlit_data.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    try:
        data = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - user-facing error path
        raise ValueError(f"Unable to read the dataset: {exc}") from exc

    expected = {"movieId", "title", "month", *FEATURE_COLUMNS}
    missing = expected.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    data = data.copy()
    data["month"] = pd.to_datetime(data["month"], errors="coerce")
    return data


def load_thresholds() -> np.ndarray:
    threshold_path = get_asset_path("streamiq_popularity_thresholds.pkl")
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold file not found: {threshold_path}")

    try:
        thresholds = joblib.load(threshold_path)
    except Exception as exc:  # pragma: no cover - user-facing error path
        raise ValueError(f"Unable to load the popularity threshold file: {exc}") from exc

    arr = np.asarray(thresholds, dtype=float)
    if arr.size < 3:
        raise ValueError("The popularity thresholds file does not contain enough thresholds for classification.")
    return arr


def get_popularity_label(predicted_count: float, thresholds: np.ndarray) -> str:
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.size < 3:
        return "N/A"
    if predicted_count >= thresholds[2]:
        return "Very High"
    if predicted_count >= thresholds[1]:
        return "High"
    if predicted_count >= thresholds[0]:
        return "Medium"
    return "Low"


def extract_release_year(title: str | None) -> str:
    if not title or not isinstance(title, str):
        return "N/A"
    match = re.search(r"\((\d{4})\)", title)
    if match:
        return match.group(1)
    return "N/A"


def make_prediction_row(row: pd.Series, model: Any, thresholds: np.ndarray) -> tuple[pd.Series, float, str]:
    feature_values = row[FEATURE_COLUMNS].astype(float).to_numpy()
    feature_frame = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)
    predicted_log = float(model.predict(feature_frame)[0])
    predicted_count = float(np.expm1(predicted_log))
    outlook = get_popularity_label(predicted_count, thresholds)
    return row, predicted_count, outlook


def make_enhanced_prediction_row(
    row: pd.Series,
    package: dict[str, Any],
    metadata: pd.DataFrame,
    thresholds: np.ndarray,
) -> tuple[pd.Series, float, str]:
    enriched = add_derived_metadata(pd.DataFrame([row]), metadata)
    feature_frame = transform_features(enriched, package["schema"])
    predicted_log = float(package["model"].predict(feature_frame)[0])
    predicted_count = float(np.expm1(predicted_log))
    outlook = get_popularity_label(predicted_count, thresholds)
    return enriched.iloc[0], predicted_count, outlook
