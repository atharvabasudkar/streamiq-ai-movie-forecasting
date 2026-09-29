from __future__ import annotations

from pathlib import Path
from typing import Any
import os

import numpy as np
import pandas as pd

BASE_FEATURE_COLUMNS = [
    "rating_count_lag1",
    "rating_count_lag2",
    "rating_count_lag3",
    "avg_rating_lag1",
    "unique_users_lag1",
    "rating_count_3m",
    "rating_count_6m",
    "users_3m",
]
METADATA_NUMERIC_COLUMNS = ["release_year", "runtime", "genre_count", "movie_age"]
METADATA_CATEGORICAL_COLUMNS = ["language", "country"]
UNKNOWN = "Unknown"


def get_artifact_dir() -> Path:
    configured = os.getenv("STREAMIQ_ARTIFACT_DIR")
    if configured:
        path = Path(configured)
    elif Path("D:/").exists():
        path = Path("D:/streamiq_artifacts")
    else:
        path = Path(__file__).resolve().parent / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_release_year(title: str | None) -> float:
    if not isinstance(title, str):
        return np.nan
    import re

    match = re.search(r"\((\d{4})\)", title)
    return float(match.group(1)) if match else np.nan


def build_base_metadata(movies_path: Path, links_path: Path) -> pd.DataFrame:
    movies = pd.read_csv(movies_path)
    links = pd.read_csv(links_path, dtype={"movieId": "int64", "tmdbId": "string"})
    metadata = movies[["movieId", "title", "genres"]].copy()
    metadata["release_year"] = metadata["title"].map(parse_release_year)
    metadata["genres"] = metadata["genres"].fillna(UNKNOWN).replace("(no genres listed)", UNKNOWN)
    metadata["genre_count"] = metadata["genres"].map(lambda value: 0 if value == UNKNOWN else len(str(value).split("|")))
    metadata["runtime"] = np.nan
    metadata["language"] = UNKNOWN
    metadata["country"] = UNKNOWN
    metadata = metadata.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    return metadata


def add_derived_metadata(data: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    result = data.merge(metadata, on="movieId", how="left", suffixes=("", "_metadata"))
    result["month"] = pd.to_datetime(result["month"], errors="coerce")
    result["analysis_year"] = result["month"].dt.year
    result["movie_age"] = result["analysis_year"] - result["release_year"]
    result["language"] = result["language"].fillna(UNKNOWN).astype(str)
    result["country"] = result["country"].fillna(UNKNOWN).astype(str)
    result["genres"] = result["genres"].fillna(UNKNOWN).astype(str)
    return result


def fit_feature_schema(training_data: pd.DataFrame) -> dict[str, Any]:
    genre_values: set[str] = set()
    for value in training_data["genres"].fillna(UNKNOWN):
        if value != UNKNOWN:
            genre_values.update(str(value).split("|"))
    genre_columns = sorted(genre_values)
    category_values = {
        column: sorted(set(training_data[column].fillna(UNKNOWN).astype(str)))
        for column in METADATA_CATEGORICAL_COLUMNS
    }
    medians = {
        column: float(training_data[column].replace([np.inf, -np.inf], np.nan).median())
        for column in METADATA_NUMERIC_COLUMNS
    }
    medians["runtime"] = medians.get("runtime", np.nan)
    if not np.isfinite(medians["runtime"]):
        medians["runtime"] = 90.0
    for column in METADATA_NUMERIC_COLUMNS:
        if not np.isfinite(medians[column]):
            medians[column] = 0.0

    feature_columns = BASE_FEATURE_COLUMNS + METADATA_NUMERIC_COLUMNS
    feature_columns += [f"genre__{genre}" for genre in genre_columns]
    feature_columns += [f"language__{value}" for value in category_values["language"]]
    feature_columns += [f"country__{value}" for value in category_values["country"]]
    return {
        "base_features": BASE_FEATURE_COLUMNS,
        "numeric_features": METADATA_NUMERIC_COLUMNS,
        "categorical_features": METADATA_CATEGORICAL_COLUMNS,
        "genre_columns": genre_columns,
        "categories": category_values,
        "medians": medians,
        "feature_columns": feature_columns,
    }


def transform_features(data: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(index=data.index)
    for column in BASE_FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(data[column], errors="coerce")

    for column in METADATA_NUMERIC_COLUMNS:
        values = pd.to_numeric(data[column], errors="coerce")
        frame[column] = values.fillna(schema["medians"][column])

    for genre in schema["genre_columns"]:
        frame[f"genre__{genre}"] = data["genres"].map(
            lambda value: float(genre in str(value).split("|")) if value != UNKNOWN else 0.0
        )

    for column in METADATA_CATEGORICAL_COLUMNS:
        values = data[column].fillna(UNKNOWN).astype(str)
        allowed = set(schema["categories"][column])
        for category in schema["categories"][column]:
            frame[f"{column}__{category}"] = (values == category).astype(float)
        unknown_mask = ~values.isin(allowed)
        unknown_column = f"{column}__{UNKNOWN}"
        if unknown_column in frame:
            frame.loc[unknown_mask, unknown_column] = 1.0

    return frame.reindex(columns=schema["feature_columns"], fill_value=0.0).astype(np.float32)


def prepare_metadata_artifact(metadata: pd.DataFrame) -> pd.DataFrame:
    columns = ["movieId", "title", "genres", "release_year", "runtime", "language", "country", "genre_count", "tmdbId"]
    result = metadata[[column for column in columns if column in metadata.columns]].copy()
    return result.drop_duplicates("movieId")
