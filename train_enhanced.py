from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from enhanced_model import (
    BASE_FEATURE_COLUMNS,
    add_derived_metadata,
    build_base_metadata,
    fit_feature_schema,
    get_artifact_dir,
    prepare_metadata_artifact,
    transform_features,
)
from utils import load_model_package

ROOT_DIR = Path(__file__).resolve().parent
RAW_DIR = ROOT_DIR / "data" / "raw" / "ml-32m"
NOTEBOOK_DIR = ROOT_DIR / "notebooks"
ARTIFACT_DIR = get_artifact_dir()
METADATA_PATH = ARTIFACT_DIR / "movie_metadata_enriched.csv"
ENHANCED_ARTIFACT_PATH = ARTIFACT_DIR / "streamiq_xgboost_enhanced.pkl"
METRICS_PATH = ARTIFACT_DIR / "streamiq_model_comparison.json"
SCHEMA_PATH = ARTIFACT_DIR / "streamiq_feature_schema.json"


def future_targets_from_ratings(ratings_path: Path) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(ratings_path, usecols=["movieId", "timestamp"], chunksize=1_000_000):
        timestamps = pd.to_datetime(chunk["timestamp"], unit="s", errors="coerce")
        chunk["month"] = timestamps.dt.to_period("M").dt.to_timestamp()
        counts = chunk.dropna(subset=["month"]).groupby(["movieId", "month"], as_index=False).size()
        counts = counts.rename(columns={"size": "actual_rating_count"})
        pieces.append(counts)
    monthly = pd.concat(pieces, ignore_index=True)
    monthly = monthly.groupby(["movieId", "month"], as_index=False)["actual_rating_count"].sum()
    monthly["month"] = monthly["month"] - pd.offsets.MonthBegin(1)
    monthly = monthly.rename(columns={"actual_rating_count": "future_rating_count"})
    return monthly


def enrich_tmdb_metadata(metadata: pd.DataFrame, output_path: Path, limit: int | None = None) -> pd.DataFrame:
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        return metadata

    session = requests.Session()
    session.headers.update({"User-Agent": "STREAMIQ/1.0"})
    pending = metadata[metadata["tmdbId"].notna()].copy()
    if limit is not None:
        pending = pending.head(limit)
    for index, row in pending.iterrows():
        try:
            response = session.get(
                f"https://api.themoviedb.org/3/movie/{int(float(row['tmdbId']))}",
                params={"api_key": api_key, "language": "en-US"},
                timeout=10,
            )
            if response.status_code == 429:
                break
            response.raise_for_status()
            payload = response.json()
            metadata.loc[index, "runtime"] = payload.get("runtime")
            metadata.loc[index, "language"] = payload.get("original_language") or "Unknown"
            countries = payload.get("production_countries") or []
            metadata.loc[index, "country"] = "|".join(
                sorted({item.get("iso_3166_1") for item in countries if item.get("iso_3166_1")})
            ) or "Unknown"
            if metadata.loc[index, "genres"] == "Unknown":
                genres = payload.get("genres") or []
                metadata.loc[index, "genres"] = "|".join(
                    sorted({item.get("name") for item in genres if item.get("name")})
                ) or "Unknown"
            metadata.loc[index, "genre_count"] = 0 if metadata.loc[index, "genres"] == "Unknown" else len(str(metadata.loc[index, "genres"]).split("|"))
        except requests.RequestException:
            continue
        if index % 100 == 0:
            metadata.to_csv(output_path, index=False)
    return metadata


def metrics(model: Any, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    prediction = model.predict(x)
    return {
        "MAE": float(mean_absolute_error(y, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(y, prediction))),
        "R2": float(r2_score(y, prediction)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the offline STREAMIQ enhanced model.")
    parser.add_argument("--tmdb-limit", type=int, default=None)
    parser.add_argument("--skip-target-scan", action="store_true")
    args = parser.parse_args()

    data = pd.read_csv(NOTEBOOK_DIR / "streamiq_streamlit_data.csv", parse_dates=["month"])
    metadata = build_base_metadata(RAW_DIR / "movies.csv", RAW_DIR / "links.csv")
    if METADATA_PATH.exists():
        metadata = pd.read_csv(METADATA_PATH)
    metadata = enrich_tmdb_metadata(metadata, METADATA_PATH, args.tmdb_limit)
    metadata = prepare_metadata_artifact(metadata)
    metadata.to_csv(METADATA_PATH, index=False)

    if args.skip_target_scan:
        target = data.sort_values(["movieId", "month"]).copy()
        target["future_rating_count"] = target.groupby("movieId")["rating_count_lag1"].shift(-1)
        target = target[["movieId", "month", "future_rating_count"]]
    else:
        target = future_targets_from_ratings(RAW_DIR / "ratings.csv")
    data = data.merge(target, on=["movieId", "month"], how="left")
    data = add_derived_metadata(data, metadata)
    data["future_popularity"] = np.log1p(data["future_rating_count"])
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=BASE_FEATURE_COLUMNS + ["future_popularity", "month"]).reset_index(drop=True)

    train = data[data["month"] <= pd.Timestamp("2021-12-01")].copy()
    validation = data[(data["month"] >= pd.Timestamp("2022-01-01")) & (data["month"] <= pd.Timestamp("2022-12-01"))].copy()
    test = data[(data["month"] >= pd.Timestamp("2023-01-01")) & (data["month"] <= pd.Timestamp("2023-09-01"))].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Chronological train, validation, and test partitions must all contain rows.")

    baseline_package = load_model_package()
    baseline_metrics = {
        "validation": metrics(baseline_package["model"], validation[BASE_FEATURE_COLUMNS], validation["future_popularity"]),
        "test": metrics(baseline_package["model"], test[BASE_FEATURE_COLUMNS], test["future_popularity"]),
    }

    schema = fit_feature_schema(train)
    x_train = transform_features(train, schema)
    x_validation = transform_features(validation, schema)
    x_test = transform_features(test, schema)
    y_train = train["future_popularity"]
    y_validation = validation["future_popularity"]
    y_test = test["future_popularity"]

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_validation, y_validation)], verbose=False)
    enhanced_metrics = {
        "validation": metrics(model, x_validation, y_validation),
        "test": metrics(model, x_test, y_test),
    }

    artifact = {
        "model": model,
        "schema": schema,
        "metadata": metadata,
        "target": "future_popularity",
        "target_transformation": "log1p",
        "inverse_transformation": "expm1",
        "model_name": "XGBoost Enhanced Historical Engagement + Movie Metadata",
        "metrics": enhanced_metrics,
        "metadata_path": str(METADATA_PATH),
    }
    joblib.dump(artifact, ENHANCED_ARTIFACT_PATH)
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    comparison = {
        "baseline": baseline_metrics,
        "enhanced": enhanced_metrics,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "enhanced_feature_count": len(schema["feature_columns"]),
        "test_period": "2023-01-01 to 2023-09-01",
    }
    METRICS_PATH.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
