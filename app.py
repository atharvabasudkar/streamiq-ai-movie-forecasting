from __future__ import annotations

import os
from html import escape
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from tmdb import get_tmdb_poster_url
from utils import (
    FEATURE_COLUMNS,
    extract_release_year,
    get_asset_path,
    get_popularity_label,
    load_enhanced_metadata,
    load_enhanced_package,
    load_model_package,
    load_streamiq_data,
    load_thresholds,
    make_prediction_row,
    make_enhanced_prediction_row,
)

st.set_page_config(
    page_title="STREAMIQ | AI Movie Popularity Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_runtime_resources() -> dict[str, Any]:
    try:
        model_package = load_model_package()
        model = model_package["model"]
        thresholds = load_thresholds()
        data = load_streamiq_data()
        enhanced_package = load_enhanced_package()
        enhanced_metadata = load_enhanced_metadata(enhanced_package) if enhanced_package else None
        return {
            "model": model,
            "thresholds": thresholds,
            "data": data,
            "model_package": model_package,
            "enhanced_package": enhanced_package,
            "enhanced_metadata": enhanced_metadata,
        }
    except Exception as exc:  # pragma: no cover - app-level safe error handling
        st.error(f"Unable to initialize STREAMIQ: {exc}")
        st.stop()


@st.cache_data(show_spinner=False)
def get_movie_catalog(data: pd.DataFrame) -> pd.DataFrame:
    movie_catalog = (
        data.groupby("movieId", as_index=False)
        .agg(
            title=("title", "first"),
            first_month=("month", "min"),
            last_month=("month", "max"),
            total_records=("movieId", "count"),
            avg_rating=("avg_rating_lag1", "mean"),
        )
        .sort_values(["title", "movieId"], kind="mergesort")
        .reset_index(drop=True)
    )
    return movie_catalog


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def format_metric(value: Any, suffix: str = "") -> str:
    numeric = safe_float(value)
    if pd.isna(numeric):
        return "N/A"
    if suffix == "%":
        return f"{numeric:.1f}%"
    return f"{numeric:,.0f}{suffix}"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp { background: #0b0f14; color: #f8fafc; }
        [data-testid="stHeader"] { background: rgba(11,15,20,0.92); }
        [data-testid="stSidebar"] { background: #111821; border-right: 1px solid #273342; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label { color: #cbd5e1 !important; }
        [data-baseweb="select"] > div { background: #1a2430; border-color: #334155; color: #f8fafc; }
        [data-testid="stMetric"] { background: #151d27; border: 1px solid #283646; border-radius: 14px; padding: 0.8rem; }
        .block-container { max-width: 1440px; padding: 2.5rem 3rem 4rem; }
        h1, h2, h3 { color: #f8fafc !important; }
        p, [data-testid="stCaptionContainer"] { color: #aebdcd; }
        [data-testid="stImage"] img { border-radius: 18px; object-fit: cover; box-shadow: 0 18px 40px rgba(0,0,0,.35); }
        @media (max-width: 800px) { .block-container { padding: 1.25rem 1rem 3rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div style='padding: 1.2rem 0 1.8rem 0;'><h1 style='margin:0; font-size:3.1rem; font-weight:800; letter-spacing:0.04em;'>STREAMIQ</h1>
        <div style='font-size:1.05rem; letter-spacing:0.22em; color:#e11d48; margin-top:0.5rem; font-weight:700;'>AI MOVIE POPULARITY INTELLIGENCE</div>
        <p style='margin-top:1.2rem; max-width:900px; color:#d4d4d8; font-size:1.1rem;'>Forecast future movie audience activity using historical engagement and movie metadata.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(resources: dict[str, Any]) -> tuple[Any, pd.DataFrame, pd.Series, str | None]:
    with st.sidebar:
        st.title("STREAMIQ")
        st.caption("AI Movie Popularity Intelligence")

        catalog = get_movie_catalog(resources["data"])
        if catalog.empty:
            st.warning("No movie records were found in the dataset.")
            st.stop()

        movie_titles = catalog["title"].astype(str).tolist()
        selected_title = st.selectbox("SEARCH MOVIE", options=movie_titles, index=0)
        selected_movie = catalog[catalog["title"] == selected_title].iloc[0]
        movie_id = int(selected_movie["movieId"])

        movie_rows = resources["data"][resources["data"]["movieId"] == movie_id].sort_values("month").copy()
        if movie_rows.empty:
            st.warning("No historical rows found for the selected movie.")
            st.stop()

        month_options = pd.to_datetime(movie_rows["month"]).dt.strftime("%Y-%m-%d").tolist()
        selected_month = st.selectbox("ANALYSIS MONTH", options=month_options, index=len(month_options) - 1)
        st.markdown("---")
        st.subheader("Movie Details")
        st.write(f"Movie ID: {movie_id}")
        st.write(f"Release year: {extract_release_year(selected_title)}")
        st.write(f"Historical months: {len(month_options)}")

        return selected_title, movie_rows, selected_movie, selected_month


def safe_tmdb_url(title: str, movie_id: int | None = None) -> str:
    try:
        return get_tmdb_poster_url(title=title, movie_id=movie_id)
    except Exception:
        return None


def render_movie_header(
    title: str,
    movie_id: int,
    selected_data: pd.DataFrame,
    selected_month: str,
    metadata: pd.Series | None = None,
) -> tuple[str, str]:
    movie_row = selected_data[selected_data["month"] == pd.to_datetime(selected_month)].iloc[0] if not selected_data.empty else selected_data.iloc[-1]
    poster_url = safe_tmdb_url(title, movie_id)

    col_left, col_middle, col_right = st.columns([1.3, 2.2, 1.3])
    with col_left:
        if poster_url:
            st.image(poster_url, width=220)
        else:
            st.markdown(
                f"""
                <div style='width:220px; height:330px; border-radius:18px; background:linear-gradient(145deg,#263244,#111827 72%); display:flex; flex-direction:column; align-items:center; justify-content:center; padding:1.2rem; box-sizing:border-box; color:#f8fafc; text-align:center; box-shadow:0 18px 40px rgba(0,0,0,.35);'>
                    <div style='font-size:.72rem; letter-spacing:.18em; color:#f87171; font-weight:800;'>STREAMIQ</div>
                    <div style='margin-top:1.2rem; font-size:1.15rem; line-height:1.25; font-weight:800;'>{escape(title.rsplit("(", 1)[0].strip())}</div>
                    <div style='margin-top:.7rem; color:#94a3b8; font-size:.8rem;'>Poster unavailable</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_middle:
        st.subheader(title)
        st.caption(f"Release year: {extract_release_year(title)}")
        if metadata is not None:
            genre = metadata.get("genres", "Unknown")
            language = metadata.get("language", "Unknown")
            country = metadata.get("country", "Unknown")
            runtime = metadata.get("runtime", np.nan)
            runtime_label = f"{int(runtime)} min" if pd.notna(runtime) else "Unknown runtime"
            st.write(f"Genre: {genre}")
            st.write(f"Language: {language} · Country: {country} · Runtime: {runtime_label}")
        st.write(f"Movie ID: {movie_id}")
        st.write(f"Available historical months: {selected_data.shape[0]}")
        st.write(f"Selected analysis month: {pd.to_datetime(selected_month).strftime('%Y-%m-%d')}")

    with col_right:
        st.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style='background: linear-gradient(135deg,#7f1d1d,#b91c1c); border-radius:18px; padding:1rem; color:white; font-weight:700; text-align:center;'>
            FORECAST READY
            </div>
            """,
            unsafe_allow_html=True,
        )

    return poster_url or "", title


def render_metric_cards(movie_row: pd.Series, model: Any, thresholds: np.ndarray, movie_title: str) -> None:
    cards = [
        {"label": "Previous Month", "value": movie_row["rating_count_lag1"] if "rating_count_lag1" in movie_row else np.nan},
        {"label": "Previous 2 Months", "value": movie_row["rating_count_lag2"] if "rating_count_lag2" in movie_row else np.nan},
        {"label": "Previous 3 Months", "value": movie_row["rating_count_lag3"] if "rating_count_lag3" in movie_row else np.nan},
        {"label": "3-Month Rating Activity", "value": movie_row["rating_count_3m"] if "rating_count_3m" in movie_row else np.nan},
        {"label": "6-Month Rating Activity", "value": movie_row["rating_count_6m"] if "rating_count_6m" in movie_row else np.nan},
        {"label": "3-Month Users", "value": movie_row["users_3m"] if "users_3m" in movie_row else np.nan},
        {"label": "Previous Average Rating", "value": movie_row["avg_rating_lag1"] if "avg_rating_lag1" in movie_row else np.nan},
    ]

    cols = st.columns(7)
    for col, card in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div style='background: rgba(24,24,27,0.96); border-radius:16px; padding:1rem 1rem 0.8rem; min-height:115px; border:1px solid rgba(255,255,255,0.06);'>
                    <div style='color:#a1a1aa; font-size:0.74rem; letter-spacing:0.08em; text-transform: uppercase;'>{card['label']}</div>
                    <div style='margin-top:0.8rem; font-size:1.8rem; font-weight:700; color:#f4f4f5;'>{format_metric(card['value'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_forecast_section(
    movie_row: pd.Series,
    model: Any,
    thresholds: np.ndarray,
    enhanced_package: dict[str, Any] | None = None,
    enhanced_metadata: pd.DataFrame | None = None,
) -> tuple[float, str]:
    if enhanced_package is not None and enhanced_metadata is not None:
        _, pred_count, outlook = make_enhanced_prediction_row(
            movie_row, enhanced_package, enhanced_metadata, thresholds
        )
    else:
        _, pred_count, outlook = make_prediction_row(movie_row, model, thresholds)
    st.markdown("### AI FORECAST")
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, rgba(127,29,29,0.25), rgba(17,24,39,0.9)); border:1px solid rgba(239,68,68,0.35); border-radius:20px; padding:1.8rem; margin-top:0.6rem;'>
            <div style='color:#fca5a5; letter-spacing:0.12em; text-transform:uppercase; font-size:0.72rem; font-weight:700;'>Predicted Next-Month Rating Activity</div>
            <div style='font-size:3.1rem; font-weight:800; margin-top:0.5rem; color:#f9fafb;'>{pred_count:,.0f} ratings</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Popularity Outlook")
    outlook_html = {
        "Low": "#7f1d1d",
        "Medium": "#b45309",
        "High": "#1d4ed8",
        "Very High": "#15803d",
    }
    color = outlook_html.get(outlook, "#7f1d1d")
    st.markdown(
        f"""
        <div style='background:{color}; border-radius:18px; padding:1.2rem; text-align:center; font-size:2rem; font-weight:800; letter-spacing:0.12em;'>{outlook}</div>
        """,
        unsafe_allow_html=True,
    )
    return pred_count, outlook


def render_metadata_panel(metadata: pd.Series | None, selected_month: str) -> None:
    if metadata is None:
        return
    release_year = metadata.get("release_year", np.nan)
    analysis_year = pd.to_datetime(selected_month).year
    movie_age = analysis_year - int(release_year) if pd.notna(release_year) else "Unknown"
    runtime = metadata.get("runtime", np.nan)
    runtime_label = f"{int(runtime)} min" if pd.notna(runtime) else "Unknown"
    values = [
        ("Genre", metadata.get("genres", "Unknown")),
        ("Release Year", int(release_year) if pd.notna(release_year) else "Unknown"),
        ("Movie Age", movie_age),
        ("Runtime", runtime_label),
        ("Language", metadata.get("language", "Unknown")),
        ("Country", metadata.get("country", "Unknown")),
        ("Genre Count", int(metadata.get("genre_count", 0))),
    ]
    st.markdown("### MOVIE METADATA")
    columns = st.columns(4)
    for index, (label, value) in enumerate(values):
        with columns[index % 4]:
            st.markdown(
                f"<div style='background:#151d27;border:1px solid #283646;border-radius:12px;padding:.8rem;margin-bottom:.7rem;min-height:72px;'><div style='color:#8fa1b5;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;'>{label}</div><div style='color:#f8fafc;font-weight:700;margin-top:.35rem;overflow-wrap:anywhere;'>{value}</div></div>",
                unsafe_allow_html=True,
            )


def render_explainability(movie_row: pd.Series) -> None:
    st.markdown("### WHY DID STREAMIQ MAKE THIS FORECAST?")
    st.write("These historical engagement signals were provided to the XGBoost model.")

    signal_cards = [
        ("Recent Rating Activity", ["rating_count_lag1", "rating_count_3m", "rating_count_6m"]),
        ("Previous Month Ratings", ["rating_count_lag1"]),
        ("Previous 2 Months Ratings", ["rating_count_lag2"]),
        ("Previous 3 Months Ratings", ["rating_count_lag3"]),
        ("Recent Average Rating", ["avg_rating_lag1"]),
        ("Previous Month Unique Users", ["unique_users_lag1"]),
        ("3-Month Unique Users", ["users_3m"]),
        ("Long-Term Momentum", ["rating_count_3m", "rating_count_6m"]),
    ]

    cols = st.columns(4)
    for index, (title, fields) in enumerate(signal_cards):
        values = [movie_row.get(field, np.nan) for field in fields]
        display = ", ".join([f"{field}: {format_metric(v)}" for field, v in zip(fields, values)])
        with cols[index % 4]:
            st.markdown(
                f"""
                <div style='background: rgba(15,23,42,0.9); border:1px solid rgba(255,255,255,0.05); border-radius:16px; padding:1rem; min-height:150px;'>
                    <div style='font-weight:700; color:#f4f4f5;'>{title}</div>
                    <div style='margin-top:0.8rem; color:#d4d4d8; line-height:1.7;'>{display}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_analytics(resources: dict[str, Any], movie_rows: pd.DataFrame, selected_title: str) -> None:
    st.markdown("---")
    st.markdown("## ANALYTICS")
    if movie_rows.empty:
        st.warning("No historical data available to visualize.")
        return

    plot_df = movie_rows.sort_values("month").copy()
    plot_df["month_label"] = plot_df["month"].dt.strftime("%Y-%m")

    fig_rating = px.line(
        plot_df,
        x="month_label",
        y="rating_count_lag1",
        title="Monthly Rating Activity",
        labels={"rating_count_lag1": "Ratings", "month_label": "Month"},
        color_discrete_sequence=["#ef4444"],
    )
    fig_rating.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_rating, use_container_width=True)

    plot_df["rolling_3m"] = plot_df["rating_count_lag1"].rolling(window=3, min_periods=1).mean()
    plot_df["rolling_6m"] = plot_df["rating_count_lag1"].rolling(window=6, min_periods=1).mean()
    fig_roll = px.line(
        plot_df,
        x="month_label",
        y=["rolling_3m", "rolling_6m"],
        title="Rolling Activity Trend",
        labels={"value": "Ratings", "variable": "Metric", "month_label": "Month"},
    )
    fig_roll.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_roll, use_container_width=True)

    fig_users = px.line(
        plot_df,
        x="month_label",
        y="unique_users_lag1",
        title="Unique User Activity",
        labels={"unique_users_lag1": "Unique Users", "month_label": "Month"},
        color_discrete_sequence=["#f59e0b"],
    )
    fig_users.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_users, use_container_width=True)

    fig_avg = px.line(
        plot_df,
        x="month_label",
        y="avg_rating_lag1",
        title="Average Rating Trend",
        labels={"avg_rating_lag1": "Average Rating", "month_label": "Month"},
        color_discrete_sequence=["#22c55e"],
    )
    fig_avg.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_avg, use_container_width=True)

    fig_pop = px.line(
        plot_df,
        x="month_label",
        y=["rating_count_lag1", "rating_count_3m", "rating_count_6m"],
        title="Historical Popularity Trend",
        labels={"value": "Rating Activity", "variable": "Metric", "month_label": "Month"},
    )
    fig_pop.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_pop, use_container_width=True)


def render_model_performance(resources: dict[str, Any]) -> None:
    st.markdown("---")
    st.markdown("## MODEL PERFORMANCE")
    baseline_package = resources["model_package"]
    enhanced_package = resources.get("enhanced_package")
    model_metrics = {
        "Baseline XGBoost": {
            "MAE": baseline_package.get("test_mae", np.nan),
            "RMSE": baseline_package.get("test_rmse", np.nan),
            "R²": baseline_package.get("test_r2", np.nan),
        }
    }
    if enhanced_package:
        enhanced_test = enhanced_package["metrics"]["test"]
        model_metrics["Enhanced XGBoost"] = {
            "MAE": enhanced_test["MAE"],
            "RMSE": enhanced_test["RMSE"],
            "R²": enhanced_test["R2"],
        }

    export_data = []
    for model_name, metrics in model_metrics.items():
        export_data.append({"Model": model_name, **metrics})

    st.write("Training period: 1996–2021")
    st.write("Validation period: 2022")
    st.write("Test period: 2023")
    st.write(
        "Deployed model: Enhanced XGBoost (historical engagement + movie metadata)"
        if enhanced_package
        else "Deployed model: Baseline XGBoost (historical engagement features)"
    )

    st.dataframe(pd.DataFrame(export_data), use_container_width=True)

    try:
        model_package = enhanced_package or baseline_package
        model = model_package["model"]
        importances = model.feature_importances_
        names = model_package.get("schema", {}).get("feature_columns", model_package.get("features", []))
        importance_df = pd.DataFrame({"feature": names, "importance": importances})
        importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
        fig = px.bar(
            importance_df,
            x="importance",
            y="feature",
            orientation="h",
            title="XGBoost Feature Importance",
            color="importance",
            color_continuous_scale="reds",
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Feature importance chart is unavailable: {exc}")


def render_about_page() -> None:
    st.markdown("---")
    st.markdown("## ABOUT / METHODOLOGY")
    st.write("STREAMIQ uses historical movie audience-rating activity to forecast future activity.")
    st.markdown(
        """
        MovieLens 32M\n↓\nMonthly aggregation\n↓\nHistorical feature engineering\n↓\nTemporal train/validation/test split\n↓\nBaseline\n↓\nRandom Forest\n↓\nXGBoost\n↓\nFuture popularity forecasting\n↓\nStreamlit application
        """
    )
    st.write("Dataset facts: 32,000,204 ratings; 200,948 users; 87,585 movies; data through October 2023.")
    st.write("This project uses MovieLens rating activity and not Netflix internal viewing data.")
    st.write("MovieLens provides audience/rating data. TMDB is used only for optional movie metadata and poster presentation.")
    st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB.")


def main() -> None:
    inject_styles()
    resources = load_runtime_resources()
    data = resources["data"]
    model = resources["model"]
    thresholds = resources["thresholds"]
    enhanced_package = resources.get("enhanced_package")
    enhanced_metadata = resources.get("enhanced_metadata")

    render_hero()
    title, movie_rows, _, selected_month = render_sidebar(resources)
    if title is None:
        st.warning("Invalid movie selection.")
        return

    month_value = pd.to_datetime(selected_month)
    selected_movie_rows = movie_rows[movie_rows["month"] == month_value]
    if selected_movie_rows.empty:
        st.warning("The selected movie-month record is not available in the dataset.")
        return

    movie_row = selected_movie_rows.iloc[0]
    metadata_row = None
    if enhanced_metadata is not None:
        matched_metadata = enhanced_metadata[enhanced_metadata["movieId"] == int(movie_row["movieId"])]
        if not matched_metadata.empty:
            metadata_row = matched_metadata.iloc[0]
    render_movie_header(title, int(movie_row["movieId"]), movie_rows, selected_month, metadata_row)
    render_metadata_panel(metadata_row, selected_month)

    render_forecast_section(movie_row, model, thresholds, enhanced_package, enhanced_metadata)
    render_analytics(resources, movie_rows, title)
    render_model_performance(resources)
    render_explainability(movie_row)
    render_about_page()


if __name__ == "__main__":
    main()
