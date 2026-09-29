from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
LINKS_PATH = ROOT_DIR / "data" / "raw" / "ml-32m" / "links.csv"

load_dotenv(dotenv_path=ROOT_DIR / ".env")


@st.cache_data(show_spinner=False)
def get_external_movie_ids(movie_id: int) -> dict[str, str]:
    if not LINKS_PATH.exists():
        return {}
    try:
        links = pd.read_csv(
            LINKS_PATH,
            usecols=["movieId", "imdbId", "tmdbId"],
            dtype={"movieId": "int64", "imdbId": "string", "tmdbId": "string"},
        )
        match = links[links["movieId"] == movie_id]
        if match.empty:
            return {}
        row = match.iloc[0]
        return {key: str(row[key]).split(".", 1)[0] for key in ("imdbId", "tmdbId") if pd.notna(row[key])}
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=86400)
def get_tmdb_poster_url(title: str | None, movie_id: int | None = None, api_key: str | None = None) -> str | None:
    api_key = api_key or os.getenv("TMDB_API_KEY")
    if not title and movie_id is None:
        return None

    external_ids = get_external_movie_ids(movie_id) if movie_id is not None else {}
    if not api_key:
        tmdb_id = external_ids.get("tmdbId")
        if not tmdb_id:
            return None
        try:
            page = requests.get(
                f"https://www.themoviedb.org/movie/{tmdb_id}?language=en-US",
                timeout=10,
                headers={"User-Agent": "STREAMIQ/1.0"},
            )
            page.raise_for_status()
            match = re.search(r"https://image\.tmdb\.org/t/p/[^\"\\ ]+", page.text)
            return match.group(0) if match else None
        except Exception:
            return None

    base_url = "https://api.themoviedb.org/3"

    try:
        if movie_id is not None:
            response = requests.get(
                f"{base_url}/movie/{movie_id}",
                params={"api_key": api_key, "language": "en-US"},
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            poster_path = payload.get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
            return None

        response = requests.get(
            f"{base_url}/search/movie",
            params={"api_key": api_key, "query": title, "include_adult": False, "language": "en-US"},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results") or []
        if not results:
            return None

        poster_path = results[0].get("poster_path")
        if not poster_path:
            return None
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        return None
