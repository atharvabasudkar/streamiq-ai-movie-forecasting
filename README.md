# STREAMIQ

## AI-Based Movie Popularity Forecasting

STREAMIQ forecasts next-month movie audience activity using historical MovieLens engagement signals and an enhanced XGBoost model with movie metadata. It provides an interactive Streamlit dashboard for movie selection, forecasting, historical analytics, model comparison, feature importance, and metadata inspection.

The project uses MovieLens rating activity, not Netflix internal viewing data. TMDB is optional and is used only for movie metadata enrichment and poster presentation. The model predicts `log1p(future_rating_count)` and displays the inverse-transformed expected rating count with a Low, Medium, High, or Very High outlook.

## Run locally

```powershell
D:\streamiq_venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The detailed research and deployment guide is in [RESEARCH_PROJECT_GUIDE.md](RESEARCH_PROJECT_GUIDE.md).

## Offline training

Training is separate from Streamlit startup:

```powershell
python train_enhanced.py
```

Keep `TMDB_API_KEY` in `.env` or Streamlit Secrets. Never commit `.env`.
