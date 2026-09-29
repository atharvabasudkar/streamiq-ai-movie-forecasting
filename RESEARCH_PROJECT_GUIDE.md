# STREAMIQ Research Project Guide

## 1. Project title

**STREAMIQ: AI-Based Movie Popularity Forecasting Using Historical Audience Engagement and Movie Metadata**

## 2. Abstract

STREAMIQ is a machine-learning application that forecasts the expected audience activity of a movie in the following month. The system uses the official MovieLens 32M dataset as its audience-engagement source. It combines historical rating behavior with movie metadata and uses XGBoost regression to estimate future rating activity. The predicted value is converted into Low, Medium, High, or Very High popularity categories and presented through an interactive Streamlit dashboard.

The project is related to content-popularity prediction systems used by streaming platforms, but it does not use Netflix internal viewing, watch-time, or completion data. MovieLens supplies the audience-rating history, while TMDB is used only for optional metadata enrichment and poster presentation.

## 3. Problem statement

Movie popularity changes over time. A movie may receive a large number of ratings in one period and much less activity later. The goal of STREAMIQ is to use information available before an analysis month to estimate how many ratings the movie will receive in the next month.

The system solves a regression problem:

`future_popularity = log1p(future_rating_count)`

At display time, the prediction is converted back:

`predicted_rating_count = expm1(predicted_log_popularity)`

The result is also mapped to four popularity classes using the saved threshold artifact.

## 4. Data sources

### MovieLens 32M

MovieLens provides the primary audience-engagement data:

- 32,000,204 ratings
- 200,948 users
- 87,585 movies
- Monthly activity through October 2023 in the source workflow

The project uses:

- `ratings.csv` for historical rating activity and target construction
- `movies.csv` for titles and MovieLens genres
- `links.csv` for MovieLens-to-TMDB ID mapping
- The prepared monthly lookup for efficient Streamlit inference

The application does not load all 32 million ratings at every startup.

### TMDB

TMDB is an optional enrichment and presentation source. When a valid API key is configured, the offline training script can retrieve runtime, original language, and production country using the TMDB ID from `links.csv`. Poster retrieval is independent from the ML feature vector.

TMDB does not provide the MovieLens ratings, and poster images are never used as model inputs.

## 5. Feature engineering

### Baseline features

The preserved baseline model uses eight historical features:

1. `rating_count_lag1`
2. `rating_count_lag2`
3. `rating_count_lag3`
4. `avg_rating_lag1`
5. `unique_users_lag1`
6. `rating_count_3m`
7. `rating_count_6m`
8. `users_3m`

These features describe recent audience activity before the prediction month.

### Enhanced features

The enhanced model uses 33 features:

- The eight baseline audience features
- `release_year`
- `runtime`
- `genre_count`
- `movie_age`
- 14 multi-hot genre features
- Encoded language and country features

`movie_age` is calculated as:

`analysis_year - release_year`

The raw multi-genre string is not passed directly to XGBoost. Genres are converted into independent binary columns such as `genre__Comedy` and `genre__Drama`. The feature schema is persisted so training and inference use the same column order.

When metadata is unavailable, categorical values use `Unknown` and numeric values use training-time imputation. No metadata is fabricated.

## 6. Temporal methodology

This is a forecasting problem, so the project uses chronological partitions rather than a random split:

- Training: June 1996 to December 2021
- Validation: January 2022 to December 2022
- Test: January 2023 to September 2023

Historical lag and rolling features are calculated from information available before the prediction month. Future rating counts are used only as the target, never as input features.

## 7. Models and evaluation

Two XGBoost regressors are maintained:

### Baseline XGBoost

The original model artifact is preserved and is not overwritten.

### Enhanced XGBoost

The enhanced model adds metadata features to the eight historical engagement features. It is trained offline by `train_enhanced.py` and saved separately.

Measured test results:

| Model | MAE | RMSE | R2 |
| --- | ---: | ---: | ---: |
| Baseline XGBoost | 0.325515 | 0.412945 | 0.838416 |
| Enhanced XGBoost | 0.326341 | 0.409897 | 0.840792 |

The enhanced model improves RMSE and R2, while the baseline has slightly better MAE. The enhanced model is used in the current application because it performs better on two of the three primary test metrics. This trade-off must be reported rather than hidden.

## 8. Application workflow

1. Streamlit loads the saved model, feature schema, thresholds, monthly lookup, and metadata lookup.
2. The user selects a movie and a valid historical month.
3. The application retrieves the row's historical audience features.
4. The application joins metadata using `movieId`.
5. The persisted schema transforms the row into the exact 33-feature vector.
6. XGBoost predicts log future popularity.
7. The application applies `expm1` to show expected next-month ratings.
8. Saved thresholds produce the popularity outlook.
9. TMDB poster retrieval runs separately for presentation.
10. Historical charts and model-performance information are rendered.

Streamlit does not train a model at startup. Training is an offline operation.

## 9. User interface

The dashboard provides:

- Movie search and historical-month selection
- Poster and movie identification
- Forecast card with expected next-month rating activity
- Popularity outlook classification
- Movie metadata panel
- Historical activity charts
- Baseline-versus-enhanced model performance
- Feature-importance visualization
- Methodology, limitations, and TMDB attribution

## 10. Current metadata status

The current local `.env` file has an empty `TMDB_API_KEY`. Therefore, the current persisted metadata artifact has:

- Runtime known: 0 of 87,585
- Language known: 0 of 87,585
- Country known: 0 of 87,585

MovieLens genres and release years are still available. To claim actual TMDB metadata coverage, the key must be configured and `train_enhanced.py` must be rerun. No manual language or country entry should be added because that would introduce unverifiable training data.

## 11. Reproducibility commands

```powershell
D:\streamiq_venv\Scripts\activate
python -m pip install -r requirements.txt
python train_enhanced.py
python -m streamlit run app.py
```

The enhanced artifacts are currently written to `D:\streamiq_artifacts` because the C: drive is full. For deployment, copy or regenerate the artifacts inside the repository under `models/` and `data/processed/`.

## 12. GitHub and Streamlit deployment

The current folder is not yet a Git repository. A safe deployment sequence is:

1. Create an empty GitHub repository named `STREAMIQ` under the user's GitHub account.
2. Move the required runtime artifacts into repository paths:
   - `models/streamiq_xgboost_enhanced.pkl`
   - `models/streamiq_feature_schema.json`
   - `models/streamiq_model_comparison.json`
   - `data/processed/movie_metadata_enriched.csv`
   - The existing baseline and monthly inference artifacts
3. Because the monthly CSV is larger than 100 MB, use Git LFS or convert the inference lookup to a compressed/columnar artifact before pushing.
4. Confirm `.env` is ignored and never commit the TMDB key.
5. Initialize Git and push from the project folder:

```powershell
git init
git add .
git commit -m "Prepare STREAMIQ for deployment"
git branch -M main
git remote add origin https://github.com/atharvabasudkar/STREAMIQ.git
git push -u origin main
```

6. In Streamlit Community Cloud, select the repository, branch `main`, and file `app.py`.
7. Add `TMDB_API_KEY` through Streamlit Cloud Secrets, not source code.
8. Deploy and verify the application after restart.

GitHub authentication must be performed by the account owner through Git Credential Manager or a personal access token. Passwords and tokens should not be shared in chat.

## 13. Limitations

- MovieLens rating activity is a proxy for audience engagement, not Netflix viewing behavior.
- The current local enhanced artifact was trained without TMDB runtime/language/country enrichment because no TMDB key was configured.
- Popularity prediction is historical forecasting, not a causal explanation of why a movie becomes popular.
- Predictions are only valid for movie-month records with sufficient historical features.
- TMDB availability, rate limits, and API coverage affect optional metadata and posters.
- The enhanced model improves RMSE and R2 but not MAE, so its selection has a documented metric trade-off.
