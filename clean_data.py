"""Create a cleaned, model-ready movie dataset from the TMDB CSV."""

import json
from pathlib import Path

import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_FOLDER / "dataset" / "tmdb_5000_movies.csv"
OUTPUT_PATH = PROJECT_FOLDER / "dataset" / "cleaned_movies.csv"

FINAL_COLUMNS = [
    "title",
    "budget",
    "runtime",
    "genre",
    "original_language",
    "release_year",
    "release_month",
    "success",
]


def extract_first_genre(genres_value: object) -> str:
    """Return the first genre name from a JSON-formatted genres value."""
    if pd.isna(genres_value):
        return "Unknown"

    try:
        genres = json.loads(str(genres_value))
    except (TypeError, json.JSONDecodeError):
        return "Unknown"

    if not isinstance(genres, list) or not genres:
        return "Unknown"

    first_genre = genres[0]
    if isinstance(first_genre, dict):
        genre_name = first_genre.get("name")
        if isinstance(genre_name, str) and genre_name.strip():
            return genre_name.strip()

    return "Unknown"


def main() -> None:
    """Load, clean, label, and save the movie dataset."""
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Original dataset was not found: {INPUT_PATH}")

    try:
        movies = pd.read_csv(INPUT_PATH)
    except pd.errors.ParserError as error:
        raise ValueError(f"Could not parse the CSV file: {INPUT_PATH}") from error

    required_columns = {
        "budget",
        "revenue",
        "status",
        "release_date",
        "runtime",
        "genres",
        "title",
        "original_language",
    }
    missing_columns = required_columns.difference(movies.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"The dataset is missing required columns: {missing}")

    # Work on a copy so the original DataFrame and CSV remain unchanged.
    cleaned_movies = movies.loc[
        (movies["budget"] > 0)
        & (movies["revenue"] > 0)
        & (movies["status"] == "Released")
    ].copy()

    cleaned_movies["release_date"] = pd.to_datetime(
        cleaned_movies["release_date"], errors="coerce"
    )
    cleaned_movies = cleaned_movies.loc[
        cleaned_movies["release_date"].notna()
    ].copy()

    # Treat zero runtime as missing, then fill missing values with the candidate median.
    cleaned_movies["runtime"] = cleaned_movies["runtime"].mask(
        cleaned_movies["runtime"] <= 0
    )
    runtime_median = cleaned_movies["runtime"].median()
    if pd.isna(runtime_median):
        raise ValueError("No valid runtime values are available for imputation.")
    cleaned_movies["runtime"] = cleaned_movies["runtime"].fillna(runtime_median)

    cleaned_movies["genre"] = cleaned_movies["genres"].apply(extract_first_genre)
    cleaned_movies["release_year"] = cleaned_movies["release_date"].dt.year
    cleaned_movies["release_month"] = cleaned_movies["release_date"].dt.month

    # Revenue is used only here to create the target and is excluded from the output.
    cleaned_movies["success"] = (
        cleaned_movies["revenue"] >= 2 * cleaned_movies["budget"]
    ).astype(int)

    cleaned_movies = cleaned_movies[FINAL_COLUMNS]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_movies.to_csv(OUTPUT_PATH, index=False)

    print("\n===== CLEANING COMPLETE =====")
    print(f"Original rows: {len(movies):,}")
    print(f"Final dataset shape: {cleaned_movies.shape}")
    print(f"Saved cleaned dataset to: {OUTPUT_PATH}")
    print(f"\nRuntime median used for imputation: {runtime_median:.2f}")

    print("\n===== MISSING VALUES IN FINAL DATASET =====")
    print(cleaned_movies.isna().sum().to_string())

    print("\n===== HIT/FLOP DISTRIBUTION =====")
    class_counts = (
        cleaned_movies["success"]
        .map({1: "HIT", 0: "FLOP"})
        .value_counts()
        .reindex(["HIT", "FLOP"], fill_value=0)
    )
    print(class_counts.to_string())


if __name__ == "__main__":
    main()
