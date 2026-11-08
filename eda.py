"""Beginner-friendly exploratory data analysis for MovieSuccess AI."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_FOLDER = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_FOLDER / "dataset" / "cleaned_movies.csv"
PLOTS_FOLDER = PROJECT_FOLDER / "plots"
MIN_GENRE_SAMPLE_SIZE = 20


def save_plot(file_name: str) -> None:
    """Save the current chart and close it to release memory."""
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / file_name, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Load the cleaned data, print findings, and save EDA charts."""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Cleaned dataset was not found: {DATASET_PATH}")

    movies = pd.read_csv(DATASET_PATH)
    required_columns = {
        "title",
        "budget",
        "runtime",
        "genre",
        "original_language",
        "release_year",
        "release_month",
        "success",
    }
    missing_columns = required_columns.difference(movies.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"The cleaned dataset is missing columns: {missing}")
    if "revenue" in movies.columns:
        raise ValueError(
            "Revenue must not be included in the cleaned dataset or EDA inputs."
        )

    PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    print("\n===== DATASET OVERVIEW =====")
    print(f"Dataset shape: {movies.shape}")
    print("\nColumn information:")
    movies.info()
    print("\nDescriptive statistics:")
    print(movies.describe(include="all").transpose().to_string())

    print("\n===== DATA QUALITY CHECKS =====")
    print("Missing values by column:")
    print(movies.isna().sum().to_string())
    print(f"\nDuplicate records: {movies.duplicated().sum():,}")

    print("\n===== UNIQUE VALUES =====")
    print(f"Unique genres ({movies['genre'].nunique()}):")
    print(", ".join(sorted(movies["genre"].dropna().unique())))
    print(f"\nUnique original languages: {movies['original_language'].nunique()}")
    print(", ".join(sorted(movies["original_language"].dropna().unique())))

    # 1. Compare the number of HIT and FLOP movies.
    class_labels = movies["success"].map({1: "HIT", 0: "FLOP"})
    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=pd.DataFrame({"result": class_labels}),
        x="result",
        order=["HIT", "FLOP"],
        hue="result",
        palette={"HIT": "#2ca02c", "FLOP": "#d62728"},
        legend=False,
    )
    plt.title("Number of HIT and FLOP Movies")
    plt.xlabel("Movie result")
    plt.ylabel("Number of movies")
    save_plot("hit_vs_flop_count.png")

    # 2. Show how movie budgets are distributed.
    plt.figure(figsize=(9, 5))
    sns.histplot(data=movies, x="budget", bins=30, color="#4472C4")
    plt.title("Movie Budget Distribution")
    plt.xlabel("Budget")
    plt.ylabel("Number of movies")
    save_plot("budget_distribution.png")

    # 3. Show the distribution of movie runtimes.
    plt.figure(figsize=(9, 5))
    sns.histplot(data=movies, x="runtime", bins=30, color="#ED7D31")
    plt.title("Movie Runtime Distribution")
    plt.xlabel("Runtime (minutes)")
    plt.ylabel("Number of movies")
    save_plot("runtime_distribution.png")

    # 4. Plot the ten most common first-genre values.
    top_genres = movies["genre"].value_counts().head(10).sort_values()
    plt.figure(figsize=(9, 6))
    sns.barplot(
        x=top_genres.values,
        y=top_genres.index,
        hue=top_genres.index,
        palette="viridis",
        legend=False,
    )
    plt.title("Top 10 Movie Genres")
    plt.xlabel("Number of movies")
    plt.ylabel("First genre")
    save_plot("top_10_genres.png")

    # 5. Compare genre success rates only when at least 20 movies are available.
    genre_summary = (
        movies.groupby("genre", dropna=False)
        .agg(movie_count=("success", "size"), hit_percentage=("success", "mean"))
        .assign(hit_percentage=lambda data: data["hit_percentage"] * 100)
    )
    eligible_genres = genre_summary[
        genre_summary["movie_count"] >= MIN_GENRE_SAMPLE_SIZE
    ].sort_values("hit_percentage")
    plt.figure(figsize=(10, 7))
    sns.barplot(
        data=eligible_genres,
        x="hit_percentage",
        y=eligible_genres.index,
        hue=eligible_genres.index,
        palette="crest",
        legend=False,
    )
    plt.title(
        f"Genre-wise HIT Percentage (Minimum {MIN_GENRE_SAMPLE_SIZE} Movies)"
    )
    plt.xlabel("HIT percentage")
    plt.ylabel("First genre")
    plt.xlim(0, 100)
    save_plot("genre_hit_percentage.png")

    # 6. Display how many movies were released in each year.
    yearly_counts = movies["release_year"].value_counts().sort_index()
    plt.figure(figsize=(11, 5))
    sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, marker="o")
    plt.title("Movie Release Year Distribution")
    plt.xlabel("Release year")
    plt.ylabel("Number of movies")
    save_plot("release_year_distribution.png")

    # 7. Check relationships between numeric inputs and the target.
    numerical_columns = [
        "budget",
        "runtime",
        "release_year",
        "release_month",
        "success",
    ]
    correlation_matrix = movies[numerical_columns].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
    )
    plt.title("Correlation Heatmap of Numerical Features")
    save_plot("correlation_heatmap.png")

    print("\n===== COMPARISON BY CLASS =====")
    class_averages = movies.assign(result=class_labels).groupby("result")[
        ["budget", "runtime"]
    ].mean()
    print("Average budget:")
    print(class_averages["budget"].reindex(["HIT", "FLOP"]).to_string())
    print("\nAverage runtime:")
    print(class_averages["runtime"].reindex(["HIT", "FLOP"]).to_string())

    highest_genre = eligible_genres["hit_percentage"].idxmax()
    highest_genre_percentage = eligible_genres.loc[
        highest_genre, "hit_percentage"
    ]
    highest_genre_count = int(eligible_genres.loc[highest_genre, "movie_count"])
    print("\n===== GENRE SUCCESS SUMMARY =====")
    print(
        f"Highest HIT percentage: {highest_genre} "
        f"({highest_genre_percentage:.2f}%, "
        f"{highest_genre_count} movies)"
    )
    print(
        f"Only genres with at least {MIN_GENRE_SAMPLE_SIZE} movies were compared."
    )

    overall_percentages = (
        class_labels.value_counts(normalize=True)
        .mul(100)
        .reindex(["HIT", "FLOP"], fill_value=0)
    )
    print("\n===== OVERALL CLASS DISTRIBUTION =====")
    print(overall_percentages.rename("percentage").round(2).to_string())
    print(f"\nSaved charts to: {PLOTS_FOLDER}")


if __name__ == "__main__":
    main()
