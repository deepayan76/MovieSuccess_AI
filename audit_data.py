
import pandas as pd

df = pd.read_csv("dataset/tmdb_5000_movies.csv")

print("\n===== ZERO VALUE ANALYSIS =====")

print("Movies with zero budget:")
print((df["budget"] == 0).sum())

print("\nMovies with zero revenue:")
print((df["revenue"] == 0).sum())

print("\nMovies with zero runtime:")
print((df["runtime"] == 0).sum())

print("\n===== MOVIE STATUS =====")
print(df["status"].value_counts())

print("\n===== RELEASE DATE CHECK =====")

dates = pd.to_datetime(
    df["release_date"], errors="coerce"
)

print("Invalid release dates:", dates.isna().sum())

print("\n===== CANDIDATE DATASET =====")

# Keep records with valid financial information
candidate = df[
    (df["budget"] > 0) &
    (df["revenue"] > 0)
].copy()

candidate["success"] = (
    candidate["revenue"] >= 2 * candidate["budget"]
).astype(int)

print("Original records:", len(df))
print("Records with valid budget and revenue:", len(candidate))

print("\n===== HIT AND FLOP DISTRIBUTION =====")

print(candidate["success"].value_counts())

print("\nClass percentages:")
print(
    candidate["success"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)
