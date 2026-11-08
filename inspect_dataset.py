import pandas as pd

# Load dataset
df = pd.read_csv("dataset/tmdb_5000_movies.csv")

# Display first 5 rows
print("\nFIRST FIVE RECORDS")
print(df.head())

# Dataset dimensions
print("\nDATASET SHAPE")
print(df.shape)

# Column information
print("\nDATASET INFORMATION")
df.info()

# Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Duplicate records
print("\nDUPLICATES")
print(df.duplicated().sum())