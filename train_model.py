"""Train, compare, evaluate, and save movie HIT/FLOP classification models."""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_FOLDER = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_FOLDER / "dataset" / "cleaned_movies.csv"
PLOTS_FOLDER = PROJECT_FOLDER / "plots"
MODEL_PATH = PROJECT_FOLDER / "movie_model.pkl"
RESULTS_PATH = PROJECT_FOLDER / "model_results.csv"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "budget",
    "runtime",
    "genre",
    "original_language",
    "release_year",
    "release_month",
]
NUMERICAL_FEATURES = [
    "budget",
    "runtime",
    "release_year",
    "release_month",
]
CATEGORICAL_FEATURES = ["genre", "original_language"]


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing steps that are fitted only within each training fold."""
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(classifier) -> Pipeline:
    """Combine the shared preprocessor and a classifier in one pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def save_model_comparison(results: pd.DataFrame) -> None:
    """Save and plot the cross-validation model comparison."""
    results.to_csv(RESULTS_PATH, index=False)
    plt.figure(figsize=(9, 5))
    chart = sns.barplot(
        data=results,
        x="model",
        y="mean_cv_f1",
        hue="model",
        palette="viridis",
        legend=False,
    )
    plt.title("Model Comparison Using Five-Fold Cross-Validation")
    plt.xlabel("Algorithm")
    plt.ylabel("Mean validation F1-score")
    plt.ylim(0, 1)
    for bar, score in zip(chart.patches, results["mean_cv_f1"]):
        chart.annotate(
            f"{score:.3f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
        )
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / "model_comparison.png", dpi=150)
    plt.close()


def main() -> None:
    """Train candidate models and save the best complete pipeline."""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Cleaned dataset was not found: {DATASET_PATH}")

    movies = pd.read_csv(DATASET_PATH)
    required_columns = set(FEATURE_COLUMNS + ["success"])
    missing_columns = required_columns.difference(movies.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"The dataset is missing required columns: {missing}")
    if "revenue" in movies.columns:
        raise ValueError("Revenue must never be used as an input feature.")

    X = movies[FEATURE_COLUMNS].copy()
    y = movies["success"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    cross_validator = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print("\n===== FIVE-FOLD CROSS-VALIDATION =====")
    cv_records = []
    successful_pipelines = {}
    for model_name, classifier in models.items():
        pipeline = build_pipeline(classifier)
        try:
            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=cross_validator,
                scoring="f1",
                error_score="raise",
            )
        except Exception as error:
            print(f"{model_name} failed during cross-validation: {error}")
            continue

        mean_score = scores.mean()
        std_score = scores.std()
        cv_records.append(
            {
                "model": model_name,
                "mean_cv_f1": mean_score,
                "std_cv_f1": std_score,
            }
        )
        successful_pipelines[model_name] = pipeline
        print(
            f"{model_name}: mean F1={mean_score:.4f}, "
            f"standard deviation={std_score:.4f}"
        )

    if not cv_records:
        raise RuntimeError("All classification algorithms failed during training.")

    results = pd.DataFrame(cv_records).sort_values(
        "mean_cv_f1", ascending=False
    )
    PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)
    save_model_comparison(results)

    best_model_name = results.iloc[0]["model"]
    best_pipeline = successful_pipelines[best_model_name]
    best_cv_score = float(results.iloc[0]["mean_cv_f1"])
    best_pipeline.fit(X_train, y_train)
    predictions = best_pipeline.predict(X_test)
    probabilities = best_pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    test_f1 = f1_score(y_test, predictions, zero_division=0)

    print("\n===== SELECTED MODEL =====")
    print(f"Algorithm: {best_model_name}")
    print(f"Mean cross-validation F1-score: {best_cv_score:.4f}")
    print("\n===== TEST SET METRICS =====")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {test_f1:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["FLOP", "HIT"],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y_test, predictions)
    print("Confusion matrix:")
    print(matrix)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["FLOP", "HIT"],
        yticklabels=["FLOP", "HIT"],
    )
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / "confusion_matrix.png", dpi=150)
    plt.close()

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.figure(figsize=(8, 6))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"{best_model_name} (AUC = {roc_auc:.3f})",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    plt.title(f"ROC Curve - {best_model_name}")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / "roc_curve.png", dpi=150)
    plt.close()

    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\nSaved model pipeline to: {MODEL_PATH}")
    print(f"Saved model comparison CSV to: {RESULTS_PATH}")
    print(f"Saved charts to: {PLOTS_FOLDER}")


if __name__ == "__main__":
    main()
