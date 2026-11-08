"""Streamlit interface for the MovieSuccess AI educational classifier."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_FOLDER = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_FOLDER / "movie_model.pkl"
MODEL_FEATURES = [
    "budget",
    "runtime",
    "genre",
    "original_language",
    "release_year",
    "release_month",
]
TRAINING_YEAR_MIN = 1916
TRAINING_YEAR_MAX = 2016

GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "Foreign",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Science Fiction",
    "Thriller",
    "Unknown",
    "War",
    "Western",
]
LANGUAGES = [
    "af",
    "cn",
    "da",
    "de",
    "en",
    "es",
    "fa",
    "fr",
    "he",
    "hi",
    "id",
    "is",
    "it",
    "ja",
    "ko",
    "nb",
    "nl",
    "no",
    "pl",
    "pt",
    "ro",
    "ru",
    "te",
    "th",
    "vi",
    "xx",
    "zh",
]
MONTHS = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


@st.cache_resource
def load_model():
    """Load and cache the trained pipeline for the current Streamlit session."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file was not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise TypeError("The saved file is not a compatible classification pipeline.")
    return model


def display_prediction(prediction: int, hit_probability: float) -> None:
    """Display the result and explain that it is an estimate, not a guarantee."""
    label = "HIT" if prediction == 1 else "FLOP"
    probability_percent = hit_probability * 100

    if label == "HIT":
        st.success(f"### Prediction: {label}")
        explanation = (
            "Based on the entered details, the model estimates a higher likelihood "
            "of the movie being a HIT."
        )
    else:
        st.error(f"### Prediction: {label}")
        explanation = (
            "Based on the entered details, the model estimates a higher likelihood "
            "of the movie being a FLOP."
        )

    st.metric("Estimated HIT probability", f"{probability_percent:.1f}%")
    st.progress(
        int(round(probability_percent)),
        text=f"Model-estimated HIT probability: {probability_percent:.1f}%",
    )
    st.info(explanation)
    st.caption(
        "This is a model estimate, not statistically calibrated confidence, and "
        "it is not a guaranteed outcome."
    )


def main() -> None:
    """Render the MovieSuccess AI application."""
    st.set_page_config(
        page_title="MovieSuccess AI",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .hero {
            padding: 2rem;
            border-radius: 1rem;
            background: linear-gradient(120deg, #17153b, #5b247a, #1b7f9b);
            color: white;
            margin-bottom: 1.5rem;
        }
        .hero h1 { margin-bottom: 0.3rem; }
        .metric-card {
            padding: 1rem;
            border-radius: 0.8rem;
            background: #f4f6fb;
            border: 1px solid #e2e6f0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("About the project")
        st.write(
            "MovieSuccess AI is an educational binary classification system that "
            "estimates whether a movie may be a HIT or FLOP from six movie details."
        )
        st.divider()
        st.subheader("Historical test results")
        st.metric("Accuracy", "58.51%")
        st.metric("F1-score", "61.49%")
        st.caption(
            "These are historical results from the held-out test set used during "
            "model evaluation, not a guarantee for new movies."
        )
        st.divider()
        st.caption(
            "The model was trained on historical movies released up to 2016. "
            "Revenue, popularity, vote average, and vote count are not inputs."
        )

    st.markdown(
        """
        <div class="hero">
            <h1>🎬 MovieSuccess AI</h1>
            <p>Explore a machine-learning estimate of a movie's HIT or FLOP outcome.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Enter movie details")
    st.write("The prediction uses only the six features used during model training.")
    with st.form("movie_prediction_form"):
        first_row = st.columns(2)
        with first_row[0]:
            budget = st.number_input(
                "Budget (USD)",
                min_value=1.0,
                value=25_000_000.0,
                step=1_000_000.0,
                format="%.0f",
            )
        with first_row[1]:
            runtime = st.number_input(
                "Runtime (minutes)",
                min_value=1.0,
                value=110.0,
                step=1.0,
            )

        second_row = st.columns(2)
        with second_row[0]:
            genre = st.selectbox("Genre", GENRES, index=6)
        with second_row[1]:
            language = st.selectbox("Original language", LANGUAGES, index=4)

        third_row = st.columns(2)
        with third_row[0]:
            release_year = st.number_input(
                "Release year",
                min_value=1888,
                max_value=2100,
                value=2016,
                step=1,
            )
        with third_row[1]:
            release_month = st.selectbox(
                "Release month",
                options=list(MONTHS),
                format_func=lambda month: MONTHS[month],
                index=11,
            )

        submitted = st.form_submit_button(
            "🚀 Predict Movie Success",
            use_container_width=True,
            type="primary",
        )

    if release_year < TRAINING_YEAR_MIN or release_year > TRAINING_YEAR_MAX:
        st.warning(
            f"Release year {release_year} is outside the model's historical training "
            f"period ({TRAINING_YEAR_MIN}-{TRAINING_YEAR_MAX}). Interpret the result "
            "with extra caution."
        )

    if submitted:
        input_data = pd.DataFrame(
            [
                {
                    "budget": budget,
                    "runtime": runtime,
                    "genre": genre,
                    "original_language": language,
                    "release_year": int(release_year),
                    "release_month": int(release_month),
                }
            ],
            columns=MODEL_FEATURES,
        )
        try:
            model = load_model()
            prediction = int(model.predict(input_data)[0])
            probabilities = model.predict_proba(input_data)[0]
            class_positions = {
                int(class_label): position
                for position, class_label in enumerate(model.classes_)
            }
            hit_probability = float(probabilities[class_positions[1]])
        except FileNotFoundError as error:
            st.error(f"Unable to load the model: {error}")
        except (KeyError, TypeError, ValueError, OSError) as error:
            st.error(f"Prediction could not be completed: {error}")
        else:
            st.divider()
            st.subheader("Prediction result")
            display_prediction(prediction, hit_probability)

    st.divider()
    st.subheader("Model limitations")
    st.markdown(
        """
        - This is an educational model trained on historical TMDB data, not a
          guarantee of commercial success.
        - The training data covers movies released up to 2016, so later years may
          reflect changing audience and industry patterns.
        - The displayed probabilities are model estimates and are not statistically
          calibrated confidence values.
        - Important factors such as marketing, cast, reviews, competition, and
          audience trends are not represented by these six inputs.
        - Correlation or a model prediction does not prove causation.
        """
    )


if __name__ == "__main__":
    main()
