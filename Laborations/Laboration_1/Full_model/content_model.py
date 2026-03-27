import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from preprocessing import clean_text
from pathlib import Path
from joblib import dump, load

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

TFIDF_ARTIFACTS_PATH = CACHE_DIR / "tfidf_artifacts.joblib"


def build_tfidf_matrix(hybrid_features):
    """Uses TfidsVectorizer to build a tfidf_matrix, uses fit_transform on column 'text_features'."""
    tfidf = TfidfVectorizer(
        stop_words = "english",
        token_pattern = r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        ngram_range = (1, 2),
        min_df = 2,
        max_df = 0.8
    )

    tfidf_matrix = tfidf.fit_transform(hybrid_features["text_features"])

    return tfidf, tfidf_matrix

def resolve_movie_query(movie_query, hybrid_features):
    """Find movies whose cleaned titles match a text query."""

    clean_query = clean_text(movie_query)

    movie_match = hybrid_features[
        hybrid_features["title_clean"].str.contains(clean_query, na=False)
    ]

    return movie_match

def get_content_candidates(movie_id, hybrid_features, tfidf_matrix, movie_to_row_idx, n=30):
    """Return the top-N content-based candidate movies for a given movie ID."""

    movie_index = movie_to_row_idx.get(movie_id)

    if movie_index is None:
        return pd.DataFrame(columns = ["movieId", "content_score"])
    
    if hybrid_features.iloc[movie_index]["text_features"] == "unknown":
        return pd.DataFrame(columns=["movieId", "content_score"])
    
    movie_vector = tfidf_matrix[movie_index]

    similarity_scores = linear_kernel(tfidf_matrix, movie_vector).flatten()
    similarity_scores[movie_index] = -np.inf

    top_n = min(n, len(similarity_scores) - 1)
    if top_n <= 0:
        return pd.DataFrame(columns=["movieId", "content_score"])

    top_indices = np.argpartition(-similarity_scores, top_n)[:top_n]
    top_indices = top_indices[np.argsort(-similarity_scores[top_indices])]

    candidates = pd.DataFrame({
        "movieId": hybrid_features.iloc[top_indices]["movieId"].to_numpy(),
        "content_score": similarity_scores[top_indices]
    })

    return candidates

def load_or_build_tfidf_artifacts(hybrid_features, force_rebuild=False):
    """Load saved TF-IDF artifacts or build and save them if needed."""
    if TFIDF_ARTIFACTS_PATH.exists() and not force_rebuild:
        artifacts = load(TFIDF_ARTIFACTS_PATH)
        return (
            artifacts["tfidf"],
            artifacts["tfidf_matrix"],
            artifacts["movie_to_row_idx"]
        )
    
    tfidf, tfidf_matrix = build_tfidf_matrix(hybrid_features)
    movie_to_row_idx = build_movie_to_row_idx(hybrid_features)

    dump({
        "tfidf" : tfidf,
        "tfidf_matrix" : tfidf_matrix,
        "movie_to_row_idx" : movie_to_row_idx
    }, TFIDF_ARTIFACTS_PATH)

    return tfidf, tfidf_matrix, movie_to_row_idx

def build_movie_to_row_idx(hybrid_features):
    """Build a mapping from movie IDs to row indices in the feature table."""
    return {
        movie_id: row_idx
        for row_idx, movie_id in enumerate(hybrid_features["movieId"].tolist())
    }