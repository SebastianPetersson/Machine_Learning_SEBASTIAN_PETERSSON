import pandas as pd
from sklearn.neighbors import NearestNeighbors
from preprocessing import clean_text, filter_ratings, build_item_user_matrix
from pathlib import Path
from joblib import dump, load

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

KNN_ARTIFACTS_PATH = CACHE_DIR / "knn_artifacts.joblib"


def build_knn_model(min_user_ratings=50, min_movie_ratings=50):
    """Build a KNN model and collaborative filtering artifacts from ratings data."""
    ratings_filtered = filter_ratings(min_user_ratings, min_movie_ratings)

    item_user_matrix, movie_to_idx, _ = build_item_user_matrix(ratings_filtered)
    idx_to_movie = {idx: movie_id for movie_id, idx in movie_to_idx.items()}

    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(item_user_matrix)
    
    return model, item_user_matrix, movie_to_idx, idx_to_movie

def resolve_movie_query(movie_query, movies):
    """Return movies whose normalized titles contain the query string."""
    clean_query = clean_text(movie_query)

    movie_match = movies[
        movies["title_clean"].str.contains(clean_query, na=False)
    ]

    return movie_match

def get_collaborative_candidates(movie_id, item_user_matrix, movie_to_idx, idx_to_movie, model, n=30):
    """Return the top-N collaborative filtering candidate movies for a given movie ID."""
    if movie_id not in movie_to_idx:
        return pd.DataFrame(columns = ["movieId", "collaborative_score"])
    
    movie_idx = movie_to_idx[movie_id]

    n_neighbors = min(n + 1, item_user_matrix.shape[0])
    distances, indices = model.kneighbors(
        item_user_matrix[movie_idx],
        n_neighbors = n_neighbors
    )

    neighbor_indices = indices.flatten()[1:]
    neighbor_distances = distances.flatten()[1:]
    recommended_movie_ids = [idx_to_movie[idx] for idx in neighbor_indices]

    candidates = pd.DataFrame({
        "movieId" : recommended_movie_ids,
        "collaborative_score" : 1 - neighbor_distances
    })

    return candidates

def load_or_build_knn_artifacts(min_user_ratings=50, min_movie_ratings=50, force_rebuild=False):
    """Load saved KNN artifacts or build and save them."""
    if KNN_ARTIFACTS_PATH.exists() and not force_rebuild:
        artifacts = load(KNN_ARTIFACTS_PATH)
        return (
            artifacts["knn_model"],
            artifacts["item_user_matrix"],
            artifacts["movie_to_idx"],
            artifacts["idx_to_movie"],
        )

    knn_model, item_user_matrix, movie_to_idx, idx_to_movie = build_knn_model(
        min_user_ratings=min_user_ratings,
        min_movie_ratings=min_movie_ratings,
    )

    dump({
        "knn_model": knn_model,
        "item_user_matrix": item_user_matrix,
        "movie_to_idx": movie_to_idx,
        "idx_to_movie": idx_to_movie,
    }, KNN_ARTIFACTS_PATH)

    return knn_model, item_user_matrix, movie_to_idx, idx_to_movie
