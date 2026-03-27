from functools import lru_cache
import pandas as pd
from preprocessing import load_or_build_hybrid_feature_table
from content_model import load_or_build_tfidf_artifacts, resolve_movie_query
from collaborative_model import load_or_build_knn_artifacts
from hybrid_model import recommend_hybrid_by_movie_id
from poster_utils import get_poster_url

FINAL_MODEL_WEIGHTS = {
    "collab_weight": 0.4,
    "content_weight": 0.35,
    "rating_weight": 0.1,
    "expert_weight": 0.1,
    "new_expert_weight": 0.05,
    "popularity_weight": 0.1,
}

@lru_cache(maxsize=1)
def load_search_artifacts():
    """Load and cache the data needed for movie search. Only loads hybrid features for quicker initial search functionality."""
    hybrid_features = load_or_build_hybrid_feature_table()

    return {
        "hybrid_features": hybrid_features
    }

@lru_cache(maxsize=1)
def load_recommendation_artifacts():
    """Load and cache the artifacts needed for movie recommendations."""
    hybrid_features = load_or_build_hybrid_feature_table()
    _, tfidf_matrix, movie_to_row_idx = load_or_build_tfidf_artifacts(hybrid_features)
    knn_model, item_user_matrix, movie_to_idx, idx_to_movie = load_or_build_knn_artifacts()

    return {
        "hybrid_features" : hybrid_features,
        "tfidf_matrix" : tfidf_matrix,
        "knn_model" : knn_model,
        "item_user_matrix" : item_user_matrix,
        "movie_to_idx" : movie_to_idx,
        "movie_to_row_idx" : movie_to_row_idx,
        "idx_to_movie" : idx_to_movie
    }

def search_movies(query, limit=20):
    """Search fro movies matching a text query."""
    query = str(query).strip()
    if not query:
        return pd.DataFrame(columns=["movieId", "title", "genres", "release_year", "tmdbId"])
    
    artifacts = load_search_artifacts()
    hybrid_features = artifacts["hybrid_features"]

    matches = resolve_movie_query(query, hybrid_features).copy()

    if matches.empty:
        return pd.DataFrame(columns=["movieId", "title", "genres", "release_year", "tmdbId"])
    
    cols = ["movieId", "title", "genres", "release_year", "tmdbId"]
    available_cols = [col for col in cols if col in matches.columns]

    return matches[available_cols].head(limit).reset_index(drop=True)

def get_dropdown_options(query, limit=10):
    """Build dropdown options from the movie search results."""
    matches = search_movies(query, limit=limit)

    options = []
    for _, row in matches.iterrows():
        title = row["title"]
        genres = row["genres"] if pd.notna(row.get("genres")) else ""
        movie_id = int(row["movieId"])

        label = f"{title} | {genres}" if genres else title
        options.append({
            "label" : label,
            "value" : movie_id
        })

    return options
    
def recommend_for_movie_id(movie_id, n=5):
    """Return top-N movie recommendations for a given movie ID."""
    artifacts = load_recommendation_artifacts()

    recommendations = recommend_hybrid_by_movie_id(
        movie_id=movie_id,
        hybrid_features=artifacts["hybrid_features"],
        tfidf_matrix=artifacts["tfidf_matrix"],
        item_user_matrix=artifacts["item_user_matrix"],
        movie_to_idx=artifacts["movie_to_idx"],
        movie_to_row_idx=artifacts["movie_to_row_idx"],
        idx_to_movie=artifacts["idx_to_movie"],
        knn_model=artifacts["knn_model"],
        n=n,
        **FINAL_MODEL_WEIGHTS,
    ).copy()
        
    if "tmdbId" in recommendations.columns:
        recommendations["poster_url"] = recommendations["tmdbId"].apply(get_poster_url)
    else:
        recommendations["poster_url"] = None

    cols = [
        "movieId",
        "title",
        "genres",
        "release_year",
        "tmdbId",
        "poster_url",
        "final_score",
        "average_rating",
        "expert_mean_rating"
    ]
    available_cols = [col for col in cols if col in recommendations.columns]

    return recommendations[available_cols].reset_index(drop=True)
    
