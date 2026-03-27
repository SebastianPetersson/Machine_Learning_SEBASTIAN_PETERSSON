import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from content_model import get_content_candidates
from collaborative_model import get_collaborative_candidates

POPULARITY_GROUP_SCORES = {
    "low": 0.0,
    "medium": 1.0 / 3.0,
    "high": 2.0 / 3.0,
    "very_high": 1.0,
}

def normalize_scores(series):
    """Normalize a score series to the range 0-1. Uses MinMaxScaler for scaling."""
    if series.empty:
        return pd.Series(index=series.index, dtype=float)

    scaler = MinMaxScaler()
    numeric_series = pd.to_numeric(series, errors="coerce").fillna(0)
    values = numeric_series.to_numpy().reshape(-1, 1)
    scaled = scaler.fit_transform(values).flatten()
    return pd.Series(scaled, index=series.index)

def build_candidate_table(movie_id,
                          hybrid_features,
                          tfidf_matrix,
                          movie_to_row_idx,
                          item_user_matrix,
                          movie_to_idx,
                          idx_to_movie,
                          knn_model):
    """Build a merged candidate table from content-based and collaborative recommendations."""
    
    content_candidates = get_content_candidates(movie_id, hybrid_features, tfidf_matrix, movie_to_row_idx, n=20)
    collaborative_candidates = get_collaborative_candidates(movie_id, item_user_matrix, movie_to_idx, idx_to_movie, knn_model, n=20)
    
    candidates = pd.merge(content_candidates, collaborative_candidates, on = "movieId", how = "outer")

    candidates["content_score"] = candidates["content_score"].fillna(0)
    candidates["collaborative_score"] = candidates["collaborative_score"].fillna(0)

    candidates = pd.merge(candidates, hybrid_features, on = "movieId", how = "left")

    return candidates

def rank_candidates(candidates,
                    collab_weight = 0.4,
                    content_weight = 0.35,
                    rating_weight = 0.1,
                    expert_weight = 0.1,
                    new_expert_weight = 0.05,
                    popularity_weight = 0.1):
    """Rank recommendation candidates using weighted hybrid scoring."""
    
    candidates = candidates.copy()

    candidates["is_new_movie"] = (candidates["release_year"] >= 2020).astype(int)

    candidates["content_score_norm"] = normalize_scores(candidates["content_score"])
    candidates["collaborative_score_norm"] = normalize_scores(candidates["collaborative_score"])
    candidates["average_rating_norm"] = normalize_scores(candidates["average_rating"])
    candidates["rating_count_norm"] = normalize_scores(candidates["rating_count"])
    candidates["expert_mean_rating_norm"] = normalize_scores(candidates["expert_mean_rating"])
    candidates["new_expert_score"] = (candidates["is_new_movie"] * candidates["expert_mean_rating_norm"])
    candidates["popularity_score"] = (
        candidates["popularity_group"]
        .map(POPULARITY_GROUP_SCORES)
        .fillna(POPULARITY_GROUP_SCORES["medium"])
    )

    candidates["final_score"] = (
    collab_weight * candidates["collaborative_score_norm"] + 
    content_weight * candidates["content_score_norm"] + 
    rating_weight * candidates["average_rating_norm"] +
    expert_weight * candidates["expert_mean_rating_norm"] +
    new_expert_weight * candidates["new_expert_score"] +
    popularity_weight * candidates["popularity_score"]
    )

    candidates = candidates.sort_values("final_score", ascending=False)

    return candidates

def recommend_hybrid_by_movie_id(movie_id,
                                 hybrid_features,
                                 tfidf_matrix,
                                 item_user_matrix,
                                 movie_to_idx,
                                 movie_to_row_idx,
                                 idx_to_movie,
                                 knn_model,
                                 n=5,
                                 collab_weight=0.4,
                                 content_weight=0.35,
                                 rating_weight=0.1,
                                 expert_weight=0.1,
                                 new_expert_weight=0.05,
                                 popularity_weight=0.1):
    """Return top-N hybrid movie recommendations for a given movie ID."""

    candidates = build_candidate_table(
        movie_id,
        hybrid_features,
        tfidf_matrix,
        movie_to_row_idx,
        item_user_matrix,
        movie_to_idx,
        idx_to_movie,
        knn_model
    )

    if candidates.empty:
        return candidates

    ranked = rank_candidates(candidates,
                             collab_weight=collab_weight,
                             content_weight=content_weight,
                             rating_weight=rating_weight,
                             expert_weight=expert_weight,
                             new_expert_weight=new_expert_weight,
                             popularity_weight=popularity_weight)

    recommendations = ranked[ranked["movieId"] != movie_id].head(n)

    return recommendations
