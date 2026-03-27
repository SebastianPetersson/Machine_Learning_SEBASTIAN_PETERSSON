import numpy as np
import pandas as pd
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
HYBRID_MODEL_DIR = CURRENT_DIR.parent
if str(HYBRID_MODEL_DIR) not in sys.path:
    sys.path.append(str(HYBRID_MODEL_DIR))

from data_loader import load_ratings
from preprocessing import load_or_build_hybrid_feature_table
from content_model import load_or_build_tfidf_artifacts
from collaborative_model import load_or_build_knn_artifacts
from hybrid_model import recommend_hybrid_by_movie_id


def build_eval_artifacts():
    """Load the artifacts and ratings needed for evaluation."""

    hybrid_features = load_or_build_hybrid_feature_table()
    _, tfidf_matrix, movie_to_row_idx = load_or_build_tfidf_artifacts(hybrid_features)
    knn_model, item_user_matrix, movie_to_idx, idx_to_movie = load_or_build_knn_artifacts()
    ratings = load_ratings()

    return {
        "hybrid_features": hybrid_features,
        "tfidf_matrix": tfidf_matrix,
        "movie_to_row_idx": movie_to_row_idx,
        "knn_model": knn_model,
        "item_user_matrix": item_user_matrix,
        "movie_to_idx": movie_to_idx,
        "idx_to_movie": idx_to_movie,
        "ratings": ratings,
    }


def create_test_cases(ratings, movie_to_idx, min_positive_movies=5, n_users=50, seed=42):
    """Create evaluation test cases from users with enough positive ratings."""
    positive_ratings = ratings[ratings["rating"] >= 4.0].copy()

    user_counts = positive_ratings.groupby("userId")["movieId"].nunique()
    eligible_users = user_counts[user_counts >= min_positive_movies].index.to_numpy()

    rng = np.random.default_rng(seed)
    if len(eligible_users) > n_users:
        eligible_users = rng.choice(eligible_users, size=n_users, replace=False)

    valid_movie_ids = set(movie_to_idx.keys())
    test_cases = []

    for user_id in eligible_users:
        liked_movies = (
            positive_ratings.loc[positive_ratings["userId"] == user_id, "movieId"]
            .drop_duplicates()
            .tolist()
        )
        liked_movies = [movie_id for movie_id in liked_movies if movie_id in valid_movie_ids]

        if len(liked_movies) < min_positive_movies:
            continue

        query_movie = int(rng.choice(liked_movies))
        relevant_movies = {movie_id for movie_id in liked_movies if movie_id != query_movie}

        if not relevant_movies:
            continue

        test_cases.append({
            "userId": int(user_id),
            "query_movie": query_movie,
            "relevant_movies": relevant_movies,
        })

    return pd.DataFrame(test_cases)


def dcg_at_k(relevances, k=5):
    """Computes DCG at rank k from a list of relevance scores."""
    relevances = np.asarray(relevances, dtype=float)[:k]
    if len(relevances) == 0:
        return 0.0

    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(relevances / discounts))


def ndcg_at_k(recommended_ids, relevant_ids, k=5):
    """Compute NDCG at rank k for recommended and relevant movie IDs."""
    relevances = [1.0 if movie_id in relevant_ids else 0.0 for movie_id in recommended_ids[:k]]
    dcg = dcg_at_k(relevances, k=k)

    ideal_len = min(len(relevant_ids), k)
    if ideal_len == 0:
        return 0.0

    ideal_relevances = [1.0] * ideal_len
    idcg = dcg_at_k(ideal_relevances, k=k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_ndcg_at_5(test_cases,
                       hybrid_features,
                       tfidf_matrix,
                       movie_to_row_idx,
                       item_user_matrix,
                       movie_to_idx,
                       idx_to_movie,
                       knn_model,
                       collab_weight=0.4,
                       content_weight=0.35,
                       rating_weight=0.1,
                       expert_weight=0.1,
                       new_expert_weight=0.05,
                       popularity_weight=0.1,
                       progress_every=10):
    """Evaluate the hybrid recommender using mean NDCG@5."""
    scores = []

    for i, (_, row) in enumerate(test_cases.iterrows(), start=1):
        if progress_every and i % progress_every == 0:
            print(f"Finished with {i}/{len(test_cases)} test cases.")

        recommendations = recommend_hybrid_by_movie_id(
            movie_id=row["query_movie"],
            hybrid_features=hybrid_features,
            tfidf_matrix=tfidf_matrix,
            movie_to_row_idx=movie_to_row_idx,
            item_user_matrix=item_user_matrix,
            movie_to_idx=movie_to_idx,
            idx_to_movie=idx_to_movie,
            knn_model=knn_model,
            n=5,
            collab_weight=collab_weight,
            content_weight=content_weight,
            rating_weight=rating_weight,
            expert_weight=expert_weight,
            new_expert_weight=new_expert_weight,
            popularity_weight=popularity_weight,
        )

        recommended_ids = recommendations["movieId"].tolist()
        score = ndcg_at_k(recommended_ids, row["relevant_movies"], k=5)
        scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


def compare_weight_configs(configs, test_cases, artifacts):
    """Compare multiple weight configurations using the NDCG@5 function."""
    results = []

    for config in configs:
        print(f"Running config: {config['name']}")
        score = evaluate_ndcg_at_5(
            test_cases=test_cases,
            hybrid_features=artifacts["hybrid_features"],
            tfidf_matrix=artifacts["tfidf_matrix"],
            movie_to_row_idx=artifacts["movie_to_row_idx"],
            item_user_matrix=artifacts["item_user_matrix"],
            movie_to_idx=artifacts["movie_to_idx"],
            idx_to_movie=artifacts["idx_to_movie"],
            knn_model=artifacts["knn_model"],
            collab_weight=config["collab_weight"],
            content_weight=config["content_weight"],
            rating_weight=config["rating_weight"],
            expert_weight=config["expert_weight"],
            new_expert_weight=config["new_expert_weight"],
            popularity_weight=config.get("popularity_weight", 0.0),
            progress_every=config.get("progress_every", 10),
        )

        results.append({
            **config,
            "popularity_weight": config.get("popularity_weight", 0.0),
            "ndcg_at_5": score,
        })

    return pd.DataFrame(results).sort_values("ndcg_at_5", ascending=False).reset_index(drop=True)
