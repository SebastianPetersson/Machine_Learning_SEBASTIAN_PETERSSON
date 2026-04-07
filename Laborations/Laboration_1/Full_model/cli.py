import time
from pathlib import Path

from preprocessing import load_or_build_hybrid_feature_table
from content_model import (
    load_or_build_tfidf_artifacts,
    resolve_movie_query as resolve_content_query,
)
from collaborative_model import load_or_build_knn_artifacts
from hybrid_model import build_candidate_table, rank_candidates


DIAGNOSTICS_DIR = Path(__file__).resolve().parent / "diagnostics"
DIAGNOSTICS_DIR.mkdir(exist_ok=True)


def save_recommendation_diagnostics(recommendations, movie_query, chosen_title, top_n=10):
    cols = [
        "movieId",
        "title",
        "genres",
        "final_score",
        "content_score",
        "collaborative_score",
        "content_score_norm",
        "collaborative_score_norm",
        "average_rating",
        "average_rating_norm",
        "rating_count",
        "expert_mean_rating",
        "expert_mean_rating_norm",
        "new_expert_score",
        "popularity_group"
    ]

    available_cols = [col for col in cols if col in recommendations.columns]
    diagnostics = recommendations[available_cols].head(top_n).copy()
    diagnostics.insert(0, "query_movie", chosen_title)
    diagnostics.insert(1, "query_text", movie_query)

    safe_name = "".join(ch.lower() if ch.isalnum() else "_" for ch in chosen_title).strip("_")
    if not safe_name:
        safe_name = "movie"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = DIAGNOSTICS_DIR / f"{safe_name}_{timestamp}.csv"
    diagnostics.to_csv(output_path, index=False)

    return output_path


def show_recommendation_diagnostics(recommendations, top_n=10):
    cols = [
        "title",
        "genres",
        "final_score",
        "content_score",
        "collaborative_score",
        "content_score_norm",
        "collaborative_score_norm",
        "average_rating",
        "average_rating_norm",
        "rating_count",
        "expert_mean_rating",
        "expert_mean_rating_norm",
        "new_expert_score",
        "popularity_group"
    ]

    available_cols = [col for col in cols if col in recommendations.columns]
    print("\nDiagnostics for recommendations:")
    print(recommendations[available_cols].head(top_n).round(3).to_string())


def recommend_hybrid_movies(movie_query,
                            hybrid_features,
                            tfidf_matrix,
                            item_user_matrix,
                            movie_to_idx,
                            idx_to_movie,
                            knn_model,
                            n=5,
                            collab_weight=0.4,
                            content_weight=0.35,
                            rating_weight=0.1,
                            expert_weight=0.1,
                            new_expert_weight=0.05,
                            debug=True):

    movie_match = resolve_content_query(movie_query, hybrid_features)

    if movie_match.empty:
        print("Movie could not be found. Check the spelling or pick a different movie.")
        return

    if len(movie_match) > 1:
        print("Several movies matched your search:")
        movie_options = movie_match[["movieId", "title", "genres"]].reset_index(drop=True)
        movie_options.index = movie_options.index + 1
        print(movie_options[["title", "genres"]])

        choice = input("Select a movie by entering the number: ")

        try:
            choice = int(choice)
        except ValueError:
            print("Invalid input. Enter the number for the movie you want to choose.")
            return

        if choice < 1 or choice > len(movie_options):
            print("Invalid number.")
            return

        chosen_movie_id = movie_options.iloc[choice - 1]["movieId"]
        chosen_title = movie_options.iloc[choice - 1]["title"]
    else:
        chosen_movie_id = movie_match.iloc[0]["movieId"]
        chosen_title = movie_match.iloc[0]["title"]

    candidates = build_candidate_table(
        chosen_movie_id,
        hybrid_features,
        tfidf_matrix,
        item_user_matrix,
        movie_to_idx,
        idx_to_movie,
        knn_model,
    )

    if candidates.empty:
        print("This movie has too little metadata or rating history to generate reliable recommendations.")
        return

    ranked = rank_candidates(
        candidates,
        collab_weight=collab_weight,
        content_weight=content_weight,
        rating_weight=rating_weight,
        expert_weight=expert_weight,
        new_expert_weight=new_expert_weight
    )

    recommendations = ranked[ranked["movieId"] != chosen_movie_id].head(n)

    if debug:
        show_recommendation_diagnostics(ranked, top_n=10)
        diagnostics_path = save_recommendation_diagnostics(
            ranked,
            movie_query,
            chosen_title,
            top_n=10,
        )
        print(f"\nDiagnostics saved to: {diagnostics_path}")

    print(f"\nIf you like '{chosen_title}', maybe you would also like:")
    print(recommendations[["title", "genres", "final_score"]])

    return recommendations


def main():
    start = time.time()
    hybrid_features = load_or_build_hybrid_feature_table()
    print(f"load_or_build_hybrid_feature_table: {time.time() - start:.2f} s")

    start = time.time()
    _, tfidf_matrix = load_or_build_tfidf_artifacts(hybrid_features)
    print(f"load_or_build_tfidf_artifacts: {time.time() - start:.2f} s")

    start = time.time()
    knn_model, item_user_matrix, movie_to_idx, idx_to_movie = load_or_build_knn_artifacts()
    print(f"load_or_build_knn_artifacts: {time.time() - start:.2f} s")

    while True:
        movie_query = input("What movie would you like to watch? ").strip()

        start = time.time()
        recommend_hybrid_movies(
            movie_query,
            hybrid_features,
            tfidf_matrix,
            item_user_matrix,
            movie_to_idx,
            idx_to_movie,
            knn_model,
            debug=False,
        )
        print(f"recommend_hybrid_movies: {time.time() - start:.2f} s")

        again = input("Do you want to search again? (yes/no): ").strip().lower()
        if again != "yes":
            break


if __name__ == "__main__":
    main()
