from preprocessing import (
    filter_ratings,
    build_item_user_matrix,
    clean_text,
    prepare_movies_for_search,
)
from sklearn.neighbors import NearestNeighbors
from data_loader import load_movies
import pandas as pd


def build_knn_model(item_user_matrix):
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(item_user_matrix)
    return model

def recommend_item_based_movies(movies, item_user_matrix, movie_to_idx, idx_to_movie, model, n=5):
    movie_title = input("Vilken film vill du se? ").strip()
    clean_query = clean_text(movie_title)

    movie_match = movies[
        movies["title_clean"].str.contains(clean_query, na=False)
    ]

    if movie_match.empty:
        print("Filmen hittades inte. Kontrollera stavningen eller välj en annan film.")
        return

    if len(movie_match) > 1:
        print("Flera filmer matchade din sökning:")

        movie_options = movie_match[["movieId", "title", "genres"]].reset_index(drop=True)
        movie_options.index = movie_options.index + 1
        print(movie_options[["title", "genres"]])

        choice = input("Välj film genom att skriva numret: ")

        try:
            choice = int(choice)
        except ValueError:
            print("Fel inmatning, skriv siffran för filmen du vill välja.")
            return
        
        if choice < 1 or choice > len(movie_options):
            print("Ogiltigt val.")
            return

        chosen_movie_id = movie_options.iloc[choice - 1]["movieId"]
        chosen_title = movie_options.iloc[choice - 1]["title"]

    else:
        chosen_movie_id = movie_match.iloc[0]["movieId"]
        chosen_title = movie_match.iloc[0]["title"]

    if chosen_movie_id not in movie_to_idx:
        print("Filmen finns i movies.csv men inte i den filtrerade ratings-datan.")
        return
        
    movie_idx = movie_to_idx[chosen_movie_id]

    distances, indices = model.kneighbors(
        item_user_matrix[movie_idx],
        n_neighbors=n + 1
    )

    neighbor_indices = indices.flatten()[1:]
    neighbor_distances = distances.flatten()[1:]
    recommend_movie_ids = [idx_to_movie[idx] for idx in neighbor_indices]

    recommendations = movies[movies["movieId"].isin(recommend_movie_ids)].copy()
    recommendations["distance"] = recommendations["movieId"].map(
        dict(zip(recommend_movie_ids, neighbor_distances))
    )

    recommendations = recommendations.sort_values("distance")
    recommendations["similarity"] = 1 - recommendations["distance"]

    print(f"\nOm du gillar '{chosen_title}' kanske du också gillar:")
    print(recommendations[["title", "genres", "similarity"]])

    return recommendations


def main():
    movies = prepare_movies_for_search(load_movies())
    ratings_filtered = filter_ratings()
    item_user_matrix, movie_to_idx, user_to_idx = build_item_user_matrix(ratings_filtered)
    idx_to_movie = {idx: movie_id for movie_id, idx in movie_to_idx.items()}
    model = build_knn_model(item_user_matrix)

    while True:
        recommend_item_based_movies(
            movies, item_user_matrix, movie_to_idx, idx_to_movie, model
        )
        again = input("Vill du söka igen? (ja/nej): ").strip().lower()
        if again != "ja":
            break


if __name__ == "__main__":
    main()
