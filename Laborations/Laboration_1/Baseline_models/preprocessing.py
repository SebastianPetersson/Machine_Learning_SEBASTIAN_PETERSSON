from functools import lru_cache
import re
import pandas as pd
from data_loader import load_movies, load_tags, load_ratings
from scipy.sparse import csr_matrix


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_movies_for_search(movies):
    movies = movies.copy()
    movies["title_no_year"] = movies["title"].str.replace(
        r"\s*\(\d{4}\)$", "", regex=True
    )
    movies["title_clean"] = movies["title_no_year"].apply(clean_text)
    movies = movies.drop(columns=["title_no_year"])
    return movies


@lru_cache(maxsize=1)
def load_movies_with_tags():
    movies = prepare_movies_for_search(load_movies())
    tags = load_tags().copy()

    movies["genres"] = (
        movies["genres"]
        .replace("(no genres listed)", "")
        .str.lower()
        .str.replace("|", " ", regex=False)
        .str.strip()
    )

    tags_grouped = (
        tags.groupby("movieId")["tag"]
        .apply(lambda x: " ".join(x))
        .reset_index()
    )

    movies_with_tags = pd.merge(movies, tags_grouped, on="movieId", how="left")
    movies_with_tags["tag"] = movies_with_tags["tag"].fillna("").str.strip()
    movies_with_tags["text_features"] = (
        movies_with_tags["genres"] + " " + movies_with_tags["tag"]
    ).str.strip()
    movies_with_tags["text_features"] = movies_with_tags["text_features"].replace("", "unknown")

    return movies_with_tags

def filter_ratings(min_user_ratings=50, min_movie_ratings=50):
    ratings = load_ratings().copy()

    active_users = ratings["userId"].value_counts()
    active_users = active_users[active_users >= min_user_ratings].index

    ratings_filtered = ratings[ratings["userId"].isin(active_users)].copy()

    popular_movies = ratings_filtered["movieId"].value_counts()
    popular_movies = popular_movies[popular_movies >= min_movie_ratings].index

    ratings_filtered = ratings_filtered[
        ratings_filtered["movieId"].isin(popular_movies)
    ].copy()

    return ratings_filtered

def build_item_user_matrix(ratings_filtered):
    movie_ids = ratings_filtered["movieId"].unique()
    user_ids = ratings_filtered["userId"].unique()

    movie_to_idx = {movie_id: i for i, movie_id in enumerate(movie_ids)}
    user_to_idx = {user_id: i for i, user_id in enumerate(user_ids)}

    row_indices = ratings_filtered["movieId"].map(movie_to_idx)
    col_indices = ratings_filtered["userId"].map(user_to_idx)
    values = ratings_filtered["rating"]

    item_user_matrix = csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(movie_ids), len(user_ids))
    )

    return item_user_matrix, movie_to_idx, user_to_idx 
