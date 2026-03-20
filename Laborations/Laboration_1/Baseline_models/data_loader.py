from functools import lru_cache
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "ml-latest"


@lru_cache(maxsize=1)
def load_movies():
    movies = pd.read_csv(
        DATA_DIR / "movies.csv",
        usecols=["movieId", "title", "genres"],
    )
    movies = movies.drop_duplicates(subset="movieId").copy()
    movies["title"] = movies["title"].fillna("").str.strip()
    movies["genres"] = movies["genres"].fillna("").str.strip()
    return movies

@lru_cache(maxsize=1)
def load_ratings():
    return pd.read_csv(DATA_DIR / "ratings.csv")

@lru_cache(maxsize=1)
def load_tags():
    tags = pd.read_csv(
        DATA_DIR / "tags.csv",
        usecols=["movieId", "tag"],
    )
    tags = tags.dropna(subset=["tag"]).copy()
    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
    tags = tags[tags["tag"] != ""]
    tags = tags.drop_duplicates(subset=["movieId", "tag"])
    return tags


def load_merged_data():
    movies = load_movies()
    ratings = load_ratings()
    rated_movies = pd.merge(ratings, movies, on="movieId", how="inner")
    return rated_movies
