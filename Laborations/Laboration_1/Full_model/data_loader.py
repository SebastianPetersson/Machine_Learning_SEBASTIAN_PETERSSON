from functools import lru_cache
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "ml-latest"


@lru_cache(maxsize=1)
def load_ratings():
    """Loads rating dataset, returns columns 'userId', 'movieId', 'rating'."""
    return pd.read_csv(
        DATA_DIR / "ratings.csv",
        usecols=["userId", "movieId", "rating"]
    )


@lru_cache(maxsize=1)
def load_links():
    """Loads links dataset, returns columns 'movieId', 'imdbId', 'tmdbId'."""
    return pd.read_csv(
        DATA_DIR / "links.csv",
        usecols=["movieId", "imdbId", "tmdbId"]
    )


@lru_cache(maxsize=1)
def load_movies():
    """Loads movies dataset, drops duplicates, replaces NaN values in columns 'title' and 'genres' 
    with empty .str and returns movies as dataframe."""
    movies = pd.read_csv(
        DATA_DIR / "movies.csv",
        usecols=["movieId", "title", "genres"],
    )
    movies = movies.drop_duplicates(subset="movieId").copy()
    movies["title"] = movies["title"].fillna("").str.strip()
    movies["genres"] = movies["genres"].fillna("").str.strip()
    return movies


@lru_cache(maxsize=1)
def load_tags():
    """Loads tags dataset, selects columns 'movieId' and 'tag'. dropna in 'tag' and makes it a lowercase str. Drops duplicates in both
    columns and returns cleaned tags dataframe."""
    tags = pd.read_csv(
        DATA_DIR / "tags.csv",
        usecols=["movieId", "tag"],
    )
    tags = tags.dropna(subset=["tag"]).copy()
    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
    tags = tags[tags["tag"] != ""]
    tags = tags.drop_duplicates(subset=["movieId", "tag"])
    return tags