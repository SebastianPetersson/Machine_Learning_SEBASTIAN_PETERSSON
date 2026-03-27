from functools import lru_cache
import re
import pandas as pd
from data_loader import load_movies, load_tags, load_ratings, load_links
from scipy.sparse import csr_matrix
from pathlib import Path
from joblib import dump, load

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

HYBRID_FEATURES_PATH = CACHE_DIR / "hybrid_features.joblib"

def clean_text(text):
    """Does a basic clean of the input text, .lower, .sub and .strip."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def prepare_movies_for_search(movies):
    """Prepare a movies Dataframe for title-based searches.
    Splits title-column into 'release_year' and 'title_clean'."""
    movies = movies.copy()
    movies["release_year"] = movies["title"].str.extract(r"\((\d{4})\)$")
    movies["release_year"] = pd.to_numeric(movies["release_year"], errors="coerce")

    movies["title_no_year"] = movies["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)

    movies["title_search"] = movies["title_no_year"].str.replace(r"^(.*),\s(The|A|An)$",
    r"\2 \1", regex = True)

    movies["title_clean"] = movies["title_search"].apply(clean_text)
    movies = movies.drop(columns=["title_no_year", "title_search"])
    return movies

@lru_cache(maxsize=1)
def prepare_movies_with_tags_and_links():
    """Loads and prepares movies with prepare_movies_for_search, loads tags and links. 
    Applies title processing through prepare_movies_for_search, cleans genre text, aggregates tags per movie, 
    merges movies with tags and links. Creates movies_full including new column text_features and returns movies_full."""
    movies = prepare_movies_for_search(load_movies())
    tags = load_tags().copy()
    links = load_links().copy()

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

    movies_full = pd.merge(movies, tags_grouped, on="movieId", how="left")
    movies_full = pd.merge(movies_full, links, on="movieId", how="left")

    movies_full["tag"] = movies_full["tag"].fillna("").str.strip()

    movies_full["text_features"] = (
        movies_full["genres"] + " " + movies_full["tag"]
    ).str.strip()

    movies_full["text_features"] = movies_full["text_features"].replace("", "unknown")

    return movies_full

def filter_ratings(min_user_ratings=50, min_movie_ratings=50):
    """
    Filter ratings to keep only active users and frequently rated movies.

    Users with fewer than `min_user_ratings` ratings are removed first.
    Then movies with fewer than `min_movie_ratings` ratings are removed
    from the remaining dataset.

    Args:
        min_user_ratings (int): Minimum number of ratings a user must have.
        min_movie_ratings (int): Minimum number of ratings a movie must have.

    Returns:
        pd.DataFrame: A filtered ratings DataFrame.
    """
    ratings = load_ratings().copy()

    active_users = ratings["userId"].value_counts()
    active_users = active_users[active_users >= min_user_ratings].index

    ratings_filtered = ratings[ratings["userId"].isin(active_users)].copy()

    popular_movies = ratings_filtered["movieId"].value_counts()
    popular_movies = popular_movies[popular_movies >= min_movie_ratings].index

    ratings_filtered = ratings_filtered[ratings_filtered["movieId"].isin(popular_movies)].copy()

    return ratings_filtered

def build_item_user_matrix(ratings_filtered):
    """Build a sparse item-user matrix and index mappings from filtered ratings.
    Returns:
        item_user_matrix, movie_to_idx, user_to_idx"""

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

def build_user_activity_stats(ratings):
    """Build user activity statistics with rating count and mean rating per user."""

    user_activity_stats = (
        ratings.groupby("userId")
        .agg(
            rating_count = ("rating", "size"),
            mean_rating = ("rating", "mean")
        )
        .reset_index()
    )

    return user_activity_stats

def create_user_activity_groups(user_activity_stats):
    """Group users into low, medium, high, and expert activity levels by rating count percentiles."""

    q50 = user_activity_stats["rating_count"].quantile(0.50)
    q90 = user_activity_stats["rating_count"].quantile(0.90)
    q99 = user_activity_stats["rating_count"].quantile(0.99)

    low_users = user_activity_stats[
        user_activity_stats["rating_count"] < q50
    ]["userId"]

    medium_users = user_activity_stats[
        (user_activity_stats["rating_count"] >= q50) & 
        (user_activity_stats["rating_count"] < q90)
    ]["userId"]

    high_users = user_activity_stats[
        (user_activity_stats["rating_count"] >= q90) &
        (user_activity_stats["rating_count"] < q99)
    ]["userId"]

    expert_users = user_activity_stats[
        user_activity_stats["rating_count"] >= q99
    ]["userId"]

    return {
        "low" : low_users,
        "medium" : medium_users,
        "high" : high_users,
        "expert" : expert_users
    }

def build_movie_stats(ratings):
    """Build movie statistics with rating count and average rating per movie."""

    movie_stats = (
        ratings.groupby("movieId")
        .agg(
            rating_count=("rating", "size"),
            average_rating=("rating", "mean")
        )
        .reset_index()
    )

    return movie_stats

def create_movie_popularity_groups(movie_stats):
    """Group movies into popularity levels based on rating count percentiles."""

    q50 = movie_stats["rating_count"].quantile(0.50)
    q90 = movie_stats["rating_count"].quantile(0.90)
    q99 = movie_stats["rating_count"].quantile(0.99)

    low_movies = movie_stats[
        movie_stats["rating_count"] < q50
    ]["movieId"]

    medium_movies = movie_stats[
        (movie_stats["rating_count"] >= q50) &
        (movie_stats["rating_count"] < q90)
    ]["movieId"]

    high_movies = movie_stats[
        (movie_stats["rating_count"] >= q90) &
        (movie_stats["rating_count"] < q99)
    ]["movieId"]

    very_high_movies = movie_stats[
        movie_stats["rating_count"] >= q99
    ]["movieId"]

    return {
        "low": low_movies,
        "medium": medium_movies,
        "high": high_movies,
        "very_high": very_high_movies
    }

def build_expert_movie_stats(ratings, user_groups):
    """Build per-movie rating statistics from ratings made by expert users."""

    expert_users = user_groups["expert"]

    expert_ratings = ratings[ratings["userId"].isin(expert_users)].copy()

    expert_movie_stats = (
        expert_ratings.groupby("movieId")
        .agg(
            expert_rating_count = ("rating", "size"),
            expert_mean_rating = ("rating", "mean")
        )
        .reset_index()
    )

    return expert_movie_stats

def build_hybrid_feature_table():
    """
    Build a hybrid movie feature table by combining metadata and rating statistics.

    Merges movie metadata, tags, and links with rating-based features such as
    rating count, average rating, expert-user statistics, and popularity group.
    Missing rating-related values are filled with defaults.

    Returns:
        pd.DataFrame: A DataFrame containing movie metadata together with
        hybrid features for recommendation or analysis.
    """

    movies_full = prepare_movies_with_tags_and_links()
    ratings_filtered = filter_ratings()

    movie_stats = build_movie_stats(ratings_filtered)
    user_activity_stats = build_user_activity_stats(ratings_filtered)
    user_groups = create_user_activity_groups(user_activity_stats)
    expert_movie_stats = build_expert_movie_stats(ratings_filtered, user_groups)
    popularity_groups = create_movie_popularity_groups(movie_stats)

    movie_stats["popularity_group"] = "medium"
    movie_stats.loc[movie_stats["movieId"].isin(popularity_groups["low"]), "popularity_group"] = "low"
    movie_stats.loc[movie_stats["movieId"].isin(popularity_groups["medium"]), "popularity_group"] = "medium"
    movie_stats.loc[movie_stats["movieId"].isin(popularity_groups["high"]), "popularity_group"] = "high"
    movie_stats.loc[movie_stats["movieId"].isin(popularity_groups["very_high"]), "popularity_group"] = "very_high"

    hybrid_features = pd.merge(
        movies_full,
        movie_stats,
        on = "movieId",
        how = "left"
    )

    hybrid_features = pd.merge(
        hybrid_features,
        expert_movie_stats,
        on = "movieId",
        how = "left"
    )

    hybrid_features["rating_count"] = hybrid_features["rating_count"].fillna(0)
    hybrid_features["average_rating"] = hybrid_features["average_rating"].fillna(0)
    hybrid_features["expert_rating_count"] = hybrid_features["expert_rating_count"].fillna(0)
    hybrid_features["expert_mean_rating"] = hybrid_features["expert_mean_rating"].fillna(0)
    hybrid_features["popularity_group"] = hybrid_features["popularity_group"].fillna("medium")


    return hybrid_features

def load_or_build_hybrid_feature_table(force_rebuild=False):
    """Load a saved hybrid feature table or build and save it if needed."""

    if HYBRID_FEATURES_PATH.exists() and not force_rebuild:
        return load(HYBRID_FEATURES_PATH)

    hybrid_features = build_hybrid_feature_table()
    dump(hybrid_features, HYBRID_FEATURES_PATH)

    return hybrid_features
