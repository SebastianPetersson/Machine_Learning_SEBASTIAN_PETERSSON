import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from functools import lru_cache

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_API_BASE = "https://api.themoviedb.org/3"
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)

def get_tmdb_headers():
    """Uses TMDB_ACCESS_TOKEN to return TMDB API headers, or None if missing. """
    token = os.getenv("TMDB_ACCESS_TOKEN")
    if not token:
        return None
    
    return {
        "Authorization": f"Bearer {token}",
        "accept" : "application/json"
    }

@lru_cache(maxsize=2048)
def get_poster_url(tmdb_id):
    """Return the TMDB poster URL for a movie, or None if unavailable."""

    if tmdb_id is None:
        return None

    try:
        tmdb_id = int(tmdb_id)
    except (TypeError, ValueError):
        return None

    headers = get_tmdb_headers()
    if headers is None:
        return None

    url = f"{TMDB_API_BASE}/movie/{tmdb_id}"

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException:
        return None

    poster_path = data.get("poster_path")
    if not poster_path:
        return None

    return f"{TMDB_IMAGE_BASE}{poster_path}"
