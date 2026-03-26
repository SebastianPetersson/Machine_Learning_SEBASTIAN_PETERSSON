# Movie recommender - hybrid model (content + collaborative)

## To Run The Movie App Locally

This project can be run locally without any Dash server account.

### 1. Open the project folder

Work from:

```powershell
cd Laborations\Laboration_1
```

### 2. Create or activate a virtual environment

If you use the included `uv` setup:

```powershell
.\.venv\Scripts\Activate.ps1
```

If no environment exists yet, create one and install dependencies:

```powershell
uv venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Optional: enable movie posters

Posters use the TMDB API.  
Without a token, the app still works, but it will show `No image`.

Create a file called `.env` in this folder and add:

```text
TMDB_ACCESS_TOKEN=your_tmdb_read_access_token_here
```

An example file is included as `.env.example`.

### 4. Make sure the dataset exists

The app expects MovieLens data in:

```text
Laboration_1/ml-latest/
```

The following files are used:

- `movies.csv`
- `ratings.csv`
- `links.csv`
- `tags.csv`

If the dataset is not already included, place the `ml-latest` folder directly inside `Laboration_1`, not inside `Full_model`.

### 5. Start the app

From `Laboration_1`, run:

```powershell
python Full_model/app.py
```

Then open the local address shown in the terminal, usually:

```text
http://127.0.0.1:8050
```

### Notes

- The first recommendation request can be slower because model artifacts may need to be loaded or rebuilt.
- `requirements.txt` in `Laboration_1` is the local run dependency file.
- `Full_model/Procfile` was added for possible deployment, but is not needed for local use.
