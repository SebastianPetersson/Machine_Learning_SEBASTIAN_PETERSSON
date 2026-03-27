from dash import Dash, Input, Output, callback, dcc, html, no_update

from app_backend import get_dropdown_options, recommend_for_movie_id, load_search_artifacts, load_recommendation_artifacts

app = Dash(__name__)
server = app.server
app.title = "Movie recommender"

app.layout = html.Div(
    [
        html.Div(
            [
                html.P("Movie Recommendations", className="hero-eyebrow"),
                html.H1("Notflix", className="hero-title"),
                html.P(
                    "Search for a film title and get a set of similar recommendations with posters, genres, and model scores.",
                    className="hero-description",
                ),
            ],
            className="hero",
        ),
        html.Div(
            [
                html.P("Start typing to search", className="search-label"),
                dcc.Dropdown(
                    id="movie-dropdown",
                    options=[],
                    placeholder="Try Titanic, The Matrix, The Room...",
                    searchable=True,
                    clearable=True,
                    maxHeight=360,
                    optionHeight=60,
                    className="movie-dropdown",
                ),
            ],
            className="search-panel",
        ),
        dcc.Loading(
            id="recommendation-loading",
            type="circle",
            color="#6D280A",
            fullscreen=False,
            children=html.Div(id="recommendation-results", className="results-shell")
        ),
    ],
    className="app-shell",
)


@callback(
    Output("movie-dropdown", "options"),
    Input("movie-dropdown", "search_value"),
)
def update_movie_dropdown(search_value):
    """Updates the dropdown bar with given search value."""
    if not search_value or not search_value.strip():
        return no_update

    return get_dropdown_options(search_value.strip())


@callback(
    Output("recommendation-results", "children"),
    Input("movie-dropdown", "value"),
    prevent_initial_call=True,
)
def show_recommendations(selected_movie_id):
    """Shows the recommendations for the  selected movie, using recommend_for_movie_id."""
    if selected_movie_id is None:
        return html.Div(
            "Choose a movie from the search results to see recommendations.",
            className="message-panel",
        )

    recommendations = recommend_for_movie_id(selected_movie_id, n=5)

    if recommendations.empty:
        return html.Div(
            "No recommendations found for that selection.",
            className="message-panel",
        )

    cards = []
    for _, row in recommendations.iterrows():
        poster = row.get("poster_url")
        genres = row["genres"] if row["genres"] else "Genre unavailable"

        if poster:
            image_component = html.Img(
                src=poster,
                className="movie-card-poster",
            )
        else:
            image_component = html.Div(
                "No image",
                className="movie-card-poster-placeholder",
            )

        cards.append(
            html.Div(
                [
                    image_component,
                    html.Div(
                        [
                            html.H4(row["title"], className="movie-card-title"),
                            html.P(genres, className="movie-card-genres"),
                        ],
                        className="movie-card-copy",
                    ),
                    html.Div(
                        [
                            html.Span("Model score", className="movie-card-score-label"),
                            html.Span(f"{row['final_score']:.3f}", className="movie-card-score-value"),
                        ],
                        className="movie-card-score",
                    ),
                ],
                className="movie-card",
            )
        )

    return html.Div(
        [
            html.Div(
                [
                    html.P("Results", className="results-eyebrow"),
                    html.H3("Recommended next watches", className="results-title"),
                ],
                className="results-header",
            ),
            html.Div(cards, className="results-grid"),
            html.P(
                "This application uses TMDB and the TMDB APIs but is not endorsed, certified, or otherwise approved by TMDB.",
                className="tmdb-note",
            ),
        ]
    )


if __name__ == "__main__":
    print("Loading and cacheing search artifacts. Please hold.")
    load_search_artifacts()
    print("Loading and cacheing recommendation artifacts. Please hold.")
    load_recommendation_artifacts()
    print("The app is now live, follow the link below and get started searching for new movies to watch!")
    app.run(debug=False)
