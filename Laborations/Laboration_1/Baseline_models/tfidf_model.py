import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from preprocessing import clean_text, load_movies_with_tags

# I denna modellen använder jag mig enbart av TF-IDF, med movies.csv och tags.csv. Jag Kollar cosinus-likhet mellan 
# orden i tag och genres, vilket jag slår ihop i en ny kolumn text-features. 
# Jag använder mig av unigram och bigram i min tfidf vectorizer, men ska även prova trigram för att se om det ger bättre effekt.
# Detta är alltså en content based modell och fortfarande väldigt grundläggande. 

def build_tfidf_matrix(movies_with_tags):
    tfidf = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
    )
    tfidf_matrix = tfidf.fit_transform(movies_with_tags["text_features"])
    return tfidf, tfidf_matrix


def recommend_tfidf_movies(movies_with_tags, tfidf_matrix, n=5):
    movie_title = input("What movie would you like to watch? ").strip()
    clean_query = clean_text(movie_title)

    movie_match = movies_with_tags[
        movies_with_tags["title_clean"].str.contains(clean_query, na=False)
    ]

    if movie_match.empty:
        print("Movie could not be found. Check the spelling or choose a different movie.")
        return

    if len(movie_match) > 1:
        print("Several movies matched your search:")

        movie_options = movie_match[["title", "genres"]].reset_index()
        movie_options.index = movie_options.index + 1
        print(movie_options[["title", "genres"]])

        choice = input("Select a movie by entering the number: ")

        try:
            choice = int(choice)
        except ValueError:
            print("Invalid input. Enter the number for the movie you want to choose.")
            return
        
        if choice < 1 or choice > len(movie_options):
            print("Invalid choice.")
            return

        chosen_row = movie_options.iloc[choice - 1]
        movie_index = chosen_row["index"]
        chosen_title = chosen_row["title"]

    else:
        movie_index = movie_match.index[0]
        chosen_title = movie_match.iloc[0]["title"]

    movie_vector = tfidf_matrix[movie_index]
    similarity_scores = linear_kernel(tfidf_matrix, movie_vector).flatten()

    similarity_df = pd.DataFrame({
        "title": movies_with_tags["title"],
        "genres": movies_with_tags["genres"],
        "similarity": similarity_scores
    })

    similarity_df = similarity_df.drop(index=movie_index)

    recommendations = (
        similarity_df
        .sort_values("similarity", ascending=False)
        .head(n)
    )

    print(f"\nIf you like '{chosen_title}', you may also like:")
    print(recommendations[["title", "genres", "similarity"]])

    return recommendations


def main():
    movies_with_tags = load_movies_with_tags()
    _, tfidf_matrix = build_tfidf_matrix(movies_with_tags)

    while True:
        recommend_tfidf_movies(movies_with_tags, tfidf_matrix)
        again = input("Do you want to search again? (yes/no): ").strip().lower()
        if again != "yes":
            break


if __name__ == "__main__":
    main()
