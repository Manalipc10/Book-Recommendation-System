import pandas as pd
import os

def process_book_data():
    # Define paths for cached preprocessed data
    cached_dir = "cached_data"
    os.makedirs(cached_dir, exist_ok=True)

    paths = {
        "books": os.path.join(cached_dir, "books.csv"),
        "final_ratings": os.path.join(cached_dir, "final_ratings.csv"),
        "avg_ratings": os.path.join(cached_dir, "avg_ratings.csv"),
        "count_ratings": os.path.join(cached_dir, "count_ratings.csv")
    }

    # If all preprocessed files exist, load and return
    if all(os.path.exists(p) for p in paths.values()):
        print("[INFO] Loading preprocessed data from cache.")
        books = pd.read_csv(paths["books"])
        final_ratings = pd.read_csv(paths["final_ratings"])
        avg_ratings = pd.read_csv(paths["avg_ratings"])
        count_ratings = pd.read_csv(paths["count_ratings"])
        return books, final_ratings, avg_ratings, count_ratings

    # Else do the full preprocessing
    print("[INFO] Preprocessing data from raw CSVs...")
    books = pd.read_csv('Books.csv', low_memory=False)
    ratings = pd.read_csv('Ratings.csv', low_memory=False)
    users = pd.read_csv('Users.csv', low_memory=False)

    merge_br = pd.merge(books, ratings, on="ISBN")
    count_ratings = merge_br.groupby('Book-Title').count()['Book-Rating'].reset_index()
    merge_br['Book-Rating'] = pd.to_numeric(merge_br['Book-Rating'], errors='coerce')

    avg_ratings = merge_br.groupby('Book-Title', as_index=False)['Book-Rating'].mean().reset_index()
    popular_df = count_ratings.merge(avg_ratings, on='Book-Title')
    popular_df.rename(columns={'Book-Rating_x': 'count_ratings', 'Book-Rating_y': 'avg_ratings'}, inplace=True)

    x = merge_br.groupby('User-ID').count()['Book-Rating'] >= 50
    users_with_most_ratings = x[x].index
    filtered_rating = merge_br[merge_br['User-ID'].isin(users_with_most_ratings)]

    y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 15
    famous_books = y[y].index
    final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

    # Save the processed results to cache
    books.to_csv(paths["books"], index=False)
    final_ratings.to_csv(paths["final_ratings"], index=False)
    avg_ratings.to_csv(paths["avg_ratings"], index=False)
    count_ratings.to_csv(paths["count_ratings"], index=False)

    print("[INFO] Preprocessed data cached successfully.")
    return books, final_ratings, avg_ratings, count_ratings
