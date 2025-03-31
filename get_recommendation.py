
import math
import pandas as pd
import requests

def get_book_description(book_title):
    base_url = "https://openlibrary.org/search.json"
    params = {"title": book_title, "limit": 1}
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "docs" in data and len(data["docs"]) > 0:
            book_key = data["docs"][0].get("key", "")
            if book_key:
                book_url = f"https://openlibrary.org{book_key}.json"
                book_response = requests.get(book_url)
                if book_response.status_code == 200:
                    book_data = book_response.json()
                    return book_data.get("description", "Description not found")
    return "Description not found"

def recommend_books(book_title, filtered_books, model, collection, top_n=10):
    row = filtered_books[filtered_books['Book-Title'].str.lower() == book_title.lower()]
    if row.empty:
        external_description = get_book_description(book_title)
        if external_description and external_description != "No description available":
            query_text = book_title + " - " + str(external_description.get("description", ""))

            query_embedding = model.encode([query_text])[0]
        else:
            print(f"No description found for {book_title}. Cannot generate recommendation.")
            return pd.DataFrame()
    else:
        query_text = row.iloc[0]['text']
        query_embedding = model.encode([query_text])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n * 3,
        include=['documents', 'metadatas', 'distances']
    )

    scored_results = []
    seen_titles = set()
    original_title_lower = book_title.lower()

    for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
        rec_title = meta['Book-Title']
        if rec_title.lower() == original_title_lower or rec_title.lower() in seen_titles:
            continue
        seen_titles.add(rec_title.lower())

        similarity = 1 - dist
        rating = meta.get('avg_ratings', 0)
        count = meta.get('count_ratings', 1)
        score = similarity * (rating / 10) * math.log1p(count)

        scored_results.append((rec_title, rating, count, score))

    scored_results = sorted(scored_results, key=lambda x: x[-1], reverse=True)[:top_n]
    return pd.DataFrame(scored_results, columns=['Book Title', 'Avg Rating', 'Rating Count', 'Score'])
