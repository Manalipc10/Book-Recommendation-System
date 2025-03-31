
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
import chromadb

def create_chroma_db(books, final_ratings, avg_ratings, count_ratings):
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="book_embeddings")

    filtered_books = final_ratings[['Book-Title']].drop_duplicates()
    filtered_books = filtered_books.merge(books, on='Book-Title', how='left')
    filtered_books = filtered_books.merge(avg_ratings, on='Book-Title', how='left')
    filtered_books = filtered_books.merge(count_ratings, on='Book-Title', how='left')
    filtered_books.rename(columns={'Book-Rating_x': 'count_ratings', 'Book-Rating_y': 'avg_ratings'}, inplace=True)

    filtered_books['text'] = filtered_books['Book-Title'] + ' by ' + filtered_books['Book-Author'].fillna('')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = filtered_books['text'].tolist()
    embeddings = model.encode(texts).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(filtered_books))],
        metadatas=filtered_books[['Book-Title', 'Book-Author', 'avg_ratings', 'count_ratings']].to_dict('records')
    )

    return collection, model, filtered_books
