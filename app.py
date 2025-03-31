import streamlit as st
import chromadb
from data_preprocessing import process_book_data
from create_database import create_chroma_db
from get_recommendation import recommend_books
import os
import pandas as pd

# --- Page Setup ---
st.set_page_config(page_title="Book Recommender", layout="centered")
st.title("📚 Book Recommendation System")

# --- Step 1: Load Preprocessed Data ---
print("Loading preprocessed data...")
books, final_ratings, avg_ratings, count_ratings = process_book_data()

# --- Step 2: Setup ChromaDB (only embed if empty) ---
collection, model, filtered_books = create_chroma_db(
    books, final_ratings, avg_ratings, count_ratings
)

# If collection was already populated, load filtered_books from disk
if filtered_books is None:
    filtered_books_path = "cached_data/filtered_books.csv"
    if os.path.exists(filtered_books_path):
        filtered_books = pd.read_csv(filtered_books_path)
    else:
        st.error("Missing 'filtered_books.csv'! Please run embedding at least once.")
        st.stop()

# Also ensure Chroma client is live
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="book_embeddings")

print("System is ready!")

# --- Step 3: User Input ---
book_input = st.text_input("Enter a book title:", "The Secret")
top_n = st.slider("How many recommendations do you want?", 1, 10, 5)

# --- Step 4: Recommendation Output ---
if st.button("Get Recommendations"):
    with st.spinner("Finding recommendations..."):
        # Load model if not returned earlier (i.e., DB already had data)
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')

        recommendations = recommend_books(
            book_input, filtered_books, model, collection, top_n=top_n
        )

    if recommendations.empty:
        st.warning("Sorry, no recommendations found.")
    else:
        st.subheader("Recommended Books:")
        st.dataframe(recommendations)
