import os
import faiss
import sqlite3
import numpy as np
import configparser
from sentence_transformers import SentenceTransformer
from .llm import query_llm

# Load config
config_path = os.getenv(
    "RAGLITE_CONFIG", "/mnt/lts/projects/raglite/config.ini")
config = configparser.ConfigParser()
config.read(config_path)

faiss_path = config["paths"]["faiss_path"]
sqlite_path = config["paths"]["sqlite_path"]
model_name = config["models"]["embedding_model"]
k_values = int(config["retrieval"]["k_values"])

rag_context = """
You are a helpful assistant. Answer the user's question using only the following context. Do not use prior knowledge, and do not include any explanation. Just return the final answer based on the most relevant chunk(s), and cite the source at the end.

Context:
{chunks}

Question: {query}

Answer:

"""


def load_faiss_index():
    return faiss.read_index(faiss_path)


def load_sqlite_chunks():
    return sqlite3.connect(sqlite_path)


def embed_query(query):
    model = SentenceTransformer(model_name)
    embedding = model.encode([query])[0]
    return np.array([embedding], dtype="float32")


index = load_faiss_index()
conn = load_sqlite_chunks()
cur = conn.cursor()


def search(query):

    # embed query and search index
    query_vec = embed_query(query)
    D, I = index.search(query_vec, k_values)

    results = []
    # for each vector, find corresponding chunk
    for idx in I[0]:
        cur.execute(
            "SELECT chunk, filename, filepath FROM chunks WHERE id = ?", (int(idx),))
        row = cur.fetchone()
        if row:
            chunk, filename, filepath = row
            results.append({
                "chunk": chunk,
                "filename": filename,
                "filepath": filepath
            })

    return results


user_input = ""

while True:
    user_input = input("What would you like to ask? (q to exit): ")
    if user_input == "q":
        break

    matches = search(user_input)
    llm_query = rag_context.format(chunks=str(matches), query=user_input)
    print(llm_query)
    llm_answer = query_llm(llm_query)
    print(llm_answer)

conn.close()
