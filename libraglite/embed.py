import os
import sqlite3
import faiss
import configparser
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

# Load config
config_path = os.getenv(
    "RAGLITE_CONFIG", "/mnt/lts/projects/raglite/config.ini")
config = configparser.ConfigParser()
config.read(config_path)

target_path = config["paths"]["target_path"]
faiss_path = config["paths"]["faiss_path"]
sqlite_path = config["paths"]["sqlite_path"]
model_name = config["model"]["model_name"]


def initDB():
    # Setup SQLite
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        chunk TEXT,
        filename TEXT,
        filepath TEXT,
        position INTEGER
    )
    """)
    conn.commit()
    return conn


def generate():
    all_embeddings = []
    vector_id = 0

    # Load embedding model
    model = SentenceTransformer(model_name)
    # Directory with .md files
    docs_path = Path(target_path)

    # sqlite db
    conn = initDB()
    cur = conn.cursor()

    def simple_chunk(text, max_chars=200):
        paragraphs = text.split("\n\n")
        chunks = []
        chunk = ""
        for para in paragraphs:
            if len(chunk) + len(para) < max_chars:
                chunk += para + "\n\n"
            else:
                chunks.append(chunk.strip())
                chunk = para + "\n\n"
        if chunk:
            chunks.append(chunk.strip())
        return chunks

    # Process each markdown file
    for file in docs_path.glob("**/*.md"):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = simple_chunk(content)
        if not chunks:
            continue

        embeddings = model.encode(chunks)
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                "INSERT INTO chunks (id, chunk, filename, filepath, position) VALUES (?, ?, ?, ?, ?)",
                (vector_id, chunk, file.name, str(file), i)
            )
            all_embeddings.append(emb)
            vector_id += 1

    conn.commit()
    conn.close()

    # Save FAISS
    embedding_matrix = np.array(all_embeddings).astype("float32")
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    faiss.write_index(index, faiss_path)
    print(f"Stored {vector_id} embeddings to FAISS + SQLite.")


generate()
