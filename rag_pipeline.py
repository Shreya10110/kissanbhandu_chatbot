# -*- coding: utf-8 -*-
"""PDF → chunks → embeddings → FAISS index pipeline with query support."""

import os
import pickle
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# sentence-transformers and FAISS for embeddings/store
from sentence_transformers import SentenceTransformer
import faiss


def load_and_split_pdfs(folder_path="data", chunk_size=800, overlap=150):
    documents = []
    print("Files inside data folder:", os.listdir(folder_path))

    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text = page.extract_text() or ""
                documents.append(text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_text(doc))

    print(f"Total chunks before cleanup: {len(chunks)}")

    # Remove near-empty/junk chunks (page numbers, stray URLs, headers)
    min_length = 30
    chunks = [c for c in chunks if len(c.strip()) >= min_length]

    # Remove exact duplicates while preserving order
    seen = set()
    deduped_chunks = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            deduped_chunks.append(c)
    chunks = deduped_chunks

    print(f"Total chunks after removing short/junk chunks and duplicates: {len(chunks)}")
    return chunks


def build_faiss_index(chunks, model_name="all-MiniLM-L6-v2", index_dir="faiss_index"):
    os.makedirs(index_dir, exist_ok=True)

    print("Loading embedding model...", model_name)
    model = SentenceTransformer(model_name)

    print("Computing embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "index.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    print(f"FAISS index saved with {len(chunks)} vectors in '{index_dir}'.")


def embed_query(query, model_name="all-MiniLM-L6-v2"):
    """Embed a query string using sentence-transformers."""
    model = SentenceTransformer(model_name)
    return model.encode([query], convert_to_numpy=True)[0]


def search_index(query, k=3, index_dir="faiss_index"):
    """Search the FAISS index for *query* and return top *k* results."""
    idx = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "index.pkl"), "rb") as f:
        chunks = pickle.load(f)

    q_vec = embed_query(query)
    D, I = idx.search(q_vec.reshape(1, -1), k)
    results = [chunks[i] for i in I[0]]
    return results


if __name__ == "__main__":
    chunks = load_and_split_pdfs()
    build_faiss_index(chunks)
    print("Chunks and FAISS index created successfully.\n")

    # Interactive query loop
    while True:
        question = input("Ask farmer question (or 'quit' to exit): ")
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        results = search_index(question)
        print("\nTop Relevant Information:\n")
        for i, chunk in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(chunk[:500])
        print("\n" + "=" * 60)