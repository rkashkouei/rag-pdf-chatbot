"""Step 3: embeddings and the Chroma vector store.

Two commands you can run from the project root:

    python -m src.vectorstore build
        Reads the PDFs, splits them, embeds every chunk and saves the vectors.

    python -m src.vectorstore search "your question here"
        Embeds your question and prints the most similar chunks.

Ollama must be running, and the embedding model must be downloaded.
"""

import argparse
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from src.config import CHROMA_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL, TOP_K
from src.ingest import load_pdfs, split_documents

# A collection is like a table inside the database. One is enough for us.
COLLECTION_NAME = "pdf_chunks"


def get_embeddings() -> OllamaEmbeddings:
    """The object that turns text into vectors, using the local Ollama model."""
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


def build_vectorstore(chunks: list[Document], reset: bool = True) -> Chroma:
    """Embed all chunks and save them to disk.

    reset=True deletes the old database first. Without that, running
    "build" twice would store every chunk twice and give duplicate results.
    """
    if reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )


def load_vectorstore() -> Chroma:
    """Open the database that "build" saved earlier (no re-embedding needed)."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            "No vector database found. Run: python -m src.vectorstore build"
        )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def search(query: str, k: int = TOP_K) -> list[tuple[Document, float]]:
    """Return the k chunks closest in meaning to the query, with a distance score.

    Lower distance = more similar.
    """
    store = load_vectorstore()
    return store.similarity_search_with_score(query, k=k)


def cmd_build() -> None:
    pages = load_pdfs()
    chunks = split_documents(pages)
    print(f"Embedding {len(chunks)} chunks with '{EMBEDDING_MODEL}'... "
          "(this can take a minute)")
    build_vectorstore(chunks)
    print(f"Done. Saved the vector database to {CHROMA_DIR}")


def cmd_search(query: str) -> None:
    results = search(query)
    print(f"Question: {query}\nTop {len(results)} chunks:")
    for rank, (chunk, score) in enumerate(results, start=1):
        source = Path(chunk.metadata.get("source", "unknown")).name
        page = chunk.metadata.get("page", "?")
        print(f"\n--- #{rank} | distance {score:.3f} | {source} | page {page} ---")
        print(chunk.page_content[:400])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or query the vector store.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="embed the PDFs and save the database")
    search_parser = sub.add_parser("search", help="find chunks similar to a question")
    search_parser.add_argument("query", help="your question, in quotes")
    args = parser.parse_args()

    if args.command == "build":
        cmd_build()
    else:
        cmd_search(args.query)


if __name__ == "__main__":
    main()
