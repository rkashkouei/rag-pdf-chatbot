"""Step 2: load PDFs and split them into chunks.

Run it from the project root to see the result:

    python -m src.ingest
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, PDF_DIR


def load_pdfs(pdf_dir: Path = PDF_DIR) -> list[Document]:
    """Read every PDF in a folder. Returns one Document per PDF page."""
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {pdf_dir}. Add at least one .pdf file there."
        )

    documents: list[Document] = []
    for pdf_path in pdf_files:
        # PyPDFLoader returns a list with one Document for each page.
        pages = PyPDFLoader(str(pdf_path)).load()
        documents.extend(pages)
    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Cut pages into smaller, overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,  # remember where in the page each chunk starts
    )
    return splitter.split_documents(documents)


def main() -> None:
    pages = load_pdfs()
    chunks = split_documents(pages)

    print(f"Loaded {len(pages)} pages from {PDF_DIR}")
    print(f"Created {len(chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # Show the first 3 chunks so we can see what the splitter produced.
    for i, chunk in enumerate(chunks[:3], start=1):
        source = Path(chunk.metadata.get("source", "unknown")).name
        page = chunk.metadata.get("page", "?")
        print(f"\n--- Chunk {i} | {source} | page {page} | {len(chunk.page_content)} chars ---")
        print(chunk.page_content[:300])


if __name__ == "__main__":
    main()
