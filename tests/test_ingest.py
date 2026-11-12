import pytest
from langchain_core.documents import Document

from src.ingest import load_pdfs, split_documents


def make_doc() -> Document:
    text = "This is a sentence about RAG. " * 100
    return Document(page_content=text, metadata={"source": "x.pdf", "page": 0})


def test_split_creates_multiple_chunks():
    chunks = split_documents([make_doc()], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1


def test_chunks_respect_size_limit():
    chunks = split_documents([make_doc()], chunk_size=200, chunk_overlap=20)
    assert all(len(c.page_content) <= 200 for c in chunks)


def test_chunks_keep_metadata():
    chunks = split_documents([make_doc()], chunk_size=200, chunk_overlap=20)
    assert all(c.metadata["source"] == "x.pdf" for c in chunks)
    assert all(c.metadata["page"] == 0 for c in chunks)


def test_load_pdfs_raises_when_folder_is_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pdfs(tmp_path)
