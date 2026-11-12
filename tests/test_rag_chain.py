"""Tests for the RAG chain that need neither Ollama nor a real database.

We replace the two parts that talk to the outside world (the retriever and
the LLM) with small fakes. This is called "monkeypatching" or "mocking".
"""

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src import rag_chain
from src.rag_chain import NOT_FOUND, format_context, unique_sources


def make_chunk(text: str, source: str = "data/pdfs/a.pdf", page: int = 0) -> Document:
    return Document(page_content=text, metadata={"source": source, "page": page})


def fake_llm(reply: str):
    """A stand-in for the real model that always answers with `reply`."""
    return lambda: RunnableLambda(lambda _prompt: AIMessage(content=reply))


# --- format_context / unique_sources ---------------------------------------

def test_format_context_labels_file_and_human_page_number():
    text = format_context([make_chunk("Hello", page=0)])
    assert "[a.pdf, page 1]" in text  # page 0 is shown as page 1
    assert "Hello" in text


def test_unique_sources_removes_duplicates_and_keeps_order():
    chunks = [
        make_chunk("1", source="b.pdf", page=2),
        make_chunk("2", source="a.pdf", page=0),
        make_chunk("3", source="b.pdf", page=2),  # duplicate of the first
    ]
    assert unique_sources(chunks) == ["b.pdf, page 3", "a.pdf, page 1"]


def test_unique_sources_handles_missing_page():
    chunk = Document(page_content="x", metadata={"source": "data/pdfs/a.pdf"})
    assert unique_sources([chunk]) == ["a.pdf"]


# --- ask() -------------------------------------------------------------------

def test_ask_returns_answer_and_sources(monkeypatch):
    monkeypatch.setattr(rag_chain, "retrieve", lambda q: [make_chunk("RAG is cool")])
    monkeypatch.setattr(rag_chain, "get_llm", fake_llm("RAG is cool."))

    answer, sources = rag_chain.ask("What is RAG?")

    assert answer == "RAG is cool."
    assert sources == ["a.pdf, page 1"]


def test_ask_hides_sources_when_answer_not_found(monkeypatch):
    monkeypatch.setattr(rag_chain, "retrieve", lambda q: [make_chunk("unrelated")])
    monkeypatch.setattr(rag_chain, "get_llm", fake_llm(NOT_FOUND))

    answer, sources = rag_chain.ask("Capital of Austria?")

    assert answer == NOT_FOUND
    assert sources == []


def test_refusal_detection_ignores_case_and_final_period(monkeypatch):
    monkeypatch.setattr(rag_chain, "retrieve", lambda q: [make_chunk("unrelated")])
    monkeypatch.setattr(
        rag_chain, "get_llm", fake_llm("i could not find this in the documents")
    )

    _, sources = rag_chain.ask("anything")

    assert sources == []
