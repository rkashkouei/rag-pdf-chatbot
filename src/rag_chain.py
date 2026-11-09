"""Step 4: retrieval + answer generation (the "RAG" part).

Try it from the project root:

    python -m src.rag_chain "your question here"

Needs: Ollama running, the llama3.2 model, and a database built with
    python -m src.vectorstore build
"""

import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.config import LLM_MODEL, OLLAMA_BASE_URL, TOP_K
from src.vectorstore import load_vectorstore

# The prompt is the most important "setting" of a RAG system.
# It tells the model: use ONLY the context, and admit when the answer is missing.
# The fixed sentence the model must use when the context has no answer.
# Keeping it in one variable lets the code detect a refusal later.
NOT_FOUND = "I could not find this in the documents."

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about documents.\n\n"
    "Rules:\n"
    "- Answer ONLY with information from the context below.\n"
    "- If the context does not contain the answer, say exactly:\n"
    f'  "{NOT_FOUND}"\n'
    "- Do not use outside knowledge and do not guess.\n"
    "- Keep the answer clear and concise.\n\n"
    "Context:\n{context}"  # {context} is filled in by LangChain at run time
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def get_llm() -> ChatOllama:
    """The local chat model. temperature=0 makes answers stable and factual."""
    return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)


def retrieve(question: str, k: int = TOP_K) -> list[Document]:
    """Find the k chunks whose meaning is closest to the question."""
    store = load_vectorstore()
    return store.similarity_search(question, k=k)


def format_context(chunks: list[Document]) -> str:
    """Join the chunks into one text block, each labelled with its source."""
    parts = []
    for chunk in chunks:
        name = Path(chunk.metadata.get("source", "unknown")).name
        page = chunk.metadata.get("page")
        label = f"{name}, page {page + 1}" if isinstance(page, int) else name
        parts.append(f"[{label}]\n{chunk.page_content}")
    return "\n\n".join(parts)


def unique_sources(chunks: list[Document]) -> list[str]:
    """List each (file, page) pair once, in the order it was retrieved."""
    seen: list[str] = []
    for chunk in chunks:
        name = Path(chunk.metadata.get("source", "unknown")).name
        page = chunk.metadata.get("page")
        label = f"{name}, page {page + 1}" if isinstance(page, int) else name
        if label not in seen:
            seen.append(label)
    return seen


def ask(question: str) -> tuple[str, list[str]]:
    """The whole RAG flow. Returns (answer, sources)."""
    chunks = retrieve(question)                       # 1. retrieve
    context = format_context(chunks)                  # 2. build the context
    chain = prompt | get_llm() | StrOutputParser()    # 3. prompt -> model -> text
    answer = chain.invoke({"context": context, "question": question})

    # If the model said "not found", the retrieved chunks were not useful,
    # so listing them as sources would be misleading.
    if NOT_FOUND.rstrip(".").lower() in answer.lower():
        return answer, []
    return answer, unique_sources(chunks)


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m src.rag_chain "your question"')
        raise SystemExit(1)

    question = " ".join(sys.argv[1:])
    answer, sources = ask(question)
    print(f"\nQuestion: {question}\n")
    print(f"Answer:\n{answer}\n")
    print("Sources:")
    for source in sources:
        print(f"  - {source}")


if __name__ == "__main__":
    main()
