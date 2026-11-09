"""Step 5: the command-line chatbot.

Start it from the project root:

    python -m src.main

Type a question and press Enter. Type "exit" to quit.
"""

from src.config import CHROMA_DIR, LLM_MODEL
from src.rag_chain import ask

EXIT_WORDS = {"exit", "quit", "q"}


def print_answer(answer: str, sources: list[str]) -> None:
    print(f"\n{answer}\n")
    if sources:  # empty when the model could not find an answer
        print("Sources:")
        for source in sources:
            print(f"  - {source}")
        print()


def main() -> None:
    # Fail early with a clear message instead of a long error later.
    if not CHROMA_DIR.exists():
        print("No vector database found. Build it first with:")
        print("  python -m src.vectorstore build")
        return

    print(f"PDF chatbot ready (model: {LLM_MODEL}). Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+D / Ctrl+C
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            print("Bye!")
            break

        try:
            answer, sources = ask(question)
        except Exception as error:  # keep the chat alive if Ollama hiccups
            print(f"\nSomething went wrong: {error}")
            print("Is Ollama running? Check with: ollama list\n")
            continue

        print_answer(answer, sources)


if __name__ == "__main__":
    main()
