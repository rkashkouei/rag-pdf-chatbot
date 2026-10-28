# RAG PDF Chatbot

A chatbot that answers questions about your own PDF files. It runs fully on
your computer: no API keys, no cost, and your documents never leave your machine.

**Stack:** Python, LangChain, Chroma (vector database), Ollama (local LLM)

## How it works (RAG in 5 steps)

1. **Load** the PDFs and extract their text
2. **Split** the text into small chunks
3. **Embed** each chunk into a vector (a list of numbers that captures meaning)
4. **Store** the vectors in Chroma
5. **Ask**: embed the question, find the most similar chunks, give them to the
   LLM, and let it write an answer based only on that context

## Setup

### 1. Install Ollama and download the models

Install Ollama from https://ollama.com, then run:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your settings and PDFs

```bash
cp .env.example .env             # optional: only needed to change defaults
```

Put your PDF files into `data/pdfs/`.

## Project structure

```
rag-pdf-chatbot/
├── data/pdfs/          # your PDF files (not uploaded to GitHub)
├── src/
│   └── config.py       # all settings in one place
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## Roadmap

- [x] Step 1: Project setup
- [x] Step 2: Load and split PDFs
- [ ] Step 3: Embeddings and Chroma vector store
- [ ] Step 4: Retrieval and answer generation
- [ ] Step 5: Command-line chat with source citations
- [ ] Step 6: Tests and polish
