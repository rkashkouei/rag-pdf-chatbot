# RAG PDF Chatbot

A chatbot that answers questions about your own PDF files. It runs fully on
your computer: no API keys, no cost, and your documents never leave your machine.

**Stack:** Python, LangChain, Chroma (vector database), Ollama (local LLM)

## How it works

```
 PDFs ──► load pages ──► split into chunks ──► embed ──► Chroma (on disk)
                                                              │
 question ──► embed question ──► find closest chunks ◄────────┘
                                        │
                                        ▼
                      prompt = rules + chunks + question
                                        │
                                        ▼
                      local LLM (llama3.2) ──► answer + sources
```

1. **Load** the PDFs and extract their text (one document per page)
2. **Split** pages into small overlapping chunks
3. **Embed** each chunk into a vector with `nomic-embed-text`
4. **Store** the vectors in Chroma
5. **Ask**: embed the question, retrieve the most similar chunks, and let the
   LLM answer using only that context. If the answer is not in the documents,
   the bot says so instead of guessing.

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
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Add your PDFs

Put one or more text-based PDF files into `data/pdfs/`
(scanned images of text are not supported).

Optional: copy `.env.example` to `.env` to change the model names.

## Usage

Run all commands from the project root with the virtual environment active.

```bash
# 1. Read the PDFs and build the vector database (repeat after adding or changing PDFs)
python -m src.vectorstore build

# 2. Start chatting (type "exit" to quit)
python -m src.main
```

Useful extras:

```bash
# See which chunks the search finds for a question (no LLM involved)
python -m src.vectorstore search "your question"

# Ask a single question without the chat loop
python -m src.rag_chain "your question"

# Preview how the PDFs are split into chunks
python -m src.ingest
```

Example:

```
You: What image format supports animation?

GIF supports animation. WebP also supports animation ...

Sources:
  - sample-1-page.pdf, page 1
```

## Project structure

```
rag-pdf-chatbot/
├── data/pdfs/            # your PDF files (not uploaded to GitHub)
├── src/
│   ├── config.py         # all settings in one place
│   ├── ingest.py         # load PDFs, split into chunks
│   ├── vectorstore.py    # embeddings + Chroma (build / search)
│   ├── rag_chain.py      # retrieval, prompt, answer generation
│   └── main.py           # command-line chat loop
├── tests/                # pytest tests (no Ollama needed)
├── requirements.txt
├── .env.example
└── README.md
```

## Tests

```bash
python -m pytest -v
```

The tests use small fakes instead of Ollama, so they run offline and fast.

## Settings

Defined in `src/config.py`:

| Setting | Default | Meaning |
|---|---|---|
| `CHUNK_SIZE` | 1000 | Max characters per chunk |
| `CHUNK_OVERLAP` | 200 | Characters shared by neighbouring chunks |
| `TOP_K` | 4 | Chunks retrieved per question |

The model names can be changed in `.env` (`LLM_MODEL`, `EMBEDDING_MODEL`).
If you change the embedding model, rebuild the database.

## Limitations

- Each question is answered on its own (no conversation memory yet)
- Only text-based PDFs are supported
- Retrieval always returns the closest chunks, even if none is really relevant;
  the prompt makes the model refuse in that case
- Answer quality depends on the small local model

## Roadmap

- [x] Step 1: Project setup
- [x] Step 2: Load and split PDFs
- [x] Step 3: Embeddings and Chroma vector store
- [x] Step 4: Retrieval and answer generation
- [x] Step 5: Command-line chat with source citations
- [x] Step 6: Tests and README
- [ ] Ideas: conversation memory, Streamlit web UI, answer evaluation,
      reranking, support for more file types
