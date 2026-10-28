"""Central place for all settings.

Every other file imports its settings from here, so if you want to change
a model name, a folder, or a chunk size, you only change it in ONE place.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Read the .env file (if it exists) and put its values into environment variables.
load_dotenv()

# --- Folders ---------------------------------------------------------------
# Path(__file__) is this file (config.py). .parent is "src/", .parent again is
# the project root. Building paths this way works no matter where you run
# the program from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# --- Models (served by Ollama) ----------------------------------------------
# os.getenv("NAME", "default") reads NAME from the environment, or uses the default.
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Chunking (used in step 2) ----------------------------------------------
# A chunk is a small piece of a document. Overlap keeps sentences that sit on
# a boundary from being cut in half and losing their meaning.
CHUNK_SIZE = 1000      # max characters per chunk
CHUNK_OVERLAP = 200    # characters shared between neighbouring chunks

# --- Retrieval (used in step 4) ---------------------------------------------
TOP_K = 4              # how many chunks we fetch for each question
