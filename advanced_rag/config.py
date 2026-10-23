"""Settings for the three-document RAG challenge.

Everything that runs per chunk runs locally and is open source: PyMuPDF
parses, fastembed embeds (BGE dense + BM25 sparse) and reranks (MiniLM
cross-encoder), and Qdrant stores and searches in local mode (a folder on
disk, no server). Only two steps call a hosted model: writing the answer, and
describing the iMac report's chart pages once at index time. Both read
GEMINI_API_KEY from `.env` at the repository root.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
QDRANT_DIR = Path(os.getenv("ADV_QDRANT_DIR") or ROOT / ".qdrant")
# Vision descriptions are paid for once and kept, so a rebuild costs nothing.
CACHE_DIR = ROOT / ".cache"

# One collection per document. The documents never answer each other's
# questions, and separate collections keep each pipeline's chunks separate.
DOCUMENTS = {
    "gpo": DOCS_DIR / "GPO-Anual-report.pdf",
    "gep": DOCS_DIR / "GEP-Jun-2026.pdf",
    "imac": DOCS_DIR / "iMac_PER_Oct2024.pdf",
}

# bge-small (384-dim) rather than bge-base: on a 15GB, GPU-less machine
# base peaked at 6.6GB and ~2 chunks/s while indexing.
EMBED_MODEL = os.getenv("ADV_EMBED_MODEL") or "BAAI/bge-small-en-v1.5"
EMBED_DIM = int(os.getenv("ADV_EMBED_DIM") or 384)
SPARSE_MODEL = "Qdrant/bm25"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
EMBED_CACHE_DIR = os.getenv("EMBED_CACHE_DIR") or "/tmp/fastembed_cache"
# Chunks per forward pass. Activations grow with batch x longest chunk, so
# this is a memory knob first and a speed knob second.
EMBED_BATCH = int(os.getenv("ADV_EMBED_BATCH") or 8)

# Candidates each search arm returns before fusion, how many fused results
# the reranker rereads, and how many passages the model finally sees.
PREFETCH = 30
RERANK_POOL = 20
TOP_K = 5

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODELS = [m.strip() for m in ",".join([
    os.getenv("GEMINI_CHAT_MODEL") or "gemini-3.6-flash",
    os.getenv("GEMINI_FALLBACK_MODELS")
    or "gemini-3.8-flash,gemini-3.7-flash,gemini-3.5-flash,gemini-3.5-flash-lite",
]).split(",") if m.strip()]
