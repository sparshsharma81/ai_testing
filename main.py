from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import threading
import os
from dotenv import load_dotenv

# Optional Gemini client (google generative ai). Import lazily when used.
try:
    import google.generativeai as genai
    _has_gemini = True
except Exception:
    genai = None
    _has_gemini = False

load_dotenv()

# --------------------
# FastAPI App
# --------------------
app = FastAPI(title="Astrology Bot API")

# --------------------
# Pydantic Models
# --------------------
class AstrologyQuery(BaseModel):
    chart: str

class IngestRequest(BaseModel):
    id: str
    text: str
    metadata: dict = {}

class QueryRequest(BaseModel):
    query: str
    top_k: int = 4

# --------------------
# Chroma client helper (lazy, compatible across versions)
# --------------------
_client: Optional[object] = None
_collection: Optional[object] = None
_persist_dir = ".chromadb"


def _create_chroma_client(persist_directory: str = _persist_dir):
    """Try several Chroma client constructors to remain compatible across versions.

    Returns a chromadb client instance or raises ValueError/RuntimeError with
    actionable instructions if the user's data uses the legacy Chroma format.
    """
    try:
        # Preferred: try the newest API where Settings class is provided
        Settings = getattr(chromadb, "config", None)
        if Settings is not None and hasattr(chromadb.config, "Settings"):
            SettingsCls = chromadb.config.Settings
            try:
                # Try constructing with a commonly-used engine + persist dir
                settings = SettingsCls(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
                return chromadb.Client(settings=settings)
            except Exception:
                # Try with just persist_directory if supported
                try:
                    settings = SettingsCls(persist_directory=persist_directory)
                    return chromadb.Client(settings=settings)
                except Exception:
                    pass

        # Fallback to older signature
        try:
            return chromadb.Client(persist_directory=persist_directory)
        except TypeError:
            pass

        # Final fallback: no-arg client
        return chromadb.Client()
    except ValueError as e:
        # Detect the legacy-config migration error and surface a clearer message
        msg = str(e)
        if "deprecated" in msg.lower() or "migrate" in msg.lower() or "legacy" in msg.lower():
            raise RuntimeError(
                "Chroma legacy configuration detected. If you do not need to migrate existing data, create a new Chroma client using the new constructor. "
                "If you do have data to migrate install `chroma-migrate` and run `chroma-migrate`. See https://docs.trychroma.com/deployment/migration for details."
            ) from e
        raise


def get_collection(name: str = "astrology"):
    """Return a collection, creating the client/collection lazily. Raises HTTPException with useful guidance on failure."""
    global _client, _collection
    if _collection is not None:
        return _collection

    if _client is None:
        try:
            _client = _create_chroma_client()
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unable to create Chroma client: {e}")

    # get_or_create_collection may be named differently across versions
    try:
        _collection = _client.get_or_create_collection(name=name)
    except AttributeError:
        try:
            _collection = _client.create_collection(name=name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unable to create collection: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error accessing collection: {e}")

    return _collection


_embedder: Optional[SentenceTransformer] = None
_embedder_lock = threading.Lock()


def get_embedder(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    """Lazily load and return the SentenceTransformer embedder in a thread-safe way.

    This avoids heavy model downloads at module import time which can block the
    ASGI app startup and cause reload/cancellation issues.
    """
    global _embedder
    if _embedder is not None:
        return _embedder

    with _embedder_lock:
        if _embedder is not None:
            return _embedder
        _embedder = SentenceTransformer(model_name)
        return _embedder

# --------------------
# Root Endpoint
# --------------------
@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI app in main.py"}

# --------------------
# Predict Endpoint (Echo for now)
# --------------------
@app.post("/predict")
def predict(query: AstrologyQuery):
    """Run RAG: retrieve relevant documents from Chroma and call Gemini to generate an answer.

    Falls back to a simple echo if Gemini isn't configured.
    """
    user_query = query.chart

    # 1) Retrieve using collection
    coll = get_collection()
    docs = []
    try:
        # Try text-based query if supported
        results = coll.query(query_texts=[user_query], n_results=4)
        docs = results.get("documents", [[]])[0]
    except Exception:
        # Fallback to embedding-based query
        try:
            embedder = get_embedder()
            q_emb = embedder.encode(user_query).tolist()
            results = coll.query(query_embeddings=[q_emb], n_results=4)
            docs = results.get("documents", [[]])[0]
        except Exception:
            docs = []

    context = "\n\n".join(docs)

    # 2) Build prompt
    prompt = (
        "You are an expert astrology assistant. Use the context below to answer the user's question. "
        "If the context is insufficient, be honest about uncertainty and avoid inventing facts.\n\n"
        f"Context:\n{context}\n\nUser Question:\n{user_query}\n\nAnswer:\n"
    )

    # 3) Call Gemini (if configured)
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if _has_gemini and gemini_api_key:
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-pro"))
            # Use a simple generation call; API may vary by genai version
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", None) or getattr(resp, "result", None) or str(resp)
            return {"query": user_query, "answer": text, "sources": docs}
        except Exception as e:
            # On failure, return retrieval + error details (do not leak secrets)
            return {"query": user_query, "answer": None, "sources": docs, "error": str(e)}

    # 4) Fallback: return the retrieved context and an echo explanation
    fallback_answer = (
        "(Gemini not configured) Retrieved context shown below. "
        "To enable generative answers, set GEMINI_API_KEY in your environment.\n\n" + context
    )
    return {"query": user_query, "answer": fallback_answer, "sources": docs}

# --------------------
# Ingest a single item
# --------------------
@app.post("/ingest")
async def ingest(req: IngestRequest):
    embedder = get_embedder()
    embedding = embedder.encode(req.text).tolist()
    coll = get_collection()
    coll.add(
        ids=[req.id],
        metadatas=[req.metadata],
        embeddings=[embedding],
        documents=[req.text]
    )
    return {"status": "ingested", "id": req.id}

# --------------------
# Ingest all data from data.txt
# --------------------
@app.post("/ingest_all")
async def ingest_all():
    try:
        with open("data.txt", "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="data.txt not found")
    
    for idx, text in enumerate(lines):
        embedder = get_embedder()
        embedding = embedder.encode(text).tolist()
        coll = get_collection()
        coll.add(
            ids=[str(idx)],
            metadatas=[{"source": f"line_{idx}"}],
            embeddings=[embedding],
            documents=[text]
        )
    return {"status": "all data ingested", "count": len(lines)}

# --------------------
# Query Endpoint
# --------------------
@app.post("/query")
async def query(req: QueryRequest):
    embedder = get_embedder()
    query_embedding = embedder.encode(req.query).tolist()
    coll = get_collection()
    results = coll.query(
        query_embeddings=[query_embedding],
        n_results=req.top_k
    )

    answers = results['documents'][0]
    sources = results['metadatas'][0]

    return {"answer": answers, "sources": sources}


@app.get("/stats")
def stats():
    """Return basic stats about the collection: count and a small sample of documents.

    This tries several chromadb collection inspection methods to be compatible
    with different versions.
    """
    try:
        coll = get_collection()
    except HTTPException as e:
        return {"error": f"Could not access collection: {e.detail}"}

    # Try collection.get() which is common
    try:
        data = coll.get()
        ids = data.get("ids", [])
        docs = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        return {"count": len(ids), "sample_ids": ids[:5], "sample_docs": docs[:5], "sample_metadatas": metadatas[:5]}
    except Exception:
        pass

    # Try collection.count() if available
    try:
        cnt = coll.count()
        # try to fetch small sample by id slice if possible
        sample = None
        try:
            sample = coll.get(ids=[0, 1, 2, 3, 4])
        except Exception:
            sample = None
        return {"count": cnt, "sample": sample}
    except Exception:
        pass

    return {"error": "Could not determine collection contents with available chromadb methods."}
