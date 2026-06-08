import os
import random
import argparse
import numpy as np
import torch
import uvicorn
from loguru import logger
from fastapi import FastAPI, HTTPException

import sys
from pathlib import Path
_SERVER_DIR = Path(__file__).resolve().parent
_APT_ROOT   = _SERVER_DIR.parents[3]   # server → vectorDB → artifacts_construction → src → APT-RAG
sys.path.insert(0, str(_APT_ROOT))

from src.artifacts_construction.vectorDB.server.store import VectorStore
from src.artifacts_construction.vectorDB.server.schemas import SearchRequest, SearchResponse



app = FastAPI(title="VectorStore Server", version="1.0.0")


@app.get("/")
async def root():
    cfg = app.state.config
    return {
        "message": "VectorStore Server",
        "endpoints": ["/health", "/search"],
        "config": {
            "index_type": cfg.index_type,
            "hnsw_subdir": cfg.hnsw_subdir,
        },
    }


@app.on_event("startup")
async def startup_event():
    cfg = app.state.config

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    logger.info(f"Seed fixed: {cfg.seed}")

    logger.info(f"Loading VectorDB from: {cfg.vectordb_path}")
    app.state.vector_store = VectorStore(
        persist_directory=cfg.vectordb_path,
        model_name=cfg.model,
        device=f"cuda:{cfg.gpu_id}",
        index_type=cfg.index_type,
        hnsw_subdir=cfg.hnsw_subdir,
        seed=cfg.seed,
    )
    logger.info(f"VectorStore ready. Documents: {app.state.vector_store.get_document_count():,}")


@app.get("/health")
async def health_check():
    cfg = app.state.config
    vector_store = getattr(app.state, "vector_store", None)
    if vector_store is None:
        return {"status": "error", "message": "VectorStore not loaded"}
    return {
        "status": "healthy",
        "document_count": vector_store.get_document_count(),
        "model_name": cfg.model,
        "index_type": cfg.index_type,
        "hnsw_subdir": cfg.hnsw_subdir,
    }


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    vector_store = getattr(app.state, "vector_store", None)
    if vector_store is None:
        raise HTTPException(status_code=503, detail="VectorStore not loaded")

    try:
        results, time_metrics = vector_store.search(query=request.query, k=request.k)
        documents = [
            {
                "title":        doc.get("metadata", {}).get("title", ""),
                "content":      doc.get("text", ""),
                "url":          doc.get("metadata", {}).get("url", ""),
                "doc_id":       doc.get("metadata", {}).get("document_id", ""),
                "page_id":      doc.get("metadata", {}).get("page_id", -1),
                "section_path": doc.get("metadata", {}).get("section_path", ""),
                "type":         doc.get("metadata", {}).get("content_type", ""),
                "score":        float(score),
            }
            for doc, score in results
        ]
        return SearchResponse(
            documents=documents,
            time_metrics=time_metrics,
            total_documents=vector_store.get_document_count(),
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VectorStore API server")
    parser.add_argument("--vectordb-path", required=True,               help="Path to the FAISS index directory")
    parser.add_argument("--model",         required=True,               help="HuggingFace embedding model name or path")
    parser.add_argument("--gpu-id",        type=int,   default=0,       help="CUDA device ID (default: 0)")
    parser.add_argument("--port",          type=int,   default=8007,    help="Server port (default: 8007)")
    parser.add_argument("--host",          default="0.0.0.0",           help="Server host (default: 0.0.0.0)")
    parser.add_argument("--seed",          type=int,   default=42,      help="Random seed (default: 42)")
    parser.add_argument("--index-type",    default="hnsw",              help="Index type: flat or hnsw (default: hnsw)")
    parser.add_argument("--hnsw-subdir",   default="hnsw_sq8",          help="HNSW index subdirectory (default: hnsw_sq8)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app.state.config = args
    uvicorn.run(app, host=args.host, port=args.port)
    
if __name__ == "__main__":
    main()