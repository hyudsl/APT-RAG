import numpy as np
import json
import pickle
import torch
import random
import time
import faiss
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger
from pathlib import Path
from datetime import datetime


class VectorStore:
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device: str = None,
        persist_directory: Optional[str] = None,
        batch_size: int = 1000,
        allow_dangerous_deserialization: bool = True,
        seed: int = 42,
        index_type: str = "flat",
        hnsw_subdir: str = "hnsw_sq8",
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.persist_directory = persist_directory
        self.batch_size = batch_size
        self.index_type = index_type
        self.hnsw_subdir = hnsw_subdir

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model_name_lower = (model_name or "").lower()
        if any(k in model_name_lower for k in ["e5", "instructor", "qwen"]):
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": self.device},
                encode_kwargs={"prompt_name": "query"},
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": self.device},
            )

        if persist_directory:
            persist_path = Path(persist_directory)

            if self.index_type == "flat":
                index_path = persist_path / "index.faiss"
                if persist_path.exists() and index_path.exists():
                    logger.info(f"Loading flat VectorDB: {persist_directory}")
                    try:
                        self.vector_store = FAISS.load_local(
                            persist_directory,
                            self.embeddings,
                            allow_dangerous_deserialization=allow_dangerous_deserialization,
                        )
                        logger.info("Flat VectorDB loaded.")
                    except Exception as e:
                        logger.warning(f"Failed to load flat VectorDB: {e}")
                        self.vector_store = None
                else:
                    logger.info(f"No existing flat VectorDB found at: {persist_directory}")
                    self.vector_store = None

            elif self.index_type == "hnsw":
                hnsw_dir = persist_path / self.hnsw_subdir
                if (hnsw_dir / "hnsw.index").exists() and (hnsw_dir / "metadata_hnsw.json").exists():
                    logger.info(f"Loading HNSW index: {hnsw_dir}")
                    self._load_hnsw_with_flat_data()
                else:
                    logger.error(f"HNSW index not found: {hnsw_dir}")
                    self.vector_store = None
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
        else:
            self.vector_store = None

    def _get_hnsw_dir(self) -> Path:
        if not self.persist_directory:
            raise ValueError("persist_directory is not set.")
        return Path(self.persist_directory) / self.hnsw_subdir

    def _load_docstore_bundle(self):
        pkl_path = Path(self.persist_directory) / "index.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"index.pkl not found: {pkl_path}")
        with open(pkl_path, "rb") as f:
            docstore, index_to_docstore_id = pickle.load(f)
        return docstore, index_to_docstore_id

    def _load_hnsw_graph(self):
        try:
            hnsw_dir = self._get_hnsw_dir()
            index_file = hnsw_dir / "hnsw.index"
            metadata_file = hnsw_dir / "metadata_hnsw.json"

            if not index_file.exists() or not metadata_file.exists():
                logger.error(f"HNSW index files not found: {hnsw_dir}")
                return None

            hnsw_index = faiss.read_index(str(index_file))

            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            if hasattr(hnsw_index, "hnsw"):
                hnsw_index.hnsw.efSearch = metadata.get("ef_search", 128)
                logger.info(
                    f"HNSW index loaded: {index_file} "
                    f"(M={metadata.get('M', 'N/A')}, "
                    f"efConstruction={metadata.get('ef_construction', 'N/A')}, "
                    f"efSearch={metadata.get('ef_search', 'N/A')})"
                )
            else:
                logger.warning("Loaded index is not an HNSW index.")
                return None

            return hnsw_index
        except Exception as e:
            logger.error(f"Failed to load HNSW index: {e}")
            return None

    def _load_hnsw_with_flat_data(self):
        
        """
        Combine HNSW index with docstore from index.pkl.
        """

        try:
            logger.info(f"Loading docstore from: {self.persist_directory}/index.pkl")
            docstore, index_to_docstore_id = self._load_docstore_bundle()

            hnsw_index = self._load_hnsw_graph()
            if hnsw_index is None:
                logger.error("Failed to load HNSW index.")
                self.vector_store = None
                return

            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index=hnsw_index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id,
            )
            logger.info(f"HNSW VectorStore loaded ({self.hnsw_subdir} + index.pkl).")
        except Exception as e:
            logger.error(f"Failed to load HNSW VectorStore: {e}")
            self.vector_store = None

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        persist: bool = True,
    ) -> None:
        if documents is not None:
            texts = [doc["text"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]

        if not texts:
            return

        if self.vector_store is None:
            if self.index_type == "hnsw":
                logger.error("HNSW index is not initialized. Build the index first.")
                return
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas,
            )
        else:
            self.vector_store.add_texts(texts=texts, metadatas=metadatas)

        logger.info(f"Added {len(texts)} documents.")

        if persist and self.persist_directory:
            self._persist()

    def _persist(self) -> None:
        if not self.persist_directory or self.vector_store is None:
            return

        directory = Path(self.persist_directory)
        directory.mkdir(parents=True, exist_ok=True)

        try:
            self.vector_store.save_local(str(directory))
            metadata = {
                "model_name": self.model_name,
                "device": self.device,
                "index_type": self.index_type,
                "last_updated": datetime.now().isoformat(),
            }
            with open(directory / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save VectorDB: {e}")
            raise

    def save(self, directory: str) -> None:
        if self.vector_store is None:
            raise ValueError("No VectorDB to save.")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(directory))

        metadata = {
            "model_name": self.model_name,
            "device": self.device,
            "index_type": self.index_type,
            "created_at": datetime.now().isoformat(),
        }
        with open(directory / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"VectorDB saved: {directory}")

    def search(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        
        """
        Search for the most similar documents to the query.
        """

        if self.vector_store is None:
            raise ValueError("VectorDB is not initialized.")

        docs_and_scores, time_metrics = self.similarity_search_with_score(query, k=k)

        results = []
        for doc, score in docs_and_scores:
            results.append((
                {"id": doc.metadata.get("id", ""), "text": doc.page_content, "metadata": doc.metadata},
                float(score),
            ))
            if len(results) >= k:
                break

        return results, time_metrics

    def search_with_query_embedding(
        self, query_embedding: torch.Tensor, k: int = 5
    ) -> List[Tuple[Document, float]]:
        
        """
        Search using a precomputed query embedding vector.
        """

        if self.vector_store is None:
            raise ValueError("VectorDB is not initialized.")
        return self.vector_store.similarity_search_with_score_by_vector(query_embedding, k)

    def get_document_count(self) -> int:
        
        """
        Return the total number of documents in the VectorDB.
        """
        
        if self.vector_store is None:
            return 0
        try:
            return len(self.vector_store.index_to_docstore_id)
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0

    def get_index_info(self) -> Dict[str, Any]:

        """
        Return index metadata including document count and HNSW parameters.
        """

        if self.vector_store is None:
            return {"document_count": 0, "dimension": 0, "index_type": "None", "hnsw_params": None}

        try:
            info = {
                "document_count": len(self.vector_store.index_to_docstore_id),
                "dimension": getattr(self.vector_store.index, "d", 0),
                "index_type": type(self.vector_store.index).__name__,
            }

            if hasattr(self.vector_store.index, "hnsw"):
                hnsw_obj = self.vector_store.index.hnsw
                m_value = "N/A"
                if self.persist_directory:
                    try:
                        metadata_file = self._get_hnsw_dir() / "metadata_hnsw.json"
                        if metadata_file.exists():
                            with open(metadata_file, "r") as f:
                                m_value = json.load(f).get("M", "N/A")
                    except Exception:
                        pass
                info["hnsw_params"] = {
                    "M": m_value,
                    "efConstruction": getattr(hnsw_obj, "efConstruction", "N/A"),
                    "efSearch": getattr(hnsw_obj, "efSearch", "N/A"),
                }
            else:
                info["hnsw_params"] = None

            return info
        except Exception as e:
            logger.error(f"Failed to get index info: {e}")
            return {"document_count": 0, "dimension": 0, "index_type": "Error", "hnsw_params": None}

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Union[Callable, Dict[str, Any]]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> Tuple[List[Tuple[Document, float]], Dict[str, float]]:
        
        """
        Return documents most similar to the query with L2 distance scores.
        """

        time_metrics = {}

        start = time.perf_counter()
        embedding = self.vector_store._embed_query(query)
        time_metrics["load_time"] = time.perf_counter() - start

        start = time.perf_counter()
        docs = self.vector_store.similarity_search_with_score_by_vector(
            embedding, k, filter=filter, fetch_k=fetch_k, **kwargs
        )
        time_metrics["search_time"] = time.perf_counter() - start

        return docs, time_metrics
