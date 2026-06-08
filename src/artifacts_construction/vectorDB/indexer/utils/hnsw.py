import json
import os
import faiss
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple


HNSW_INDEX_NAME = "hnsw.index"
HNSW_PARTIAL_NAME = "hnsw.index.partial"
METADATA_NAME = "metadata_hnsw.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metric_name(metric_type: int) -> str:
    names = {
        faiss.METRIC_L2: "METRIC_L2",
        faiss.METRIC_INNER_PRODUCT: "METRIC_INNER_PRODUCT",
    }
    return names.get(metric_type, f"UNKNOWN({metric_type})")


def _temp_path(path: Path) -> Path:
    return Path(f"{path}.tmp")


def create_hnsw_sq8_index(
    dim: int,
    m: int = 64,
    ef_construction: int = 400,
    ef_search: int = 128,
    metric_type: int = faiss.METRIC_L2,
) -> faiss.Index:
    """IndexHNSWSQ(QT_8bit) 인덱스를 생성한다.

    build_hnsw_index.py 의 'sq8' 경로와 동일한 파라미터.
    """
    qtype = faiss.ScalarQuantizer.QT_8bit
    try:
        index = faiss.IndexHNSWSQ(dim, qtype, m, metric_type)
    except TypeError:
        if metric_type != faiss.METRIC_L2:
            raise RuntimeError(
                "This FAISS build cannot create non-L2 IndexHNSWSQ directly."
            )
        index = faiss.IndexHNSWSQ(dim, qtype, m)

    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    return index


def load_or_create(
    hnsw_dir: Path,
    dim: int,
    m: int,
    ef_construction: int,
    ef_search: int,
    metric_type: int = faiss.METRIC_L2,
) -> Tuple[faiss.Index, bool]:
    """partial 파일이 있으면 로드, 없으면 새로 생성.

    반환: (index, resumed)
    """
    hnsw_dir = Path(hnsw_dir)
    hnsw_dir.mkdir(parents=True, exist_ok=True)

    partial_path = hnsw_dir / HNSW_PARTIAL_NAME
    final_path = hnsw_dir / HNSW_INDEX_NAME

    if partial_path.exists():
        print(f"[INFO] Loading HNSW partial checkpoint: {partial_path}")
        index = faiss.read_index(str(partial_path))
        if not hasattr(index, "hnsw"):
            raise TypeError(
                f"Partial index at {partial_path} is not an HNSW index."
            )
        if int(index.d) != int(dim):
            raise ValueError(
                f"Partial index dimension {index.d} != expected {dim}."
            )
        # efSearch 는 런타임 튜너블 값이므로 설정값으로 덮어씀
        index.hnsw.efSearch = ef_search
        print(
            f"[INFO] Resumed HNSW: ntotal={index.ntotal:,}, "
            f"is_trained={bool(index.is_trained)}, "
            f"M={m}, efConstruction={ef_construction}, efSearch={ef_search}"
        )
        return index, True

    if final_path.exists():
        raise FileExistsError(
            f"Final HNSW index already exists: {final_path}. "
            "재인덱싱하려면 output_dir를 백업하거나 런처의 --cold 옵션을 사용하세요."
        )

    print(f"[INFO] Creating new IndexHNSWSQ(sq8): dim={dim}, M={m}, "
          f"efConstruction={ef_construction}, efSearch={ef_search}")
    index = create_hnsw_sq8_index(dim, m, ef_construction, ef_search, metric_type)
    return index, False


def atomic_write(index: faiss.Index, path: Path) -> None:
    """faiss.write_index 를 원자 쓰기(임시파일 -> replace)로 수행."""
    path = Path(path)
    tmp_path = _temp_path(path)
    faiss.write_index(index, str(tmp_path))
    os.replace(tmp_path, path)


def finalize(partial_path: Path, final_path: Path) -> None:
    """partial -> final 원자 rename."""
    partial_path = Path(partial_path)
    final_path = Path(final_path)
    if not partial_path.exists():
        raise FileNotFoundError(f"Partial index not found: {partial_path}")
    os.replace(partial_path, final_path)


def write_metadata(
    out_dir: Path,
    index: faiss.Index,
    *,
    model_name: str,
    m: int,
    ef_construction: int,
    ef_search: int,
    train_size: int,
    storage: str = "sq8",
    source_corpus: str = "",
    elapsed_sec: float = 0.0,
    extra: Dict[str, Any] = None,
) -> None:
    """store.py 가 읽는 metadata_hnsw.json 과 동일한 포맷으로 기록한다.

    build_hnsw_index.py.write_final_metadata 와 호환되는 필드 구성.
    """
    out_dir = Path(out_dir)
    metric_type = int(getattr(index, "metric_type", faiss.METRIC_L2))

    payload: Dict[str, Any] = {
        "source_index_path": source_corpus,
        "source_index_type": "DirectStream",
        "source_ntotal": int(index.ntotal),
        "dimension": int(index.d),
        "metric_type": metric_type,
        "metric_name": _metric_name(metric_type),
        "target_index_type": type(index).__name__,
        "storage": storage,
        "M": int(m),
        "ef_construction": int(ef_construction),
        "ef_search": int(ef_search),
        "train_size": int(train_size),
        "built_from_existing_vectors": False,
        "builder_library": "faiss",
        "model_name": model_name,
        "created_at": _utc_now_iso(),
        "elapsed_sec": round(float(elapsed_sec), 3),
        "note": (
            "Built in a single streaming pass: embeddings are added directly "
            "to IndexHNSWSQ without materializing a flat IndexFlatL2."
        ),
    }
    if extra:
        payload["extra"] = extra

    _write_json_atomic(out_dir / METADATA_NAME, payload)
    _write_json_atomic(out_dir / f"metadata_hnsw_{m}_{ef_construction}.json", payload)


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    tmp_path = _temp_path(path)
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)
