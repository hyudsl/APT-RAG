import json
import os
import pickle
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document


SIDECAR_FILENAME = "docs.jsonl"
# maximum chunk size for reverse search (if a line is longer than this, read it in chunks)
_REVERSE_READ_CHUNK = 1 << 15  # 32KB


class SidecarWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.path, "a", encoding="utf-8")

    def append_many(
        self,
        start_row_id: int,
        doc_ids,
        texts,
        metadatas,
    ) -> int:
        row_id = int(start_row_id)
        write = self._fp.write
        for doc_id, text, metadata in zip(doc_ids, texts, metadatas):
            record = {
                "row_id": row_id,
                "doc_id": doc_id,
                "text": text,
                "metadata": metadata,
            }
            write(json.dumps(record, ensure_ascii=False) + "\n")
            row_id += 1
        return row_id

    def flush(self):
        if getattr(self._fp, "closed", False):
            return
        self._fp.flush()
        try:
            os.fsync(self._fp.fileno())
        except OSError:
            pass

    def close(self):
        if getattr(self._fp, "closed", False):
            return
        try:
            self.flush()
        finally:
            self._fp.close()


def last_row_id(path: Path) -> Optional[int]:
    path = Path(path)
    if not path.exists():
        return None

    size = path.stat().st_size
    if size == 0:
        return None

    with open(path, "rb") as f:
        chunk_size = _REVERSE_READ_CHUNK
        buf = b""
        pos = size
        while pos > 0:
            read_size = min(chunk_size, pos)
            pos -= read_size
            f.seek(pos)
            buf = f.read(read_size) + buf

            stripped = buf.rstrip(b"\n\r")
            nl_idx = stripped.rfind(b"\n")
            if nl_idx != -1:
                last_line = stripped[nl_idx + 1:]
                break
            if pos == 0:
                last_line = stripped
                break
        else:
            return None

    if not last_line:
        return None
    try:
        record = json.loads(last_line.decode("utf-8"))
    except json.JSONDecodeError:
        return None
    rid = record.get("row_id")
    if rid is None:
        return None
    return int(rid)


def truncate_to_row_id(path: Path, target_next_row_id: int) -> int:
    path = Path(path)
    if target_next_row_id < 0:
        raise ValueError(f"target_next_row_id must be >= 0 (got {target_next_row_id})")
    if not path.exists():
        if target_next_row_id == 0:
            return 0
        raise FileNotFoundError(f"sidecar not found: {path}")

    cut_offset = None
    kept = 0
    expected = 0
    with open(path, "rb") as f:
        while True:
            pos_before = f.tell()
            line = f.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                # allow empty lines (trailing newlines, etc.)
                continue
            try:
                record = json.loads(stripped.decode("utf-8"))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"sidecar JSON decode failed at offset={pos_before}: {e}"
                ) from e
            rid = record.get("row_id")
            if rid is None:
                raise ValueError(f"sidecar record missing row_id at offset={pos_before}")
            rid = int(rid)
            if rid != expected:
                raise ValueError(
                    f"sidecar row_id discontinuity at offset={pos_before}: "
                    f"expected {expected}, got {rid}"
                )
            if rid >= target_next_row_id:
                cut_offset = pos_before
                break
            kept += 1
            expected += 1

    # if we read to the end of the file and didn't reach the target, there's nothing to truncate
    if cut_offset is None:
        return kept

    # truncate atomically: first truncate in the same file, then fsync.
    with open(path, "r+b") as f:
        f.truncate(cut_offset)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass

    return kept


def iter_sidecar(path: Path) -> Iterator[Tuple[int, str, str, Dict]]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            yield (
                int(record["row_id"]),
                record["doc_id"],
                record["text"],
                record.get("metadata", {}),
            )


def build_docstore_pkl(sidecar_path: Path, pkl_path: Path, expected_ntotal: int) -> int:
    sidecar_path = Path(sidecar_path)
    pkl_path = Path(pkl_path)

    docstore_dict: Dict[str, Document] = {}
    index_to_docstore_id: Dict[int, str] = {}
    expected_next = 0
    count = 0

    for row_id, doc_id, text, metadata in iter_sidecar(sidecar_path):
        if row_id != expected_next:
            raise ValueError(
                f"Sidecar row_id discontinuity at count={count}: "
                f"expected {expected_next}, got {row_id}. sidecar={sidecar_path}"
            )
        # prevent duplicate doc_ids (resume/retry, etc.)
        if doc_id in docstore_dict:
            raise ValueError(
                f"Duplicate doc_id in sidecar: '{doc_id}' at row_id={row_id}."
            )

        docstore_dict[doc_id] = Document(page_content=text, metadata=metadata)
        index_to_docstore_id[row_id] = doc_id

        expected_next += 1
        count += 1

    if count != expected_ntotal:
        raise ValueError(
            f"Sidecar count {count} != HNSW ntotal {expected_ntotal}. "
            "index and sidecar are not synchronized."
        )

    docstore = InMemoryDocstore(docstore_dict)

    tmp_path = pkl_path.with_suffix(pkl_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump((docstore, index_to_docstore_id), f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp_path, pkl_path)

    return count
