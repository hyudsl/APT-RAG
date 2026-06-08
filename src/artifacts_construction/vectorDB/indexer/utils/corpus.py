import os
import json
import signal
import tiktoken
from pathlib import Path
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from .logger import ProcessLogger


def save_skipped_line(
    line: str,
    line_num: int,
    error_msg: str,
    skip_file: str
):
    record = {
        "line_number": line_num,
        "raw_line": line,
        "error": error_msg,
        "timestamp": datetime.utcnow().isoformat()
    }

    with open(skip_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_corpus_jsonl(
    corpus_jsonl: str, 
    logger: ProcessLogger = None,
    skip_file: str = "skipped_lines.jsonl"
):
    with open(corpus_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                yield json.loads(line)

            except json.JSONDecodeError as e:
                error_msg = f"JSON decode error at line {line_num}: {str(e)}"
                print(f"[WARNING] {error_msg}")

                save_skipped_line(
                    line=line,
                    line_num=line_num,
                    error_msg=error_msg,
                    skip_file=skip_file
                )

                if logger:
                    logger.log_error(
                        error_type="json_parse_error",
                        doc_id=f"line_{line_num}",
                        error_msg=error_msg,
                        additional_info={
                            "line_number": line_num,
                            "line_preview": line[:200] if len(line) > 200 else line
                        }
                    )
                continue


def get_doc_id_at_index(corpus_jsonl: str, target_idx: int) -> str:
    import subprocess
    try:
        # sed uses 1-based indexing
        result = subprocess.run(
            ["sed", "-n", f"{target_idx}p", corpus_jsonl],
            capture_output=True,
            text=True,
            check=True
        )
        line = result.stdout.strip()
        if line:
            obj = json.loads(line)
            return obj.get("document_id", "")
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return ""
    return ""


def create_embedding_model(
    model_name: str,
    gpu_id: int,
    batch_size: int,
    normalize: bool = True
) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": f"cuda:{gpu_id}", "trust_remote_code": True},
        encode_kwargs={
            "batch_size": batch_size,
            "normalize_embeddings": normalize,
        }
    )


class GracefulShutdown:
    def __init__(self):
        self._stop = False
        self._received_count = 0
        self._orig_sigint = None
        self._orig_sigterm = None

    def install(self):
        self._orig_sigint = signal.signal(signal.SIGINT, self._handle)
        self._orig_sigterm = signal.signal(signal.SIGTERM, self._handle)

    def restore(self):
        if self._orig_sigint is not None:
            signal.signal(signal.SIGINT, self._orig_sigint)
            self._orig_sigint = None
        if self._orig_sigterm is not None:
            signal.signal(signal.SIGTERM, self._orig_sigterm)
            self._orig_sigterm = None

    def should_stop(self) -> bool:
        return self._stop

    def _handle(self, signum, frame):
        self._received_count += 1
        name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        ts = datetime.now().isoformat()

        if self._received_count == 1:
            self._stop = True
            print(
                f"\n[SIGNAL] {name} received at {ts} -> scheduling graceful shutdown "
                f"(will checkpoint and exit after current batch).",
                flush=True,
            )
            return

        # force immediate exit if received more than once
        print(
            f"\n[SIGNAL] {name} received again at {ts} -> forcing immediate exit.",
            flush=True,
        )
        raise KeyboardInterrupt()


def get_tokenizer(encoding_name: str = "cl100k_base"):
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text))


def _fsync_file(f):
    f.flush()
    os.fsync(f.fileno())


def load_progress(progress_path: Path):
    if progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as f:
            j = json.load(f)
        return j.get("last_processed_doc_id", None)
    return None


def save_progress(progress_path: Path, last_doc_id: str):
    progress_path = Path(progress_path)
    tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")  # progress.json.tmp
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"last_processed_doc_id": last_doc_id}, f, ensure_ascii=False, indent=2)
        _fsync_file(f)
    os.replace(tmp_path, progress_path)
