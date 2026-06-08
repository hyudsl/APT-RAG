import json
from pathlib import Path
from datetime import datetime
from .gpu import get_gpu_memory_info


class ProcessLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.error_log_path = self.log_dir / "error_log.jsonl"
        self.failed_docs_path = self.log_dir / "failed_documents.json"
        self.skipped_docs_path = self.log_dir / "skipped_documents.json"
        self.processed_ids_path = self.log_dir.parent / "processed_document_ids.txt"
        self.summary_path = self.log_dir.parent / "processing_summary.json"
        
        self.failed_docs = self.load_failed_docs()
        self.skipped_docs = self.load_skipped_docs()
        self.processed_ids = set()
        
    def load_failed_docs(self):
        if self.failed_docs_path.exists():
            with open(self.failed_docs_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"failed_doc_ids": [], "error_count": 0}
    
    def load_skipped_docs(self):
        if self.skipped_docs_path.exists():
            with open(self.skipped_docs_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"skipped_doc_ids": [], "skip_count": 0, "skips": []}
    
    def log_error(
        self, 
        error_type: str, 
        doc_id: str, 
        error_msg: str, 
        gpu_id: int = None, 
        additional_info: dict = None
    ):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "doc_id": doc_id,
            "error_message": error_msg,
            "gpu_id": gpu_id,
        }
        
        if additional_info:
            log_entry.update(additional_info)
        
        with open(self.error_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    def add_failed_doc(self, doc_id: str, error_msg: str, gpu_id: int = None):
        if doc_id not in self.failed_docs["failed_doc_ids"]:
            self.failed_docs["failed_doc_ids"].append(doc_id)
            self.failed_docs["error_count"] += 1
            
            failure_info = {
                "doc_id": doc_id,
                "error": error_msg,
                "gpu_id": gpu_id,
                "timestamp": datetime.now().isoformat()
            }
            
            if "failures" not in self.failed_docs:
                self.failed_docs["failures"] = []
            self.failed_docs["failures"].append(failure_info)
            
            self.save_failed_docs()
    
    def save_failed_docs(self):
        with open(self.failed_docs_path, "w", encoding="utf-8") as f:
            json.dump(self.failed_docs, f, indent=2, ensure_ascii=False)
    
    def add_skipped_doc(self, doc_id: str, skip_reason: str, record_scanned: int = None, additional_info: dict = None):
        if doc_id not in self.skipped_docs["skipped_doc_ids"]:
            self.skipped_docs["skipped_doc_ids"].append(doc_id)
            self.skipped_docs["skip_count"] += 1
            
            skip_info = {
                "doc_id": doc_id,
                "reason": skip_reason,
                "record_scanned": record_scanned,
                "timestamp": datetime.now().isoformat()
            }
            
            if additional_info:
                skip_info.update(additional_info)
            
            self.skipped_docs["skips"].append(skip_info)
            self.save_skipped_docs()
    
    def save_skipped_docs(self):
        with open(self.skipped_docs_path, "w", encoding="utf-8") as f:
            json.dump(self.skipped_docs, f, indent=2, ensure_ascii=False)
    
    def add_processed_doc(self, doc_id: str):
        self.processed_ids.add(doc_id)
    
    def add_processed_docs(self, doc_ids: list):
        self.processed_ids.update(doc_ids)
    
    def save_processed_ids(self):
        def get_sort_key(doc_id: str) -> tuple:
            try:
                parts = doc_id.split(':')
                first_num = int(parts[0])  # doc_id
                last_num = int(parts[-1])  # chunk_index
                return (first_num, last_num)
            except (ValueError, IndexError):
                return (0, 0)
        
        with open(self.processed_ids_path, "w", encoding="utf-8") as f:
            for doc_id in sorted(self.processed_ids, key=get_sort_key):
                f.write(f"{doc_id}\n")
    
    def save_processing_summary(self, total_scanned: int = 0):
        summary = {
            "total_scanned": total_scanned,
            "total_processed": len(self.processed_ids),
            "total_failed": len(self.failed_docs["failed_doc_ids"]),
            "total_skipped": len(self.skipped_docs["skipped_doc_ids"]),
            "success_rate": f"{len(self.processed_ids) / max(total_scanned, 1) * 100:.2f}%",
            "failed_doc_ids": self.failed_docs["failed_doc_ids"],
            "skipped_doc_ids": self.skipped_docs["skipped_doc_ids"],
            "processed_ids_file": str(self.processed_ids_path),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def get_stats(self):
        return {
            "total_processed": len(self.processed_ids),
            "total_failed": len(self.failed_docs["failed_doc_ids"]),
            "total_skipped": len(self.skipped_docs["skipped_doc_ids"]),
            "error_count": self.failed_docs["error_count"],
            "skip_count": self.skipped_docs["skip_count"],
            "log_file": str(self.error_log_path),
            "failed_docs_file": str(self.failed_docs_path),
            "skipped_docs_file": str(self.skipped_docs_path),
            "processed_ids_file": str(self.processed_ids_path),
            "summary_file": str(self.summary_path)
        }


def log_config(title: str, config: dict):
    print(f"\n[INFO] {'=' * 20} {title} {'=' * 20}")
    for k, v in config.items():
        print(f"[INFO] {k:<24}: {v}")


def init_stats():
    return {
        "min": float('inf'),
        "max": 0,
        "sum": 0,
        "count": 0
    }


def update_token_stats(token_stats: dict, token_count: int):
    token_stats["min"] = min(token_stats["min"], token_count)
    token_stats["max"] = max(token_stats["max"], token_count)
    token_stats["sum"] += token_count
    token_stats["count"] += 1


def print_checkpoint_stats(
    doc_id: str,
    n_docs_since_last_save: int,
    device: str,
    gpu_id: int,
    token_stats: dict,
    batch_size_stats: dict,
    n_oom_recoveries: int
):
    print(f"\n[INFO] Checkpoint: saving at doc_id={doc_id}, docs_since_save={n_docs_since_last_save}")
    
    # gpu memory status
    if device == "cuda":
        allocated, reserved, total = get_gpu_memory_info(gpu_id)
        print(f"[INFO] GPU {gpu_id} memory: {reserved:.2f}/{total:.2f} GB ({reserved/total*100:.1f}%)")
    
    # token stats
    if token_stats["count"] > 0:
        avg_tokens = token_stats["sum"] / token_stats["count"]
        print(f"[INFO] Token stats: min={token_stats['min']}, max={token_stats['max']}, avg={avg_tokens:.1f}")
    
    # batch size stats
    if batch_size_stats["count"] > 0:
        avg_batch = batch_size_stats["sum"] / batch_size_stats["count"]
        print(f"[INFO] Batch size stats: min={batch_size_stats['min']}, max={batch_size_stats['max']}, avg={avg_batch:.1f}")
    
    # OOM recovery stats
    if n_oom_recoveries > 0:
        print(f"[INFO] OOM recoveries so far: {n_oom_recoveries}")


def print_final_stats(
    n_skipped_duplicates: int,
    total_docs: int,
    n_oom_recoveries: int,
    doc_id: str,
    token_stats: dict,
    batch_size_stats: dict,
    logger,
    total_scanned: int = 0
):
    print(f"[INFO] Total scanned: {total_scanned}")
    print(f"[INFO] Skipped duplicates: {n_skipped_duplicates}")
    print(f"[INFO] Total docs in index: {total_docs}")
    print(f"[INFO] OOM recoveries: {n_oom_recoveries}")
    print(f"[INFO] Last processed doc_id: {doc_id}")
    
    # logger stats
    log_stats = logger.get_stats()
    print(f"\n[LOG] Processing statistics:")
    print(f"      Total processed: {log_stats['total_processed']}")
    print(f"      Total failed: {log_stats['total_failed']}")
    print(f"      Total skipped: {log_stats['total_skipped']}")
    print(f"      Total errors logged: {log_stats['error_count']}")
    print(f"      Success rate: {log_stats['total_processed'] / max(total_scanned, 1) * 100:.2f}%")
    
    print(f"\n[LOG] Output files:")
    print(f"      Error log: {log_stats['log_file']}")
    print(f"      Failed docs: {log_stats['failed_docs_file']}")
    print(f"      Skipped docs: {log_stats['skipped_docs_file']}")
    print(f"      Processed IDs: {log_stats['processed_ids_file']}")
    print(f"      Summary: {log_stats['summary_file']}")
    
    # final token stats
    if token_stats["count"] > 0:
        avg_tokens = token_stats["sum"] / token_stats["count"]
        print(f"\n[INFO] Final token stats: min={token_stats['min']}, max={token_stats['max']}, avg={avg_tokens:.1f}")
    
    # final batch size stats
    if batch_size_stats["count"] > 0:
        avg_batch = batch_size_stats["sum"] / batch_size_stats["count"]
        print(f"[INFO] Final batch size stats: min={batch_size_stats['min']}, max={batch_size_stats['max']}, avg={avg_batch:.1f}")
