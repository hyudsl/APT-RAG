import os
import time
import argparse
import numpy as np
import torch
import faiss
from tqdm import tqdm
from pathlib import Path
from utils.logger import ProcessLogger, log_config, init_stats, update_token_stats, print_checkpoint_stats, print_final_stats
from utils.gpu import get_gpu_vram_size, get_optimal_max_batch_size, get_dynamic_batch_size
from utils.corpus import (
    get_tokenizer, count_tokens,
    iter_corpus_jsonl, get_doc_id_at_index,
    create_embedding_model,
    GracefulShutdown,
    load_progress, save_progress,
)
from utils.hnsw import (
    load_or_create as hnsw_load_or_create,
    atomic_write as hnsw_atomic_write,
    finalize as hnsw_finalize,
    write_metadata as hnsw_write_metadata,
    HNSW_INDEX_NAME,
    HNSW_PARTIAL_NAME,
)
from utils.sidecar import (
    SidecarWriter,
    SIDECAR_FILENAME,
    last_row_id as sidecar_last_row_id,
    truncate_to_row_id as sidecar_truncate_to_row_id,
    build_docstore_pkl,
)


def _apply_smoke_overrides(args: argparse.Namespace) -> argparse.Namespace:
    
    """
    Override args with small smoke-test values when SMOKE=1 is set.
    """

    if os.environ.get("SMOKE", "0") != "1":
        return args

    args.output_dir = args.output_dir + "_smoke"
    args.limit_idx = 5000
    args.hnsw_train_size = 2000
    args.save_every_n_docs = 2000

    smoke_limit_env = os.environ.get("SMOKE_LIMIT")
    if smoke_limit_env:
        try:
            args.limit_idx = int(smoke_limit_env)
        except ValueError:
            print(f"[SMOKE] WARNING: SMOKE_LIMIT='{smoke_limit_env}' is not an integer, ignored.")

    print("[SMOKE] Enabled. Using smoke overrides:")
    print(f"[SMOKE]   output_dir      = {args.output_dir}")
    print(f"[SMOKE]   limit_idx       = {args.limit_idx}")
    print(f"[SMOKE]   hnsw_train_size = {args.hnsw_train_size}")
    print(f"[SMOKE]   save_every_n    = {args.save_every_n_docs}")
    return args


def _select_embed_batch_size(
    token_counts_batch,
    vram_gb: float,
    max_embed_batch_size: int,
    batch_size_stats: dict,
    enable_dynamic_batching: bool,
) -> int:
    
    """
    Select dynamic batch size and update stats.
    """

    optimal = max_embed_batch_size
    if enable_dynamic_batching and token_counts_batch:
        optimal = get_dynamic_batch_size(
            max(token_counts_batch),
            vram_gb,
            max_embed_batch_size,
        )
        batch_size_stats["min"] = min(batch_size_stats["min"], optimal)
        batch_size_stats["max"] = max(batch_size_stats["max"], optimal)
        batch_size_stats["sum"] += optimal
        batch_size_stats["count"] += 1
    return optimal


def _embed_batch(embedding_model, texts, encode_batch_size: int) -> np.ndarray:
    
    """
    Embed a list of texts and return a (N, dim) float32 contiguous array.
    """

    embedding_model.encode_kwargs["batch_size"] = encode_batch_size
    vectors = embedding_model.embed_documents(texts)
    return np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))


def _embed_batch_with_oom_recovery(
    texts,
    ids,
    main_model,
    fallback_holder,
    encode_batch_size: int,
    model_name: str,
    gpu_id: int,
    logger: ProcessLogger,
    doc_id_hint: str,
):

    """
    Embed a batch with OOM fallback to per-document processing.

    Returns: (vectors, kept_texts, kept_ids, n_oom_delta, n_failed_delta)
    """
    
    try:
        vectors = _embed_batch(main_model, texts, encode_batch_size)
        return vectors, list(texts), list(ids), 0, 0
    except torch.cuda.OutOfMemoryError as e:
        error_msg = str(e)
        print("\n[OOM] CUDA Out of Memory detected!")
        print(f"[OOM] Batch size: {len(texts)}, embed_batch_size: {encode_batch_size}")
        print(f"[OOM] Error: {error_msg[:200]}...")
        print("[OOM] Switching to individual processing with fallback model (batch_size=1)")

        logger.log_error(
            error_type="cuda_oom",
            doc_id=doc_id_hint,
            error_msg=error_msg,
            gpu_id=gpu_id,
            additional_info={"batch_size": len(texts), "embed_batch_size": encode_batch_size},
        )
        torch.cuda.empty_cache()

        fallback_model = fallback_holder.get_or_load(model_name, gpu_id)
        kept_vectors, kept_texts, kept_ids = [], [], []
        failed = 0

        for i, (text, did) in enumerate(zip(texts, ids)):
            try:
                v = fallback_model.embed_documents([text])
                kept_vectors.append(np.asarray(v[0], dtype=np.float32))
                kept_texts.append(text)
                kept_ids.append(did)
                if i % 32 == 31:
                    torch.cuda.empty_cache()
                print(f"\r[OOM Recovery] Processed {i + 1}/{len(texts)}", end="", flush=True)
            except Exception as inner_e:
                failed += 1
                inner_msg = str(inner_e)
                print(f"\n[ERROR] Failed: {did} on GPU {gpu_id}: {inner_msg}")
                logger.log_error(
                    error_type="individual_processing_error",
                    doc_id=did,
                    error_msg=inner_msg,
                    gpu_id=gpu_id,
                    additional_info={"batch_position": i, "total_batch_size": len(texts)},
                )
                logger.add_failed_doc(did, inner_msg, gpu_id)

        print(f"\n[OOM Recovery] Completed: {len(kept_ids)} succeeded, {failed} failed")
        torch.cuda.empty_cache()

        if not kept_vectors:
            return np.empty((0,), dtype=np.float32), [], [], 1, failed

        arr = np.ascontiguousarray(np.stack(kept_vectors, axis=0).astype(np.float32))
        return arr, kept_texts, kept_ids, 1, failed


class _FallbackModelHolder:

    """
    Lazily loads a fallback embedding model (batch_size=1) on first OOM.
    """

    def __init__(self):
        self._model = None

    def get_or_load(self, model_name: str, gpu_id: int):
        if self._model is None:
            print("[INFO] Lazy-loading fallback embedding model (batch_size=1) ...")
            self._model = create_embedding_model(model_name, gpu_id, batch_size=1)
        return self._model


def main(args: argparse.Namespace) -> None:
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    hnsw_dir = out_path / args.hnsw_subdir
    hnsw_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir = out_path / "sidecar"
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    logger = ProcessLogger(out_path / "logs")

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
    print(f"[INFO] GPU: {args.gpu_id} (device={device})")

    shutdown = GracefulShutdown()
    shutdown.install()

    vram_gb = 0.0
    max_embed_batch_size = args.max_embed_batch_size
    if torch.cuda.is_available():
        vram_gb = get_gpu_vram_size()
        if max_embed_batch_size is None:
            max_embed_batch_size = get_optimal_max_batch_size(vram_gb)
    else:
        if max_embed_batch_size is None:
            max_embed_batch_size = 1

    log_config("GPU", {"device": device, "GPU VRAM (GB)": f"{vram_gb:.1f}"})
    log_config("Input / Output / Model", {
        "model": args.model,
        "input": args.corpus,
        "output": args.output_dir,
    })
    log_config("Range", {"start_idx": args.start_idx, "limit_idx": args.limit_idx})
    log_config("Batch", {
        "max_embed_batch_size": max_embed_batch_size,
        "batch_size_docs": args.batch_size_docs,
        "save_every_n_docs": args.save_every_n_docs,
        "dynamic_batching": "enabled" if args.dynamic_batching else "disabled",
    })
    log_config("HNSW", {
        "subdir": args.hnsw_subdir,
        "M": args.hnsw_m,
        "efConstruction": args.hnsw_ef_construction,
        "efSearch": args.hnsw_ef_search,
        "train_size": args.hnsw_train_size,
        "storage": "sq8",
    })

    tokenizer = get_tokenizer() if args.dynamic_batching else None
    embedding_model = create_embedding_model(
        args.model,
        args.gpu_id,
        batch_size=1 if args.dynamic_batching else max_embed_batch_size,
    )
    fallback_holder = _FallbackModelHolder()

    probe_vec = np.asarray(embedding_model.embed_documents(["dimension probe"]), dtype=np.float32)
    dim = int(probe_vec.shape[1])
    del probe_vec
    print(f"[INFO] Embedding dimension = {dim}")

    hnsw_index, resumed = hnsw_load_or_create(
        hnsw_dir=hnsw_dir,
        dim=dim,
        m=args.hnsw_m,
        ef_construction=args.hnsw_ef_construction,
        ef_search=args.hnsw_ef_search,
        metric_type=faiss.METRIC_L2,
    )
    partial_path = hnsw_dir / HNSW_PARTIAL_NAME
    final_path   = hnsw_dir / HNSW_INDEX_NAME

    progress_path = out_path / "progress.json"
    sidecar_path  = sidecar_dir / SIDECAR_FILENAME

    last_processed_doc_id = load_progress(progress_path)
    if args.start_idx != 1:
        last_processed_doc_id = get_doc_id_at_index(args.corpus, args.start_idx - 1)
    resume_from_next = last_processed_doc_id is not None

    sidecar_last       = sidecar_last_row_id(sidecar_path)
    sidecar_next_row_id = 0 if sidecar_last is None else sidecar_last + 1
    hnsw_ntotal         = int(hnsw_index.ntotal)

    if sidecar_next_row_id != hnsw_ntotal:
        if sidecar_next_row_id > hnsw_ntotal:
            overflow = sidecar_next_row_id - hnsw_ntotal
            print(f"[RESUME-HEAL] sidecar is {overflow:,} rows ahead of HNSW. Rewinding to ntotal={hnsw_ntotal:,}.")
            try:
                kept = sidecar_truncate_to_row_id(sidecar_path, hnsw_ntotal)
            except ValueError as e:
                raise RuntimeError(
                    f"[RESUME-HEAL] sidecar rewind failed (corrupted): {e}\n"
                    f"  sidecar={sidecar_path}\n"
                    f"  hnsw   ={partial_path}\n"
                    "Run with --cold to start fresh."
                ) from e
            print(f"[RESUME-HEAL] Rewind done: kept={kept:,} rows")
        else:
            raise RuntimeError(
                f"Unrecoverable state: HNSW ntotal={hnsw_ntotal:,} > "
                f"sidecar next_row_id={sidecar_next_row_id:,}. "
                "Run with --cold to start fresh."
            )

    if resume_from_next:
        print(f"[INFO] Resuming after doc_id='{last_processed_doc_id}'")
    else:
        print("[INFO] Starting from the beginning")

    next_row_id = int(hnsw_index.ntotal)
    print(f"[INFO] HNSW starting ntotal = {next_row_id:,}")

    sidecar_writer = SidecarWriter(sidecar_path)

    texts_batch        = []
    ids_batch          = []
    metas_batch        = []
    token_counts_batch = []
    n_docs_since_last_save = 0
    n_oom_recoveries   = 0
    n_failed           = 0
    record_scanned     = 0
    current_doc_id     = None

    token_stats      = init_stats()
    batch_size_stats = init_stats()

    train_buffer_vecs  = []
    train_buffer_texts = []
    train_buffer_ids   = []
    train_buffer_metas = []
    train_buffer_count = 0

    limit_doc_id = get_doc_id_at_index(args.corpus, args.limit_idx) if args.limit_idx else None

    pbar = tqdm(desc="Indexing", unit=" doc", initial=next_row_id)
    run_started_at = time.perf_counter()

    def _flush_to_index(vectors: np.ndarray, texts_list, ids_list, metas_list) -> int:
        nonlocal next_row_id, train_buffer_count

        if vectors.size == 0 or not texts_list:
            return 0

        if not hnsw_index.is_trained:
            train_buffer_vecs.append(vectors)
            train_buffer_texts.extend(texts_list)
            train_buffer_ids.extend(ids_list)
            train_buffer_metas.extend(metas_list)
            train_buffer_count += len(texts_list)

            if train_buffer_count >= args.hnsw_train_size:
                buf = np.ascontiguousarray(np.concatenate(train_buffer_vecs, axis=0).astype(np.float32))
                print(f"\n[INFO] Training IndexHNSWSQ(sq8) with {buf.shape[0]:,} vectors (dim={buf.shape[1]})...")
                t0 = time.perf_counter()
                hnsw_index.train(buf)
                print(f"[INFO] Training done in {time.perf_counter() - t0:.1f}s. Adding training vectors.")
                hnsw_index.add(buf)

                added = int(buf.shape[0])
                sidecar_writer.append_many(next_row_id, train_buffer_ids, train_buffer_texts, train_buffer_metas)
                next_row_id += added

                train_buffer_vecs.clear(); train_buffer_texts.clear()
                train_buffer_ids.clear();  train_buffer_metas.clear()
                train_buffer_count = 0
                del buf
                return added
            return 0

        hnsw_index.add(vectors)
        added = int(vectors.shape[0])
        sidecar_writer.append_many(next_row_id, ids_list, texts_list, metas_list)
        next_row_id += added
        return added

    def _checkpoint(last_doc_id: str) -> None:
        nonlocal n_docs_since_last_save
        if last_doc_id is None:
            return
        if not hnsw_index.is_trained:
            print("[INFO] Skip checkpoint: HNSW not trained yet.")
            n_docs_since_last_save = 0
            return

        t0 = time.perf_counter()
        sidecar_writer.flush()
        hnsw_atomic_write(hnsw_index, partial_path)
        save_progress(progress_path, last_doc_id)
        print(
            f"[INFO] Checkpoint: ntotal={int(hnsw_index.ntotal):,}, "
            f"elapsed={time.perf_counter() - t0:.1f}s"
        )
        n_docs_since_last_save = 0

    for obj in iter_corpus_jsonl(args.corpus, logger=logger, skip_file=str(out_path / "skipped_lines.jsonl")):
        text   = obj.get("text", "")
        doc_id = obj.get("document_id", "")

        if resume_from_next:
            if doc_id == last_processed_doc_id:
                print(f"[INFO] Found last processed doc_id='{doc_id}'")
                resume_from_next = False
            continue

        if limit_doc_id is not None and doc_id == limit_doc_id:
            break

        record_scanned += 1

        if not text or not doc_id:
            missing = [f for f, v in [("text", text), ("document_id", doc_id)] if not v]
            skip_reason = f"missing_fields: {', '.join(missing)}"
            print(f"[WARNING] Record {record_scanned}: Skipped - {skip_reason}")
            logger.log_error("missing_required_fields", doc_id, skip_reason,
                             additional_info={"record_scanned": record_scanned, "missing_fields": missing})
            logger.add_skipped_doc(doc_id, skip_reason, record_scanned)
            continue

        token_count = 0
        if tokenizer and args.dynamic_batching:
            token_count = count_tokens(text, tokenizer)
            update_token_stats(token_stats, token_count)

        metadata = {
            "document_id":  doc_id,
            "page_id":       obj.get("page_id", -1),
            "title":         obj.get("title", ""),
            "url":           obj.get("url", ""),
            "section":       obj.get("section", ""),
            "content_type":  obj.get("content_type", ""),
            "chunk_index":   obj.get("chunk_index", -1),
            "token_count":   token_count,
        }

        texts_batch.append(text)
        ids_batch.append(doc_id)
        metas_batch.append(metadata)
        token_counts_batch.append(token_count)
        current_doc_id = doc_id
        pbar.update(1)

        if len(texts_batch) >= args.batch_size_docs:
            optimal = _select_embed_batch_size(token_counts_batch, vram_gb, max_embed_batch_size, batch_size_stats, args.dynamic_batching)
            vectors, kept_texts, kept_ids, oom_d, fail_d = _embed_batch_with_oom_recovery(
                texts_batch, ids_batch, embedding_model, fallback_holder,
                optimal, args.model, args.gpu_id, logger, doc_id,
            )
            n_oom_recoveries += oom_d
            n_failed         += fail_d

            if kept_ids:
                kept_metas = metas_batch if len(kept_ids) == len(texts_batch) else [dict(zip(ids_batch, metas_batch))[d] for d in kept_ids]
                _flush_to_index(vectors, kept_texts, kept_ids, kept_metas)

            n_docs_since_last_save += len(ids_batch)
            texts_batch = []; ids_batch = []; metas_batch = []; token_counts_batch = []
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if shutdown.should_stop():
            print("[INFO] Graceful shutdown requested.")
            break

        if n_docs_since_last_save >= args.save_every_n_docs:
            print_checkpoint_stats(current_doc_id, n_docs_since_last_save, str(device), args.gpu_id, token_stats, batch_size_stats, n_oom_recoveries)
            _checkpoint(current_doc_id)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pbar.close()

    # flush remaining batch
    if texts_batch:
        optimal = _select_embed_batch_size(token_counts_batch, vram_gb, max_embed_batch_size, batch_size_stats, args.dynamic_batching)
        vectors, kept_texts, kept_ids, oom_d, fail_d = _embed_batch_with_oom_recovery(
            texts_batch, ids_batch, embedding_model, fallback_holder,
            optimal, args.model, args.gpu_id, logger, current_doc_id or "",
        )
        n_oom_recoveries += oom_d
        n_failed         += fail_d
        if kept_ids:
            kept_metas = metas_batch if len(kept_ids) == len(texts_batch) else [dict(zip(ids_batch, metas_batch))[d] for d in kept_ids]
            _flush_to_index(vectors, kept_texts, kept_ids, kept_metas)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # flush remaining train buffer
    if train_buffer_count > 0 and not hnsw_index.is_trained:
        buf = np.ascontiguousarray(np.concatenate(train_buffer_vecs, axis=0).astype(np.float32))
        print(f"\n[INFO] Final training on {buf.shape[0]:,} vectors (< train_size={args.hnsw_train_size:,}).")
        hnsw_index.train(buf)
        hnsw_index.add(buf)
        added = int(buf.shape[0])
        sidecar_writer.append_many(next_row_id, train_buffer_ids, train_buffer_texts, train_buffer_metas)
        next_row_id += added
        del buf

    # graceful shutdown exit
    if shutdown.should_stop():
        if hnsw_index.is_trained and int(hnsw_index.ntotal) > 0:
            _checkpoint(current_doc_id)
        sidecar_writer.close()
        print(f"[INFO] Graceful shutdown done. ntotal={int(hnsw_index.ntotal):,}, last_doc_id={current_doc_id}.")
        return

    if int(hnsw_index.ntotal) == 0:
        sidecar_writer.close()
        raise RuntimeError("No documents were indexed. Check --corpus / --start-idx / --limit-idx.")

    hnsw_index.hnsw.efSearch = args.hnsw_ef_search
    sidecar_writer.flush()
    hnsw_atomic_write(hnsw_index, partial_path)
    hnsw_finalize(partial_path, final_path)
    print(f"[DONE] HNSW final index: {final_path} (ntotal={int(hnsw_index.ntotal):,})")

    elapsed_sec = time.perf_counter() - run_started_at
    hnsw_write_metadata(
        out_dir=hnsw_dir, index=hnsw_index,
        model_name=args.model, m=args.hnsw_m,
        ef_construction=args.hnsw_ef_construction, ef_search=args.hnsw_ef_search,
        train_size=args.hnsw_train_size, storage="sq8",
        source_corpus=args.corpus, elapsed_sec=elapsed_sec,
    )

    if current_doc_id:
        save_progress(progress_path, current_doc_id)

    sidecar_writer.close()
    print(f"[INFO] Building index.pkl from sidecar: {sidecar_path}")
    t0 = time.perf_counter()
    pkl_count = build_docstore_pkl(sidecar_path, out_path / "index.pkl", int(hnsw_index.ntotal))
    print(f"[DONE] index.pkl: {pkl_count:,} entries in {time.perf_counter() - t0:.1f}s")

    logger.save_processing_summary(total_scanned=record_scanned)
    print(f"\n[DONE] Output: {out_path}")

    print_final_stats(
        n_skipped_duplicates=0, total_docs=int(hnsw_index.ntotal),
        n_oom_recoveries=n_oom_recoveries, doc_id=current_doc_id,
        token_stats=token_stats, batch_size_stats=batch_size_stats,
        logger=logger, total_scanned=record_scanned,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a FAISS HNSW index from a corpus JSONL file.")

    # required
    parser.add_argument("--corpus",      required=True, help="Path to corpus .jsonl file")
    parser.add_argument("--output-dir",  required=True, help="Output directory for index and metadata")
    parser.add_argument("--model",       required=True, help="HuggingFace embedding model name or path")

    # corpus range
    parser.add_argument("--start-idx",  type=int, default=1,    help="Start from this line number (1-based, default: 1)")
    parser.add_argument("--limit-idx",  type=int, default=None, help="Stop at this line number (exclusive, default: no limit)")

    # batching
    parser.add_argument("--batch-size-docs",      type=int,   default=4096,    help="Number of documents per embedding batch (default: 4096)")
    parser.add_argument("--max-embed-batch-size", type=int,   default=None,    help="Max embedding sub-batch size (default: auto from VRAM)")
    parser.add_argument("--save-every-n-docs",    type=int,   default=200_000, help="Checkpoint interval in documents (default: 200000)")
    parser.add_argument("--no-dynamic-batching",  action="store_true",         help="Disable token-count-based dynamic batch sizing")

    # HNSW
    parser.add_argument("--hnsw-subdir",         default="hnsw_sq8", help="Subdirectory for HNSW index files (default: hnsw_sq8)")
    parser.add_argument("--hnsw-m",              type=int, default=64,      help="HNSW M parameter (default: 64)")
    parser.add_argument("--hnsw-ef-construction",type=int, default=400,     help="HNSW efConstruction (default: 400)")
    parser.add_argument("--hnsw-ef-search",      type=int, default=128,     help="HNSW efSearch (default: 128)")
    parser.add_argument("--hnsw-train-size",     type=int, default=100_000, help="Number of vectors for SQ8 training (default: 100000)")

    # hardware
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA device ID (default: 0)")

    args = parser.parse_args()
    args.dynamic_batching = not args.no_dynamic_batching

    args = _apply_smoke_overrides(args)
    main(args)
