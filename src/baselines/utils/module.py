import re
import json
from collections import defaultdict
import bm25s
import time
import Stemmer
from langchain_community.vectorstores import FAISS
from typing import Dict, List
from utils.utils import *


class Retriever:
    def __init__(self,
        corpus_path,
        base_offset,
        top_k=20,
        final_top_k=10,
        db_path=None, embedding_model=None, tokenizer=None,
        api_url=None,
        use_local=False,
        retrieval_type="dense",
        expand=True, 
        post_processing=True
        ):
        self.corpus_path = corpus_path
        self.top_k = top_k
        self.final_top_k=final_top_k
        self.base_to_offset = base_offset
        self.retrieval_type = retrieval_type
        self.expand = expand
        self.post_processing = post_processing
        self.tokenizer = tokenizer

        self.use_local = use_local
        if use_local:
            if retrieval_type == "dense":
                self.embedding_model = embedding_model
                self.vector_db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
            elif retrieval_type == "bm25":
                self.bm25_retriever = bm25s.BM25.load(db_path, load_corpus=True)
                self.bm25_stemmer = Stemmer.Stemmer("english")
        else:
            self.api_url = api_url.rstrip('/')
            check_server_health(api_url)

    @staticmethod
    def split_doc_id(doc_id: str):
        base, _, suffix = doc_id.rpartition(":")
        return base, suffix

    @staticmethod
    def split_header_body(text:str):
        t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        parts = re.split(r"\n\s*\n", t, maxsplit=1)
        
        if len(parts) == 1:
            return "", parts[0].strip()
        
        header, body = parts[0].strip(), parts[1].strip()
        return header, body

    @staticmethod
    def split_markdown_table(table_text: str):
        lines = table_text.splitlines()
        
        header_lines = []
        body_lines = []
        
        table_start_idx = None
        for idx, line in enumerate(lines):
            if line.strip().startswith("|"):
                table_start_idx = idx
                break

        if table_start_idx is None:
            return None, None

        header_lines = lines[table_start_idx: table_start_idx + 2]

        body_lines = lines[table_start_idx + 2:]

        header_text = "\n".join(header_lines)
        body_text = "\n".join(body_lines)

        return header_text, body_text

    @staticmethod
    def build_inverted_index(corpus_jsonl_path:str):
        base_to_offsets = defaultdict(list)
        docid_to_offset = {}

        with open(corpus_jsonl_path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break

                obj = json.loads(line.decode("utf-8"))
                doc_id = obj.get("document_id")
                base_id, suffix = Retriever.split_doc_id(doc_id)

                base_to_offsets[base_id].append(offset)
                docid_to_offset[doc_id] = offset
        
        return dict(base_to_offsets), docid_to_offset

    def read_chunk_at_offset(self, offset:int):
        with open(self.corpus_path, "rb") as f:
            f.seek(offset)
            line = f.readline()
            return json.loads(line.decode("utf-8"))

    def load_group_by_base_id(self, base_id:str, base_to_offset: Dict[str, List[int]]):
        offsets = base_to_offset.get(base_id, [])
        chunks = [self.read_chunk_at_offset(off) for off in offsets]

        def key_fn(c):
            _, suffix = Retriever.split_doc_id(c["document_id"])
            return float(suffix)
        chunks.sort(key=key_fn)

        return chunks

    def expand_from_topk(self, top_k_doc_ids:list):
        bases = []
        seen = set()

        for doc_id in top_k_doc_ids:
            base_id, _ = Retriever.split_doc_id(doc_id)
            if base_id not in seen:
                seen.add(base_id)
                bases.append(base_id)

        evidence_texts = []
        for base_id in bases:
            group_chunks = self.load_group_by_base_id(base_id, self.base_to_offset)
             
            merged_text = ""
            for i, c in enumerate(group_chunks):
                title = c['title']
                text = c['text']
                type_ = c['content_type']

                header, body = Retriever.split_header_body(text)

                merged_text += (header+'\n\n') if i == 0 else ""
                if type_ == 'table': 
                    table_header, body = Retriever.split_markdown_table(body)
                    merged_text += (table_header+'\n') if i == 0 else ""
                merged_text += (body+'\n')
                
                token_length = len(self.tokenizer.encode(merged_text))
                if token_length >= 2560:
                    break
                if not self.post_processing:
                    evidence_texts.append({"title":title, "content":text})
            
            if self.post_processing:
                evidence_texts.append({"title":title, "content":merged_text})
        
        return evidence_texts

    def dpr_local(self, query, r_cost):
        retrieved_list = []
    
        t0 = time.perf_counter()
        embedding = self.vector_db._embed_query(query)
        elapsed_sec0 = time.perf_counter() - t0

        t1 = time.perf_counter()
        docs_and_scores = self.vector_db.similarity_search_with_score_by_vector(embedding, k=self.top_k)
        elapsed_sec1 = time.perf_counter() - t1


        r_cost = {
            "call": r_cost.get("call", 0) + 1,
            "embed_latency": elapsed_sec0 + r_cost.get("embed_latency", 0),
            "search_latency": elapsed_sec1 + r_cost.get("search_latency", 0),
        }
        r_cost_individual = {
            "call": 1,
            "embed_latency": elapsed_sec0,
            "search_latency": elapsed_sec1,
        }

        for doc, score in docs_and_scores:
            metadata = doc.metadata or {}

            chunk_id = metadata.get("original_chunk_index") or metadata.get("chunk_index", "")
            
            retrieved_list.append({
                "title": metadata.get("title", ""),
                "content": doc.page_content,
                "url": metadata.get("url", ""),
                "doc_id": metadata.get("doc_id", ""),
                "page_id": metadata.get("page_id", -1),
                "section": metadata.get("section", ""),
                "type": metadata.get("content_type", ""),

                "chunk_id": chunk_id,
                "sub_chunk_id": metadata.get("sub_chunk_index", ""),
                "total_sub_chunk": metadata.get("total_sub_chunks", ""),

                "score": float(score)
            })

        return retrieved_list, r_cost, r_cost_individual

    def dpr_api(self, query, r_cost):
        try:
            response = requests.post(
                f"{self.api_url}/search",
                json={"query": query, "k": self.top_k},
                timeout=120
            )
            
            if response.status_code != 200:
                raise Exception(f"API request failed: {response.status_code}")

            result = response.json()
            documents = result.get("documents", [])
            elapsed_sec0 = result.get("time_metrics", {}).get("load_time", 0)
            elapsed_sec1 = result.get("time_metrics", {}).get("search_time", 0)

            r_cost = {
                "call": r_cost.get("call", 0) + 1,
                "embed_latency": elapsed_sec0 + r_cost.get("embed_latency", 0),
                "search_latency": elapsed_sec1 + r_cost.get("search_latency", 0),
            }
            r_cost_individual = {
                "call": 1,
                "embed_latency": elapsed_sec0,
                "search_latency": elapsed_sec1,
            }
            return documents, r_cost, r_cost_individual
            
        except requests.exceptions.Timeout:
            raise Exception(f"API request timed out (query: {query})")
        except Exception as e:
            raise Exception(f"Document retrieval failed: {e}")

    def DPR(self, query, r_cost):
        if self.use_local:
            return self.dpr_local(query, r_cost)
        else:
            return self.dpr_api(query, r_cost)

    def Retrieve(self, query, r_cost):
        if self.retrieval_type == "dense":
            retrieved_list, r_cost, r_cost_individual = self.DPR(query, r_cost)
        # elif self.retrieval_type == "bm25":
        #     retrieved_list = self.BM25_Retrieve(query)
        else:
            raise ValueError(f"Unknown retrieval type: {self.retrieval_type}")

        top_doc_ids = [retrieved["doc_id"] for retrieved in retrieved_list]

        if self.expand:
            evidence_texts = self.expand_from_topk(top_doc_ids)
        else: 
            evidence_texts = retrieved_list
        evidence_texts = cap_retrieved_documents_by_chars(evidence_texts)
        return evidence_texts, r_cost, r_cost_individual
    
