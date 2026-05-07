"""
基于 FAISS + SapBERT 的语义向量检索（论文4.2.3节）

加载 faiss_builder.py 构建的索引，对查询文本编码后做余弦相似度检索。
"""
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.retrieval.graph_retriever import Evidence

SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


class VectorRetriever:
    def __init__(self, index_dir: str | Path, device: str = "cuda"):
        index_dir = Path(index_dir)
        self.index = faiss.read_index(str(index_dir / "index.bin"))
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.meta: list[dict] = []
        with open(index_dir / "index_meta.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.meta.append(json.loads(line))

        self.tokenizer = AutoTokenizer.from_pretrained(SAPBERT_MODEL)
        self.model = AutoModel.from_pretrained(SAPBERT_MODEL).to(self.device)
        self.model.eval()

    def _encode(self, text: str) -> np.ndarray:
        with torch.no_grad():
            enc = self.tokenizer(
                text,
                max_length=128,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc)
            vec = out.last_hidden_state[:, 0, :].cpu().float().numpy()
        faiss.normalize_L2(vec)
        return vec

    def retrieve(self, query: str, top_k: int = 10) -> list[Evidence]:
        """
        Args:
            query:  原始问题文本
            top_k:  返回最相似的证据条数
        """
        vec = self._encode(query)
        scores, indices = self.index.search(vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            m = self.meta[idx]
            results.append(Evidence(
                subject_id=m["subject_id"],
                subject_name=m["subject_name"],
                predicate=m["predicate"],
                object_id=m["object_id"],
                object_name=m["object_name"],
                confidence=float(m["confidence"]),
                pmids=[m["doc_id"]],
                sentences=[m["sentence"]],
                source="vector",
            ))

        return results
