#!/usr/bin/env python3
"""
FAISS 向量索引构建（论文4.2.3节）

从 Neo4j 中读取所有边的 evidence_sentences，用 SapBERT 编码为 768 维向量，
构建 FAISS 平面索引（IndexFlatIP，内积等价余弦相似度）。

同时保存索引元数据文件（index_meta.jsonl），记录向量下标对应的三元组信息。

用法：
    python -m src.retrieval.faiss_builder \\
        --uri bolt://localhost:7687 \\
        --user neo4j --password your_password \\
        --output-dir data/faiss \\
        --batch-size 256
"""
import argparse
import json
import logging
import os
from pathlib import Path

import faiss
import numpy as np
import torch
from neo4j import GraphDatabase
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

FETCH_EDGES_CYPHER = """
MATCH (s)-[r]->(o)
WHERE type(r) IN $predicates
RETURN
    s.id        AS subject_id,
    s.name      AS subject_name,
    type(r)     AS predicate,
    o.id        AS object_id,
    o.name      AS object_name,
    r.evidence_sentences AS sentences,
    r.evidence_pmids     AS pmids,
    r.confidence         AS confidence
"""

VALID_PREDICATES = [
    "UPREGULATES", "DOWNREGULATES", "ACTIVATES",
    "INHIBITS", "SUBSTRATE_OF", "INTERACTS_WITH", "ASSOCIATED_WITH",
]


def fetch_edges(driver) -> list[dict]:
    with driver.session() as session:
        result = session.run(FETCH_EDGES_CYPHER, predicates=VALID_PREDICATES)
        edges = []
        for record in result:
            sents = record["sentences"] or []
            pmids = record["pmids"] or []
            for sent, pmid in zip(sents, pmids):
                if sent.strip():
                    edges.append({
                        "subject_id": record["subject_id"],
                        "subject_name": record["subject_name"] or "",
                        "predicate": record["predicate"],
                        "object_id": record["object_id"],
                        "object_name": record["object_name"] or "",
                        "sentence": sent,
                        "doc_id": pmid,
                        "confidence": record["confidence"],
                    })
    logger.info("Fetched %d evidence sentences from Neo4j", len(edges))
    return edges


def _text_for_edge(edge: dict) -> str:
    """将三元组文本化为SapBERT编码输入（论文4.2.3节）。"""
    return (
        f"{edge['subject_name']} {edge['predicate'].lower().replace('_', ' ')} "
        f"{edge['object_name']}: {edge['sentence']}"
    )


def encode_texts(texts: list[str], tokenizer, model, device, batch_size: int) -> np.ndarray:
    all_vecs = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", unit="batch"):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch,
                max_length=256,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            # SapBERT: 取 [CLS] token 的隐状态作为句子向量
            vecs = out.last_hidden_state[:, 0, :].cpu().float().numpy()
            all_vecs.append(vecs)
    return np.vstack(all_vecs)


def build_index(vecs: np.ndarray) -> faiss.Index:
    """构建 FAISS 内积平面索引（L2归一化后内积 = 余弦相似度）。"""
    d = vecs.shape[1]  # 768
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(d)
    index.add(vecs)
    logger.info("FAISS index: %d vectors, dim=%d", index.ntotal, d)
    return index


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    edges = fetch_edges(driver)
    driver.close()

    if not edges:
        logger.error("No edges found in Neo4j. Run neo4j_ingestor first.")
        return

    # 保存索引元数据（向量下标 → 三元组信息）
    meta_path = output_dir / "index_meta.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for edge in edges:
            f.write(json.dumps(edge, ensure_ascii=False) + "\n")
    logger.info("Metadata saved: %s", meta_path)

    texts = [_text_for_edge(e) for e in edges]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading SapBERT on %s...", device)
    tokenizer = AutoTokenizer.from_pretrained(SAPBERT_MODEL)
    model = AutoModel.from_pretrained(SAPBERT_MODEL).to(device)

    logger.info("Encoding %d texts...", len(texts))
    vecs = encode_texts(texts, tokenizer, model, device, args.batch_size)

    index = build_index(vecs)
    index_path = output_dir / "index.bin"
    faiss.write_index(index, str(index_path))
    logger.info("FAISS index saved: %s", index_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", ""))
    parser.add_argument("--output-dir", default="data/faiss")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
