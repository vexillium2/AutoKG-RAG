#!/usr/bin/env python3
"""
知识图谱批量入库

输入：
  - data/triples/triples.jsonl   (RE推理产出)
  - data/mentions/mentions.jsonl  (节点embedding来源)

Neo4j 模式层：
  节点标签: Gene | Disease | Chemical
  节点属性: id, name, source_ontology, embedding (768维)
  边属性:   confidence, evidence_count, evidence_pmids[≤5], evidence_sentences[≤5], model_version

幂等设计：
  节点: MERGE on id → ON CREATE SET / ON MATCH SET
  边:   MERGE on (subject_id, predicate, object_id) → 累加evidence，保留最高confidence

用法：
    python -m src.kg.neo4j_ingestor \\
        --triple-file data/triples/triples.jsonl \\
        --mention-file data/mentions/mentions.jsonl \\
        --uri bolt://localhost:7687 \\
        --user neo4j --password your_password \\
        --batch-size 500
"""
import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

from neo4j import GraphDatabase
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 本体前缀 → 节点标签
ONTOLOGY_TO_LABEL = {
    "ENSEMBL": "Gene",
    "MONDO": "Disease",
    "CHEMBL": "Chemical",
    "CHEBI": "Chemical",
    "MESH": "Disease",
    "OMIM": "Disease",
    "HGNC": "Gene",
    "NCBIGENE": "Gene",
}

# 合法的关系谓词（防止动态Cypher中出现意外值）
VALID_PREDICATES = {
    "UPREGULATES", "DOWNREGULATES", "ACTIVATES",
    "INHIBITS", "SUBSTRATE_OF", "INTERACTS_WITH", "ASSOCIATED_WITH",
}

MAX_EVIDENCE = 5  # 每条边最多保留的证据数（论文表3-5）


_ENTITY_TYPE_TO_LABEL = {
    "Chemical": "Chemical",
    "Drug": "Chemical",
    "Gene": "Gene",
    "Disease": "Disease",
}


def _infer_label(normalized_id: str, entity_type: str = "") -> str:
    # 优先用 NER 直接输出的 entity_type，比前缀推断更可靠
    if entity_type and entity_type in _ENTITY_TYPE_TO_LABEL:
        return _ENTITY_TYPE_TO_LABEL[entity_type]
    prefix = normalized_id.split(":")[0].upper() if ":" in normalized_id else ""
    return ONTOLOGY_TO_LABEL.get(prefix, "Entity")


def _infer_ontology(normalized_id: str) -> str:
    return normalized_id.split(":")[0] if ":" in normalized_id else "UNKNOWN"


# ---------- 建库和约束 ----------

def setup_constraints(session):
    """建立唯一性约束和向量索引（Neo4j 5.x语法）。"""
    labels = ["Gene", "Disease", "Chemical"]
    for label in labels:
        session.run(
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
        )
    logger.info("Constraints ensured.")


# ---------- 节点入库 ----------

# 从 mentions 构建节点信息：{normalized_id: {name, source_ontology, embedding}}
def build_node_registry(mention_file: Path) -> dict[str, dict]:
    registry: dict[str, dict] = {}
    with open(mention_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            nid = m.get("normalized_id", "")
            if not nid or nid in registry:
                continue
            registry[nid] = {
                "id": nid,
                "name": m.get("surface", nid),
                "source_ontology": _infer_ontology(nid),
                "label": _infer_label(nid, m.get("entity_type", "")),
            }
    logger.info("Node registry: %d unique entities", len(registry))
    return registry


_MERGE_NODE_TMPL = """
MERGE (n:{label} {{id: $id}})
ON CREATE SET
    n.name            = $name,
    n.source_ontology = $source_ontology
ON MATCH SET
    n.name = coalesce(n.name, $name)
"""


def ingest_nodes(session, registry: dict[str, dict], batch_size: int = 500):
    nodes = list(registry.values())
    label_groups: dict[str, list] = defaultdict(list)
    for node in nodes:
        label_groups[node["label"]].append(node)

    total = 0
    for label, group in label_groups.items():
        cypher = _MERGE_NODE_TMPL.format(label=label)
        for i in range(0, len(group), batch_size):
            batch = group[i:i + batch_size]
            session.run(
                f"UNWIND $rows AS row {cypher.replace('$id', 'row.id').replace('$name', 'row.name').replace('$source_ontology', 'row.source_ontology')}",
                rows=[{"id": n["id"], "name": n["name"], "source_ontology": n["source_ontology"]} for n in batch],
            )
            total += len(batch)

    logger.info("Ingested %d nodes", total)


# ---------- 边入库 ----------

def _build_edge_merge_cypher(predicate: str) -> str:
    """
    生成特定谓词的 MERGE Cypher。
    谓词从合法枚举中取，不存在注入风险。
    """
    assert predicate in VALID_PREDICATES, f"Invalid predicate: {predicate}"
    return f"""
MATCH (s {{id: $subject_id}})
MATCH (o {{id: $object_id}})
MERGE (s)-[r:{predicate}]->(o)
ON CREATE SET
    r.confidence         = $confidence,
    r.evidence_count     = 1,
    r.evidence_pmids     = [$doc_id],
    r.evidence_sentences = [$sentence],
    r.model_version      = $model_version
ON MATCH SET
    r.evidence_count     = r.evidence_count + 1,
    r.confidence         = CASE WHEN $confidence > r.confidence
                           THEN $confidence ELSE r.confidence END,
    r.evidence_pmids     = CASE WHEN size(r.evidence_pmids) < {MAX_EVIDENCE}
                           THEN r.evidence_pmids + [$doc_id]
                           ELSE r.evidence_pmids END,
    r.evidence_sentences = CASE WHEN size(r.evidence_sentences) < {MAX_EVIDENCE}
                           THEN r.evidence_sentences + [$sentence]
                           ELSE r.evidence_sentences END
"""


def ingest_triples(session, triple_file: Path, batch_size: int = 500):
    # 先按谓词分组，减少 Cypher 切换
    by_predicate: dict[str, list] = defaultdict(list)

    with open(triple_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            pred = t.get("predicate", "")
            if pred not in VALID_PREDICATES:
                continue
            by_predicate[pred].append(t)

    total = 0
    for predicate, triples in by_predicate.items():
        cypher = _build_edge_merge_cypher(predicate)
        logger.info("Ingesting %s: %d triples", predicate, len(triples))
        for i in tqdm(range(0, len(triples), batch_size),
                      desc=predicate, unit="batch"):
            batch = triples[i:i + batch_size]
            for t in batch:
                session.run(
                    cypher,
                    subject_id=t["subject_id"],
                    object_id=t["object_id"],
                    confidence=t["confidence"],
                    doc_id=t["doc_id"],
                    sentence=t["sentence"][:500],  # 截断避免过长属性
                    model_version=t["model_version"],
                )
            total += len(batch)

    logger.info("Total edges ingested: %d", total)


# ---------- 统计 ----------

def print_kg_stats(session):
    for label in ["Gene", "Disease", "Chemical"]:
        count = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
        logger.info("  %s nodes: %d", label, count)

    for pred in VALID_PREDICATES:
        count = session.run(f"MATCH ()-[r:{pred}]->() RETURN count(r) AS c").single()["c"]
        logger.info("  %s edges: %d", pred, count)


# ---------- 主流程 ----------

def run(args):
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))

    with driver.session() as session:
        logger.info("Setting up constraints...")
        setup_constraints(session)

        logger.info("Building node registry from mentions...")
        registry = build_node_registry(Path(args.mention_file))

        logger.info("Ingesting nodes...")
        ingest_nodes(session, registry, batch_size=args.batch_size)

        logger.info("Ingesting triples (edges)...")
        ingest_triples(session, Path(args.triple_file), batch_size=args.batch_size)

        logger.info("=== KG Statistics ===")
        print_kg_stats(session)

    driver.close()
    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triple-file", default="data/triples/triples.jsonl")
    parser.add_argument("--mention-file", default="data/mentions/mentions.jsonl")
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", ""))
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
