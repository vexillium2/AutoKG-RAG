"""
基于图拓扑的结构化检索（论文4.2.2节）

策略：
  1. 以查询实体的 normalized_id 为种子节点
  2. 依次执行 1跳、2跳 邻居扩展
  3. 按 confidence 降序剪枝，返回 top-K 个三元组证据
"""
from dataclasses import dataclass

from neo4j import Driver

VALID_PREDICATES = [
    "UPREGULATES", "DOWNREGULATES", "ACTIVATES",
    "INHIBITS", "SUBSTRATE_OF", "INTERACTS_WITH", "ASSOCIATED_WITH",
]

# 1跳：直接邻居
_CYPHER_1HOP = """
MATCH (seed {id: $seed_id})-[r]->(neighbor)
WHERE type(r) IN $predicates
RETURN
    seed.id        AS subject_id,
    seed.name      AS subject_name,
    type(r)        AS predicate,
    neighbor.id    AS object_id,
    neighbor.name  AS object_name,
    r.confidence         AS confidence,
    r.evidence_pmids     AS pmids,
    r.evidence_sentences AS sentences
UNION
MATCH (neighbor)-[r]->(seed {id: $seed_id})
WHERE type(r) IN $predicates
RETURN
    neighbor.id    AS subject_id,
    neighbor.name  AS subject_name,
    type(r)        AS predicate,
    seed.id        AS object_id,
    seed.name      AS object_name,
    r.confidence         AS confidence,
    r.evidence_pmids     AS pmids,
    r.evidence_sentences AS sentences
ORDER BY confidence DESC
LIMIT $limit
"""

# 2跳：通过中间节点
_CYPHER_2HOP = """
MATCH (seed {id: $seed_id})-[r1]->(mid)-[r2]->(neighbor)
WHERE type(r1) IN $predicates AND type(r2) IN $predicates
  AND neighbor.id <> $seed_id
RETURN
    mid.id         AS subject_id,
    mid.name       AS subject_name,
    type(r2)       AS predicate,
    neighbor.id    AS object_id,
    neighbor.name  AS object_name,
    r2.confidence        AS confidence,
    r2.evidence_pmids    AS pmids,
    r2.evidence_sentences AS sentences
ORDER BY confidence DESC
LIMIT $limit
"""


@dataclass
class Evidence:
    subject_id: str
    subject_name: str
    predicate: str
    object_id: str
    object_name: str
    confidence: float
    pmids: list
    sentences: list
    source: str  # "graph-1hop" | "graph-2hop" | "vector"

    def unique_key(self) -> str:
        return f"{self.subject_id}|{self.predicate}|{self.object_id}"

    def to_text(self) -> str:
        """三元组文本化，用于 Prompt 构建（论文4.3.1节）。"""
        sent = self.sentences[0] if self.sentences else ""
        pmid = self.pmids[0] if self.pmids else ""
        return (
            f"[PMID:{pmid}] {self.subject_name} --{self.predicate}--> "
            f"{self.object_name}: {sent}"
        )


def retrieve_graph(
    driver: Driver,
    seed_ids: list[str],
    top_k: int = 10,
    hops: int = 2,
) -> list[Evidence]:
    """
    对每个种子实体执行图检索，合并去重后按 confidence 降序返回。

    Args:
        driver:   Neo4j driver 实例
        seed_ids: 查询实体归一化 ID 列表（来自 Kazu NER）
        top_k:    每个种子最多召回的三元组数（论文4.4.1节默认10）
        hops:     子图扩展跳数（论文默认2跳）
    """
    seen: dict[str, Evidence] = {}

    with driver.session() as session:
        for seed_id in seed_ids:
            # 1跳
            records = session.run(
                _CYPHER_1HOP,
                seed_id=seed_id,
                predicates=VALID_PREDICATES,
                limit=top_k,
            )
            for r in records:
                ev = _record_to_evidence(r, "graph-1hop")
                key = ev.unique_key()
                if key not in seen or ev.confidence > seen[key].confidence:
                    seen[key] = ev

            # 2跳（可选）
            if hops >= 2:
                records = session.run(
                    _CYPHER_2HOP,
                    seed_id=seed_id,
                    predicates=VALID_PREDICATES,
                    limit=top_k,
                )
                for r in records:
                    ev = _record_to_evidence(r, "graph-2hop")
                    key = ev.unique_key()
                    if key not in seen or ev.confidence > seen[key].confidence:
                        seen[key] = ev

    results = sorted(seen.values(), key=lambda e: e.confidence, reverse=True)
    return results[:top_k]


def _record_to_evidence(record, source: str) -> Evidence:
    return Evidence(
        subject_id=record["subject_id"] or "",
        subject_name=record["subject_name"] or "",
        predicate=record["predicate"] or "",
        object_id=record["object_id"] or "",
        object_name=record["object_name"] or "",
        confidence=float(record["confidence"] or 0.0),
        pmids=list(record["pmids"] or []),
        sentences=list(record["sentences"] or []),
        source=source,
    )
