"""
互倒排名融合（Reciprocal Rank Fusion，RRF）（论文4.2.4节）

公式：score(e) = Σ_i  1 / (k + rank_i(e))
其中 k=60（经验常数），i 遍历各检索路径。

被多路径召回的证据得分累加，体现"多路验证"原则。
"""
from collections import defaultdict

from src.retrieval.graph_retriever import Evidence

RRF_K = 60  # 论文4.2.4节经验常数


def reciprocal_rank_fusion(
    *ranked_lists: list[Evidence],
    top_k: int = 10,
) -> list[Evidence]:
    """
    融合任意数量的有序证据列表，返回前 top_k 条。

    Args:
        *ranked_lists: 各检索路径产出的有序 Evidence 列表
        top_k:         最终返回条数（论文默认10）
    """
    scores: dict[str, float] = defaultdict(float)
    best_evidence: dict[str, Evidence] = {}

    for ranked in ranked_lists:
        for rank, ev in enumerate(ranked, start=1):
            key = ev.unique_key()
            scores[key] += 1.0 / (RRF_K + rank)
            # 保留 confidence 最高的那个 Evidence 对象（用于输出）
            if key not in best_evidence or ev.confidence > best_evidence[key].confidence:
                best_evidence[key] = ev

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)

    results = []
    for key in sorted_keys[:top_k]:
        ev = best_evidence[key]
        results.append(ev)

    return results
