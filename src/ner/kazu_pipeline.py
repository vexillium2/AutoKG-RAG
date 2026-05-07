#!/usr/bin/env python3
"""
基于 Kazu v2.3.0 的 NER + 归一化 pipeline（论文3.3节实现）。

输入：Document JSONL（src/data/fetch_pubmed.py 产出）
输出：Mention JSONL，字段见论文表3-2

用法：
    export KAZU_MODEL_PACK=/path/to/kazu_model_pack
    python -m src.ner.kazu_pipeline \\
        --input-dir data/pubmed \\
        --output data/mentions/mentions.jsonl \\
        --batch-size 32 \\
        --workers 4
"""
import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import hydra
from hydra.utils import instantiate
from kazu.data import Document as KazuDoc, Entity
from kazu.pipeline import Pipeline
from kazu.utils.constants import HYDRA_VERSION_BASE
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 保留的实体类型（论文3.3.2节，去除细胞系/细胞类型/物种）
KEPT_ENTITY_CLASSES = {"gene", "disease", "drug"}

# Kazu entity_class → 论文中实体类型名称
CLASS_MAP = {
    "gene": "Gene",
    "disease": "Disease",
    "drug": "Chemical",
}

# Kazu uses StringMatchConfidence / DisambiguationConfidence enums instead of floats
_CONF_RANK = {"HIGHLY_LIKELY": 2, "POSSIBLE": 1}


def _mapping_score(mapping) -> float:
    smc = getattr(mapping, "string_match_confidence", None)
    dc = getattr(mapping, "disambiguation_confidence", None)
    s = _CONF_RANK.get(getattr(smc, "value", ""), 0)
    d = _CONF_RANK.get(getattr(dc, "value", ""), 0)
    return (s * 2 + d) / 6.0  # normalise to (0, 1]


@dataclass
class Mention:
    doc_id: str
    char_start: int
    char_end: int
    surface: str
    entity_type: str
    normalized_id: str
    confidence: float


def _load_pipeline() -> Pipeline:
    cdir = Path(os.environ["KAZU_MODEL_PACK"]).joinpath("conf")

    @hydra.main(version_base=HYDRA_VERSION_BASE, config_path=str(cdir), config_name="config")
    def _build(cfg):
        return instantiate(cfg.Pipeline)

    # hydra.main 装饰器直接调用会返回None；用 compose API 更稳定
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(cdir), version_base=HYDRA_VERSION_BASE):
        cfg = compose(config_name="config")
    return instantiate(cfg.Pipeline)


def _iter_documents(input_dir: Path) -> Iterator[dict]:
    for fpath in sorted(input_dir.glob("*.jsonl")):
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _extract_mentions(doc_id: str, kazu_doc: KazuDoc) -> list[Mention]:
    mentions = []
    for section in kazu_doc.sections:
        for entity in section.entities:
            ec = (entity.entity_class or "").lower()
            if ec not in KEPT_ENTITY_CLASSES:
                continue

            mappings = list(entity.mappings) if entity.mappings else []
            if not mappings:
                continue

            best_mapping = max(mappings, key=_mapping_score)
            best_conf = _mapping_score(best_mapping)

            norm_id = getattr(best_mapping, "idx", "") or ""
            source = getattr(best_mapping, "source", "") or ""
            if source and norm_id and not norm_id.startswith(source):
                norm_id = f"{source}:{norm_id}"

            mentions.append(Mention(
                doc_id=doc_id,
                char_start=entity.start,
                char_end=entity.end,
                surface=entity.match,
                entity_type=CLASS_MAP.get(ec, ec),
                normalized_id=norm_id,
                confidence=round(best_conf, 4),
            ))

    return mentions


def run(
    input_dir: Path,
    output_path: Path,
    batch_size: int,
) -> None:
    pipeline = _load_pipeline()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_docs = list(_iter_documents(input_dir))
    logger.info("Total documents to process: %d", len(all_docs))

    if len(all_docs) == 0:
        logger.error(
            "No documents found in %s\n"
            "  Run step 02 first: bash scripts/02_fetch_pubmed.sh\n"
            "  Then retry: bash scripts/03_run_ner.sh",
            input_dir,
        )
        return

    total_mentions = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for i in tqdm(range(0, len(all_docs), batch_size), desc="NER batches"):
            batch = all_docs[i:i + batch_size]
            kazu_docs = [
                KazuDoc.create_simple_document(d["body"])
                for d in batch
            ]
            pipeline(kazu_docs)

            for raw_doc, kazu_doc in zip(batch, kazu_docs):
                mentions = _extract_mentions(raw_doc["doc_id"], kazu_doc)
                for m in mentions:
                    out_f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")
                total_mentions += len(mentions)

    logger.info("Done. Total mentions: %d", total_mentions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/pubmed"))
    parser.add_argument("--output", type=Path, default=Path("data/mentions/mentions.jsonl"))
    parser.add_argument("--batch-size", type=int, default=32,
                        help="GPU批大小，RTX4080建议32")
    args = parser.parse_args()

    run(args.input_dir, args.output, args.batch_size)


if __name__ == "__main__":
    main()
