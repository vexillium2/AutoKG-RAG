#!/usr/bin/env python3
"""
BioBERT RE 全量推理脚本（论文3.4节 → Triple JSONL）

输入：
  - data/mentions/mentions.jsonl  (Kazu NER产出)
  - data/pubmed/*.jsonl           (Document JSONL)
输出：
  - data/triples/triples.jsonl    (Triple schema，论文表3-3)

推理策略：
  - 按文档重建句子边界，枚举句内类型约束实体对
  - 三个模型分别处理三类实体对：
      ChemProt 模型: Chemical ↔ Gene
      DDI 模型:      Chemical ↔ Chemical
      GAD 模型:      Gene ↔ Disease
  - 置信度阈值 0.6（论文3.4.3节）

用法：
    python -m src.re.infer \\
        --mention-file data/mentions/mentions.jsonl \\
        --doc-dir data/pubmed \\
        --output data/triples/triples.jsonl \\
        --chemprot-dir models/re_chemprot \\
        --ddi-dir models/re_ddi \\
        --gad-dir models/re_gad/final \\
        --batch-size 256
"""
import argparse
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.re.dataset import DATASET_INFO, _insert_markers
from src.re.model import BioBERTRelationClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.6

# 每个模型处理的实体对类型约束
# (e1_type, e2_type) → (model_key, 是否也检查反向对)
TYPE_TO_MODEL = {
    ("Chemical", "Gene"): ("chemprot", False),
    ("Gene", "Chemical"): ("chemprot", True),   # 反向，交换E1/E2
    ("Chemical", "Chemical"): ("ddi", False),
    ("Gene", "Disease"): ("gad", False),
    ("Disease", "Gene"): ("gad", True),
}


@dataclass
class Triple:
    subject_id: str
    predicate: str
    object_id: str
    doc_id: str
    sentence: str
    confidence: float
    model_version: str


# ---------- 句子分割 ----------

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def split_sentences(text: str) -> list[tuple[int, int]]:
    """返回每个句子的字符级 (start, end) 区间列表。"""
    spans = []
    prev = 0
    for m in _SENT_SPLIT.finditer(text):
        spans.append((prev, m.start() + 1))
        prev = m.end()
    spans.append((prev, len(text)))
    return [(s, e) for s, e in spans if e > s]


# ---------- 数据加载 ----------

def load_documents(doc_dir: Path) -> dict[str, dict]:
    """从 data/pubmed/*.jsonl 加载文档，返回 {doc_id: doc_dict}。"""
    docs = {}
    for fpath in sorted(doc_dir.glob("*.jsonl")):
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    docs[d["doc_id"]] = d
    return docs


def load_mentions(mention_file: Path) -> dict[str, list[dict]]:
    """按 doc_id 分组加载 Mention，返回 {doc_id: [mention, ...]}。"""
    by_doc = defaultdict(list)
    with open(mention_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                m = json.loads(line)
                by_doc[m["doc_id"]].append(m)
    return dict(by_doc)


# ---------- 候选对枚举 ----------

@dataclass
class Candidate:
    doc_id: str
    sentence: str
    e1_mention: dict
    e2_mention: dict
    e1_type: str
    e2_type: str
    model_key: str
    reversed_pair: bool  # True 表示原始对是 (e2_type, e1_type)


def enumerate_candidates(
    doc_id: str,
    body: str,
    mentions: list[dict],
) -> list[Candidate]:
    """在句子粒度内枚举所有类型约束实体对。"""
    sent_spans = split_sentences(body)
    candidates = []

    for sent_start, sent_end in sent_spans:
        sent_text = body[sent_start:sent_end]
        # 取在此句子内的 mention（字符偏移对齐）
        sent_mentions = [
            m for m in mentions
            if m["char_start"] >= sent_start and m["char_end"] <= sent_end
        ]
        if len(sent_mentions) < 2:
            continue

        for i, m1 in enumerate(sent_mentions):
            for m2 in sent_mentions[i + 1:]:
                t1 = m1["entity_type"]
                t2 = m2["entity_type"]

                for (et1, et2), (model_key, is_reversed) in TYPE_TO_MODEL.items():
                    if is_reversed and (t1 == et2 and t2 == et1):
                        candidates.append(Candidate(
                            doc_id=doc_id,
                            sentence=sent_text,
                            e1_mention=m2,  # 交换
                            e2_mention=m1,
                            e1_type=et1,
                            e2_type=et2,
                            model_key=model_key,
                            reversed_pair=True,
                        ))
                    elif not is_reversed and (t1 == et1 and t2 == et2):
                        candidates.append(Candidate(
                            doc_id=doc_id,
                            sentence=sent_text,
                            e1_mention=m1,
                            e2_mention=m2,
                            e1_type=et1,
                            e2_type=et2,
                            model_key=model_key,
                            reversed_pair=False,
                        ))

    return candidates


# ---------- 推理 Dataset ----------

class CandidateDataset(Dataset):
    def __init__(
        self,
        candidates: list[Candidate],
        tokenizer,
        max_length: int = 512,
    ):
        self.candidates = candidates
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, idx: int) -> dict:
        c = self.candidates[idx]
        sent = c.sentence

        # 将 mention 字符偏移转换为句子内偏移
        doc_start = c.e1_mention["char_start"]
        sent_offset = sent.find(sent.strip())  # 已是句子切片

        # 计算在句子内的偏移
        e1_s = c.e1_mention["char_start"] - (
            c.e1_mention["char_start"] - sent.find(c.e1_mention["surface"])
            if c.e1_mention["surface"] in sent else 0
        )
        # 简化：直接在句子中搜索 surface 字符串
        e1_surf = c.e1_mention["surface"]
        e2_surf = c.e2_mention["surface"]
        e1_pos_in_sent = sent.find(e1_surf)
        e2_pos_in_sent = sent.find(e2_surf)

        if e1_pos_in_sent == -1:
            e1_pos_in_sent = 0
        if e2_pos_in_sent == -1:
            e2_pos_in_sent = len(sent) // 2

        marked = _insert_markers(
            sent,
            e1_pos_in_sent, e1_pos_in_sent + len(e1_surf),
            e2_pos_in_sent, e2_pos_in_sent + len(e2_surf),
            c.e1_type, c.e2_type,
        )

        enc = self.tokenizer(
            marked,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        e1_token_id = self.tokenizer.convert_tokens_to_ids(f"[E1-{c.e1_type}]")
        e2_token_id = self.tokenizer.convert_tokens_to_ids(f"[E2-{c.e2_type}]")

        e1_tok_pos = (input_ids == e1_token_id).nonzero(as_tuple=True)[0]
        e2_tok_pos = (input_ids == e2_token_id).nonzero(as_tuple=True)[0]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "e1_pos": torch.tensor(e1_tok_pos[0].item() if len(e1_tok_pos) > 0 else 0),
            "e2_pos": torch.tensor(e2_tok_pos[0].item() if len(e2_tok_pos) > 0 else 0),
        }


# ---------- 模型加载 ----------

def load_model(model_dir: Path, device: torch.device) -> tuple:
    with open(model_dir / "meta.json") as f:
        meta = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    model = BioBERTRelationClassifier(
        model_name=meta.get("args", {}).get("model", "dmis-lab/biobert-base-cased-v1.1"),
        num_labels=meta["num_labels"],
    )
    model.resize_token_embeddings(len(tokenizer))
    state = torch.load(model_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, tokenizer, meta["labels"]


# ---------- 批量推理 ----------

def batch_infer(
    model,
    tokenizer,
    labels_list: list[str],
    candidates: list[Candidate],
    dataset_name: str,
    batch_size: int,
    device: torch.device,
    max_length: int = 512,
) -> list[Triple]:
    if not candidates:
        return []

    to_kg = DATASET_INFO[dataset_name]["to_kg"]
    ds = CandidateDataset(candidates, tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                e1_pos=batch["e1_pos"],
                e2_pos=batch["e2_pos"],
            )
            probs = torch.softmax(out["logits"], dim=-1).cpu()
            all_probs.append(probs)

    all_probs = torch.cat(all_probs, dim=0)  # (N, num_labels)
    triples = []

    for i, cand in enumerate(candidates):
        probs = all_probs[i]
        pred_label_idx = probs.argmax().item()
        pred_label = labels_list[pred_label_idx]
        confidence = probs[pred_label_idx].item()

        if pred_label == "NONE" or confidence < CONFIDENCE_THRESHOLD:
            continue
        if pred_label not in to_kg:
            continue

        kg_predicate = to_kg[pred_label]
        subject_id = cand.e1_mention["normalized_id"]
        object_id = cand.e2_mention["normalized_id"]

        if not subject_id or not object_id:
            continue

        triples.append(Triple(
            subject_id=subject_id,
            predicate=kg_predicate,
            object_id=object_id,
            doc_id=cand.doc_id,
            sentence=cand.sentence,
            confidence=round(confidence, 4),
            model_version=f"biobert-{dataset_name}-v1",
        ))

    return triples


# ---------- 主流程 ----------

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # 加载三个模型
    models = {}
    for key, model_dir_attr in [
        ("chemprot", args.chemprot_dir),
        ("ddi", args.ddi_dir),
        ("gad", args.gad_dir),
    ]:
        model_dir = Path(model_dir_attr)
        if model_dir.exists():
            logger.info("Loading %s model from %s", key, model_dir)
            models[key] = load_model(model_dir, device)
        else:
            logger.warning("Model dir not found: %s, skipping", model_dir)

    if not models:
        raise RuntimeError("No RE models found. Train first.")

    logger.info("Loading documents...")
    docs = load_documents(Path(args.doc_dir))
    logger.info("Loaded %d documents", len(docs))

    logger.info("Loading mentions...")
    mentions_by_doc = load_mentions(Path(args.mention_file))
    logger.info("Loaded mentions for %d documents", len(mentions_by_doc))

    args.output = Path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_triples = 0
    processed_docs = 0

    with open(args.output, "w", encoding="utf-8") as out_f:
        # 按模型分批处理，减少GPU切换
        for model_key, (model, tokenizer, labels_list) in models.items():
            logger.info("=== Running %s model ===", model_key)

            model_candidates: list[Candidate] = []
            candidate_sources = []  # 用于最终写入

            for doc_id, doc in docs.items():
                mentions = mentions_by_doc.get(doc_id, [])
                if len(mentions) < 2:
                    continue
                cands = enumerate_candidates(doc_id, doc["body"], mentions)
                model_cands = [c for c in cands if c.model_key == model_key]
                model_candidates.extend(model_cands)

            logger.info("%s candidates: %d", model_key, len(model_candidates))

            # 分批推理
            for i in range(0, len(model_candidates), args.batch_size * 10):
                chunk = model_candidates[i:i + args.batch_size * 10]
                triples = batch_infer(
                    model, tokenizer, labels_list, chunk,
                    model_key, args.batch_size, device,
                    max_length=args.max_length,
                )
                for t in triples:
                    out_f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
                total_triples += len(triples)

            logger.info("%s model → %d triples so far", model_key, total_triples)

    logger.info("Done. Total triples: %d → %s", total_triples, args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mention-file", default="data/mentions/mentions.jsonl")
    parser.add_argument("--doc-dir", default="data/pubmed")
    parser.add_argument("--output", default="data/triples/triples.jsonl")
    parser.add_argument("--chemprot-dir", default="models/re_chemprot")
    parser.add_argument("--ddi-dir", default="models/re_ddi")
    parser.add_argument("--gad-dir", default="models/re_gad/final")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="推理批大小，RTX3090建议256")
    parser.add_argument("--max-length", type=int, default=512,
                        help="tokenizer截断长度，须与训练时一致")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
