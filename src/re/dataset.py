"""
RE 数据集加载器（论文3.4.2节）

支持三个数据集，通过 HuggingFace datasets 自动下载：
  - chemprot : Chemical-Protein 相互作用 (5类 + NONE)
  - ddi       : Drug-Drug 相互作用 (4类 + NONE)
  - gad       : Gene-Disease 关联 (二分类)

实体标记格式（论文3.4.1节）：
  [E1-Chemical] aspirin [/E1-Chemical] inhibits [E2-Gene] COX-2 [/E2-Gene]
"""
import io
import logging
import os
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

# ---------- 标签映射 ----------

# HF Hub Parquet 版本的 bigbio/chemprot 使用细粒度类型名（非原始 CPR:X 标签）
CHEMPROT_LABELS = ["NONE", "Upregulator", "Downregulator", "Agonist", "Antagonist", "Substrate"]
CHEMPROT_EVAL_GROUPS = {"Upregulator", "Downregulator", "Agonist", "Antagonist", "Substrate"}

# ChemProt 关系类型 → KG关系类型（用于推理阶段输出）
CHEMPROT_TO_KG = {
    "Upregulator": "UPREGULATES",
    "Downregulator": "DOWNREGULATES",
    "Agonist": "ACTIVATES",
    "Antagonist": "INHIBITS",
    "Substrate": "SUBSTRATE_OF",
}

DDI_LABELS = ["NONE", "advise", "effect", "mechanism", "int"]
DDI_TO_KG = {
    "advise": "INTERACTS_WITH",
    "effect": "INTERACTS_WITH",
    "mechanism": "INTERACTS_WITH",
    "int": "INTERACTS_WITH",
}

GAD_LABELS = ["NONE", "ASSOCIATED_WITH"]
GAD_TO_KG = {"ASSOCIATED_WITH": "ASSOCIATED_WITH"}

DATASET_INFO = {
    "chemprot": {
        "hf_name": "bigbio/chemprot",
        "hf_config": "chemprot_bigbio_kb",
        "labels": CHEMPROT_LABELS,
        "eval_labels": ["Upregulator", "Downregulator", "Agonist", "Antagonist", "Substrate"],
        "to_kg": CHEMPROT_TO_KG,
        "entity_types": ("Chemical", "Gene"),
        "class_boost": {"Substrate": 1.5},   # Substrate 最低频，逆频率权重基础上再×1.5
    },
    "ddi": {
        "hf_name": "bigbio/ddi_corpus",
        "hf_config": "ddi_corpus_bigbio_kb",
        "labels": DDI_LABELS,
        "eval_labels": ["advise", "effect", "mechanism", "int"],
        "to_kg": DDI_TO_KG,
        "entity_types": ("Chemical", "Chemical"),
    },
    "gad": {
        "hf_name": "bigbio/gad",
        "hf_config": "gad_fold0_bigbio_text",  # 10-fold，取fold0
        "labels": GAD_LABELS,
        "eval_labels": ["ASSOCIATED_WITH"],
        "to_kg": GAD_TO_KG,
        "entity_types": ("Gene", "Disease"),
    },
}


@dataclass
class REExample:
    text: str           # 插入实体标记后的句子
    label: int          # 标签索引
    e1_type: str
    e2_type: str
    # 以下字段仅推理时使用
    doc_id: str = ""
    e1_norm_id: str = ""
    e2_norm_id: str = ""
    sentence: str = ""  # 原始句子（证据句）


def _insert_markers(sentence: str, e1_start: int, e1_end: int,
                    e2_start: int, e2_end: int,
                    e1_type: str, e2_type: str) -> str:
    """在字符级别插入类型化实体标记（论文3.4.1节）。"""
    # 处理重叠或顺序：先插入靠后的实体，避免偏移错位
    e1_open = f"[E1-{e1_type}]"
    e1_close = f"[/E1-{e1_type}]"
    e2_open = f"[E2-{e2_type}]"
    e2_close = f"[/E2-{e2_type}]"

    spans = [
        (e1_start, e1_end, e1_open, e1_close),
        (e2_start, e2_end, e2_open, e2_close),
    ]
    spans.sort(key=lambda x: x[0])

    result = sentence
    offset = 0
    for start, end, open_tag, close_tag in spans:
        s = start + offset
        e = end + offset
        result = result[:s] + open_tag + result[s:e] + close_tag + result[e:]
        offset += len(open_tag) + len(close_tag)

    return result


# ---------- ChemProt ----------

def _load_chemprot(split: str) -> list[REExample]:
    ds = load_dataset("bigbio/chemprot", "chemprot_bigbio_kb")
    split_map = {"train": "train", "dev": "validation", "test": "test"}
    data = ds[split_map[split]]
    examples = []
    for item in data:
        text = " ".join(
            p["text"][0] if isinstance(p["text"], list) else p["text"]
            for p in item["passages"]
        )
        relations = item["relations"]

        # 已知关系对
        pos_pairs = {}
        for rel in relations:
            label_str = rel["type"]
            if label_str not in CHEMPROT_EVAL_GROUPS:
                continue
            key = (rel["arg1_id"], rel["arg2_id"])
            pos_pairs[key] = CHEMPROT_LABELS.index(label_str)

        # 枚举所有 Chemical-Gene 对（论文3.4.2节候选对策略）
        chemicals = [e for e in item["entities"] if e["type"] == "CHEMICAL"]
        genes = [e for e in item["entities"] if e["type"].startswith("GENE")]

        if not chemicals or not genes:
            continue

        for chem in chemicals:
            for gene in genes:
                key = (chem["id"], gene["id"])
                label = pos_pairs.get(key, 0)  # 0 = NONE

                c_start, c_end = chem["offsets"][0]
                g_start, g_end = gene["offsets"][0]
                marked = _insert_markers(
                    text, c_start, c_end, g_start, g_end, "Chemical", "Gene"
                )
                examples.append(REExample(text=marked, label=label,
                                          e1_type="Chemical", e2_type="Gene"))
    return examples


# ---------- DDI-2013 ----------

_DDI_CACHE_DIR = Path(os.environ.get("DDI_CACHE_DIR",
                      str(Path.home() / ".cache" / "biokg_rag" / "ddi_corpus")))
_DDI_URL = "https://github.com/isegura/DDICorpus/raw/master/DDICorpus-2013(BRAT).zip"
# bigbiohub.py 所在目录：克隆 https://github.com/bigbio/biomedical 后指向其根目录，
# 或设置 DDI_BIGBIOHUB_DIR 环境变量
_DDI_BIGBIOHUB_DIR = os.environ.get("DDI_BIGBIOHUB_DIR", str(Path.cwd()))
_DDI_DRUG_TYPES = {"DRUG", "DRUG_N", "BRAND", "GROUP"}
_DDI_VALID_TYPES = {"advise", "effect", "mechanism", "int"}


def _ensure_ddi_data() -> Path:
    brat_dir = _DDI_CACHE_DIR / "DDICorpusBrat"
    if brat_dir.exists():
        return brat_dir
    _DDI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = _DDI_CACHE_DIR / "ddi_corpus.zip"
    logger.info("Downloading DDI Corpus BRAT data from %s ...", _DDI_URL)
    with urllib.request.urlopen(_DDI_URL) as resp, open(zip_path, "wb") as f:
        f.write(resp.read())
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(_DDI_CACHE_DIR)
    zip_path.unlink()
    return brat_dir


def _load_ddi(split: str) -> list[REExample]:
    if _DDI_BIGBIOHUB_DIR not in sys.path:
        sys.path.insert(0, _DDI_BIGBIOHUB_DIR)
    from bigbiohub import parse_brat_file, brat_parse_to_bigbio_kb  # type: ignore

    brat_dir = _ensure_ddi_data()
    split_map = {"train": "Train", "dev": "Test", "test": "Test"}
    split_dir = brat_dir / split_map[split]

    examples = []
    for txt_file in sorted(split_dir.rglob("*.txt")):
        brat_example = parse_brat_file(txt_file)
        item = brat_parse_to_bigbio_kb(brat_example)

        text = " ".join(
            p["text"][0] if isinstance(p["text"], list) else p["text"]
            for p in item["passages"]
        )
        drugs = [e for e in item["entities"] if e["type"] in _DDI_DRUG_TYPES]

        pos_pairs = {}
        for rel in item["relations"]:
            rel_type = rel["type"].replace("DDI-", "").lower()
            if rel_type not in _DDI_VALID_TYPES:
                continue
            key = (rel["arg1_id"], rel["arg2_id"])
            pos_pairs[key] = DDI_LABELS.index(rel_type)

        for i, d1 in enumerate(drugs):
            for d2 in drugs[i + 1:]:
                key = (d1["id"], d2["id"])
                key_rev = (d2["id"], d1["id"])
                label = pos_pairs.get(key, pos_pairs.get(key_rev, 0))

                d1_start, d1_end = d1["offsets"][0]
                d2_start, d2_end = d2["offsets"][0]
                marked = _insert_markers(
                    text, d1_start, d1_end, d2_start, d2_end, "Chemical", "Chemical"
                )
                examples.append(REExample(text=marked, label=label,
                                          e1_type="Chemical", e2_type="Chemical"))
    return examples


# ---------- GAD ----------

def _load_gad(split: str, fold: int = 0) -> list[REExample]:
    from huggingface_hub import hf_hub_download  # type: ignore
    zip_path = hf_hub_download(
        repo_id="bigbio/gad", filename="data/REdata.zip", repo_type="dataset"
    )
    # zip内目录编号从1开始，fold参数从0开始
    folder = str(fold + 1)
    split_map = {"train": "train", "dev": "dev", "test": "test"}
    tsv_name = f"GAD/{folder}/{split_map[split]}.tsv"

    examples = []
    with zipfile.ZipFile(zip_path) as z:
        with io.TextIOWrapper(z.open(tsv_name), encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if "\t" not in line:
                    continue
                text, label_str = line.rsplit("\t", 1)
                if not label_str.strip().lstrip("-").isdigit():
                    continue
                label = int(label_str.strip())
                text = text.replace("@GENE$", "[E1-Gene] GENE [/E1-Gene]")
                text = text.replace("@DISEASE$", "[E2-Disease] DISEASE [/E2-Disease]")
                examples.append(REExample(
                    text=text,
                    label=label,
                    e1_type="Gene",
                    e2_type="Disease",
                ))
    return examples


# ---------- 统一接口 ----------

LOADERS = {
    "chemprot": _load_chemprot,
    "ddi": _load_ddi,
    "gad": _load_gad,
}


def load_re_examples(dataset_name: str, split: str) -> list[REExample]:
    loader = LOADERS[dataset_name]
    return loader(split)


# ---------- PyTorch Dataset ----------

class REDataset(Dataset):
    def __init__(
        self,
        examples: list[REExample],
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 512,
        neg_ratio: Optional[float] = 2.0,
    ):
        if neg_ratio is not None:
            examples = _downsample_negatives(examples, neg_ratio)
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # 定位 E1 和 E2 起始标记的位置（用于分类头，论文3.4.1节）
        e1_token_id = self.tokenizer.convert_tokens_to_ids(f"[E1-{ex.e1_type}]")
        e2_token_id = self.tokenizer.convert_tokens_to_ids(f"[E2-{ex.e2_type}]")

        e1_pos = (input_ids == e1_token_id).nonzero(as_tuple=True)[0]
        e2_pos = (input_ids == e2_token_id).nonzero(as_tuple=True)[0]

        e1_pos = e1_pos[0].item() if len(e1_pos) > 0 else 0
        e2_pos = e2_pos[0].item() if len(e2_pos) > 0 else 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "e1_pos": torch.tensor(e1_pos, dtype=torch.long),
            "e2_pos": torch.tensor(e2_pos, dtype=torch.long),
            "label": torch.tensor(ex.label, dtype=torch.long),
        }


def _downsample_negatives(
    examples: list[REExample], ratio: float
) -> list[REExample]:
    """按 ratio:1 下采样负样本（论文3.4.2节）。"""
    pos = [e for e in examples if e.label != 0]
    neg = [e for e in examples if e.label == 0]

    max_neg = int(len(pos) * ratio)
    if len(neg) > max_neg:
        import random
        random.seed(42)
        neg = random.sample(neg, max_neg)

    return pos + neg


def get_special_tokens(dataset_name: str) -> list[str]:
    """返回该数据集需要添加到 tokenizer 词表的特殊标记。"""
    info = DATASET_INFO[dataset_name]
    e1_type, e2_type = info["entity_types"]
    return [
        f"[E1-{e1_type}]", f"[/E1-{e1_type}]",
        f"[E2-{e2_type}]", f"[/E2-{e2_type}]",
    ]
