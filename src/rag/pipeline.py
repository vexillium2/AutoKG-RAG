#!/usr/bin/env python3
"""
KG-RAG 端到端推理 pipeline（论文第4章）

流程：
  1. Kazu NER → 查询实体归一化 ID（种子节点）
  2. 图检索（1-2跳）+ 向量检索（FAISS）
  3. RRF 融合，取 top-K 证据
  4. 文本化证据 + 构建 Prompt
  5. Qwen3-8B-Instruct 生成答案

用法：
    python -m src.rag.pipeline \\
        --faiss-dir data/faiss \\
        --neo4j-uri bolt://localhost:7687 \\
        --neo4j-user neo4j --neo4j-password your_password \\
        --model Qwen/Qwen3-8B-Instruct \\
        --top-k 10 --hops 2 \\
        --question "What genes are associated with Alzheimer disease?"

    # 批量评测模式
    python -m src.rag.pipeline \\
        --bioasq-file data/bioasq/13B1_golden.json \\
        --output results/13B1_predictions.json \\
        ...
"""
import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import torch
from neo4j import GraphDatabase
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.retrieval.graph_retriever import Evidence, retrieve_graph
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.rrf import reciprocal_rank_fusion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------- Prompt 模板（论文4.3.1节）----------

SYSTEM_PROMPT = (
    "You are a biomedical expert assistant. "
    "Answer the question using the provided evidence. "
    "Cite evidence by number [1], [2], etc. at the end of each factual claim. "
    "If the evidence is insufficient, use your biomedical knowledge but mark such "
    "claims with [LLM]."
)

# 题型专用格式指令（确保答案格式与评测标准一致）
_QTYPE_INSTRUCTIONS = {
    "yesno": (
        "IMPORTANT: Your answer MUST begin with 'yes' or 'no' as the very first word "
        "(lowercase, no punctuation before it). Even if the evidence is insufficient, "
        "use your biomedical knowledge and still start with 'yes' or 'no'. "
        "Then provide a brief explanation."
    ),
    "factoid": (
        "IMPORTANT: Write the specific answer entity/term ALONE on the very first line "
        "(no articles, no 'The answer is', just the term or name, 1–6 words). "
        "Then explain on the following lines. Example first line: 'neurokinin 3 receptor'"
    ),
    "list": (
        "IMPORTANT: Your answer MUST be a numbered list, one specific entity per line:\n"
        "1. <entity name>\n2. <entity name>\n...\n"
        "Do NOT write prose paragraphs. Each line contains ONLY a specific entity, "
        "gene name, drug name, or term."
    ),
}

_EVIDENCE_TMPL = "[{idx}] {text}"


def build_prompt(question: str, evidence_list: list[Evidence], q_type: str = "") -> str:
    evidence_block = "\n".join(
        _EVIDENCE_TMPL.format(idx=i + 1, text=ev.to_text())
        for i, ev in enumerate(evidence_list)
    )
    instruction = _QTYPE_INSTRUCTIONS.get(q_type, "")
    instruction_block = f"\n{instruction}\n" if instruction else ""
    return (
        f"Evidence:\n{evidence_block}\n\n"
        f"Question: {question}\n"
        f"{instruction_block}"
        f"\nAnswer:"
    )


# ---------- Kazu 查询实体识别（复用 src/ner）----------

def _get_kazu_pipeline():
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from kazu.utils.constants import HYDRA_VERSION_BASE

    cdir = Path(os.environ["KAZU_MODEL_PACK"]) / "conf"
    with initialize_config_dir(config_dir=str(cdir), version_base=HYDRA_VERSION_BASE):
        cfg = compose(config_name="config")
    return instantiate(cfg.Pipeline)


def extract_seed_ids(question: str, kazu_pipeline) -> list[str]:
    """从查询文本中提取实体归一化 ID，作为图检索种子节点。"""
    from kazu.data import Document as KazuDoc
    doc = KazuDoc.create_simple_document(question)
    kazu_pipeline([doc])

    seed_ids = []
    for section in doc.sections:
        for entity in section.entities:
            ec = (entity.entity_class or "").lower()
            if ec not in {"gene", "disease", "drug"}:
                continue
            for mapping in (entity.mappings or []):
                nid = getattr(mapping, "idx", "")
                source = getattr(mapping, "source", "")
                if nid:
                    full_id = f"{source}:{nid}" if source and not nid.startswith(source) else nid
                    seed_ids.append(full_id)
                    break

    return list(set(seed_ids))


# ---------- Qwen3 生成（论文4.4.1节：零温度采样）----------

class Qwen3Generator:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B", load_in_8bit: bool = True):
        logger.info("Loading %s (8bit=%s)...", model_name, load_in_8bit)
        quant_config = BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model.eval()

    def generate(self, system: str, user: str, max_new_tokens: int = 512) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # enable_thinking=False：关闭 Qwen3 思考模式，RAG场景直接输出答案
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # 零温度（论文4.4.1节）
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------- 主 pipeline ----------

class KGRAGPipeline:
    def __init__(
        self,
        faiss_dir: str,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        model_name: str = "Qwen/Qwen3-8B",
        top_k: int = 10,
        hops: int = 2,
        load_in_8bit: bool = True,
        graph_only: bool = False,
        vector_only: bool = False,
        no_ner: bool = False,
    ):
        self.top_k = top_k
        self.hops = hops
        self.graph_only = graph_only
        self.vector_only = vector_only
        self.no_ner = no_ner

        if not graph_only:
            logger.info("Loading vector retriever...")
            self.vector_retriever = VectorRetriever(faiss_dir)
        else:
            logger.info("Graph-only mode: skipping vector retriever.")
            self.vector_retriever = None

        logger.info("Connecting to Neo4j...")
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_uri, auth=(neo4j_user, neo4j_password)
        )

        if not no_ner:
            logger.info("Loading Kazu NER pipeline...")
            self.kazu = _get_kazu_pipeline()
        else:
            logger.info("No-NER mode: skipping Kazu pipeline.")
            self.kazu = None

        logger.info("Loading Qwen3...")
        self.generator = Qwen3Generator(model_name, load_in_8bit=load_in_8bit)

    def answer(self, question: str, q_type: str = "") -> dict:
        # 1. 查询实体识别
        if self.no_ner:
            seed_ids = []
        else:
            seed_ids = extract_seed_ids(question, self.kazu)
        logger.debug("Seed IDs: %s", seed_ids)

        # 2. 检索（消融开关控制路径）
        if self.vector_only or self.no_ner:
            graph_ev = []
        elif seed_ids:
            graph_ev = retrieve_graph(
                self.neo4j_driver, seed_ids, top_k=self.top_k, hops=self.hops
            )
        else:
            graph_ev = []  # 无实体时跳过图检索（论文4.2.1节）

        if self.graph_only:
            fused = graph_ev[:self.top_k]
        else:
            vector_ev = self.vector_retriever.retrieve(question, top_k=self.top_k)
            if self.vector_only or self.no_ner:
                fused = vector_ev
            else:
                fused = reciprocal_rank_fusion(graph_ev, vector_ev, top_k=self.top_k)

        # 3. RRF 融合（graph_only 路径已在上方直接赋值，此处占位保持注释对齐）

        # 4. Prompt 构建（含题型专用格式指令）
        prompt = build_prompt(question, fused, q_type)

        # 5. 生成
        raw_answer = self.generator.generate(SYSTEM_PROMPT, prompt)

        return {
            "question": question,
            "answer": raw_answer,
            "seed_ids": seed_ids,
            "evidence_count": len(fused),
            "evidence": [
                {
                    "rank": i + 1,
                    "text": ev.to_text(),
                    "source": ev.source,
                    "confidence": ev.confidence,
                }
                for i, ev in enumerate(fused)
            ],
        }

    def close(self):
        self.neo4j_driver.close()


# ---------- CLI ----------

def run_single(args):
    pipeline = KGRAGPipeline(
        faiss_dir=args.faiss_dir,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        model_name=args.model,
        top_k=args.top_k,
        hops=args.hops,
        load_in_8bit=not args.no_8bit,
        graph_only=args.graph_only,
        vector_only=args.vector_only,
        no_ner=args.no_ner,
    )
    result = pipeline.answer(args.question)
    print("\n=== Answer ===")
    print(result["answer"])
    print(f"\n[{result['evidence_count']} evidence items, seeds: {result['seed_ids']}]")
    pipeline.close()


def run_batch(args):
    pipeline = KGRAGPipeline(
        faiss_dir=args.faiss_dir,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        model_name=args.model,
        top_k=args.top_k,
        hops=args.hops,
        load_in_8bit=not args.no_8bit,
        graph_only=args.graph_only,
        vector_only=args.vector_only,
        no_ner=args.no_ner,
    )

    with open(args.bioasq_file, encoding="utf-8") as f:
        data = json.load(f)

    questions = [
        q for q in data["questions"]
        if q["type"] in {"factoid", "list", "yesno"}
    ]
    logger.info("Running on %d questions...", len(questions))

    predictions = []
    for i, q in enumerate(questions):
        logger.info("[%d/%d] %s", i + 1, len(questions), q["body"][:80])
        result = pipeline.answer(q["body"], q_type=q["type"])
        predictions.append({
            "id": q["id"],
            "type": q["type"],
            "body": q["body"],
            "answer": result["answer"],
            "ideal_answer": q.get("ideal_answer", []),
            "exact_answer": q.get("exact_answer", []),
            "evidence_count": result["evidence_count"],
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    logger.info("Predictions saved: %s", output)
    pipeline.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--faiss-dir", default="data/faiss")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", ""))
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--no-8bit", action="store_true",
                        help="不使用8bit量化（需要更多VRAM）")
    # 消融实验开关
    parser.add_argument("--graph-only", action="store_true",
                        help="消融：仅图检索，跳过FAISS向量检索")
    parser.add_argument("--vector-only", action="store_true",
                        help="消融：仅向量检索，跳过图检索")
    parser.add_argument("--no-ner", action="store_true",
                        help="消融：跳过NER实体链接，使用纯文本向量检索")
    # 单题模式
    parser.add_argument("--question", default=None)
    # 批量模式
    parser.add_argument("--bioasq-file", default=None)
    parser.add_argument("--output", default="results/predictions.json")
    args = parser.parse_args()

    if args.question:
        run_single(args)
    elif args.bioasq_file:
        run_batch(args)
    else:
        parser.error("Provide --question or --bioasq-file")


if __name__ == "__main__":
    main()
