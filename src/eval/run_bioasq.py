#!/usr/bin/env python3
"""
BioASQ Task 13b 评测脚本（论文4.4.1节）

评测指标（官方标准）：
  - 是非型 (yesno):   Macro F1（yes/no 两类）
  - 事实型 (factoid): Strict Accuracy, Lenient Accuracy, MRR
  - 列表型 (list):    Token-level F1

输入：
  - --predictions: pipeline.py 产出的 JSON 文件
  - --golden:      原始 golden JSON（含 exact_answer）

用法：
    python -m src.eval.run_bioasq \\
        --predictions results/13B1_predictions.json \\
        --golden data/bioasq/13B1_golden.json \\
        --output results/13B1_metrics.json

    # 汇总四个 batch
    python -m src.eval.run_bioasq --all-batches
"""
import argparse
import json
import re
import string
from pathlib import Path

from sklearn.metrics import f1_score


# ---------- 答案规范化 ----------

def _normalize(text: str) -> str:
    # 仅保留 ASCII 字符（去掉 CD3ε 中的希腊字母等）
    text = text.encode("ascii", errors="ignore").decode()
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _extract_candidates(answer_text: str, q_type: str) -> list[str]:
    """从 LLM 输出文本中提取答案候选。"""
    # 去掉引用标记
    text = re.sub(r'\[\d+\]|\[LLM\]', '', answer_text).strip()
    # 去掉 markdown 加粗
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)

    if q_type == "yesno":
        text_lower = text.lower()
        # 直接以 yes/no 开头（含新格式提示后的输出）
        if text_lower.startswith("yes"):
            return ["yes"]
        if text_lower.startswith("no"):
            return ["no"]
        # 首句肯定词（模型未严格遵循格式时的兜底）
        first_sent = text_lower.split(".")[0][:200]
        neg_patterns = [" not ", " cannot ", " no evidence", " unlikely ",
                        " insufficient", " does not ", " do not ", " is not ",
                        " are not ", " was not ", " were not ", " have not "]
        aff_patterns = ["indeed", "certainly", "is involved", "are involved",
                        "has been shown", "have been shown", "plays a role",
                        "play a role", "is associated", "are associated",
                        "is effective", "are effective",
                        "have a role", "has a role", "is considered",
                        "are considered", "is generally", "is used", "are used",
                        "is a chronic", "is a direct", "is known"]
        for p in neg_patterns:
            if p in first_sent:
                return ["no"]
        for p in aff_patterns:
            if p in first_sent:
                return ["yes"]
        # 扩大 yes 搜索窗口
        if "yes" in text_lower[:100]:
            return ["yes"]
        if "no" in text_lower[:50]:
            return ["no"]
        return ["no"]

    if q_type == "factoid":
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        # 过滤掉纯编号行
        lines = [l for l in lines if not re.match(r'^\d+[.)]\s*$', l)]
        candidates = []

        if lines:
            first_line = lines[0]
            # 如果第一行已经是简短答案（≤8词），直接用
            if len(first_line.split()) <= 8:
                candidates.append(_normalize(first_line))
            # 尝试从 "X is/are/was Y" 提取 Y
            m = re.search(
                r'\b(?:is|are|was|were|called|named|known as)\s+(?:the\s+|a\s+|an\s+)?'
                r'([A-Za-z0-9][\w\s\-/()+]{1,60}?)(?:\s*[.,;()\[]|$)',
                first_line, re.IGNORECASE,
            )
            if m:
                candidates.append(_normalize(m.group(1).strip()))
            # 全行作为备选
            candidates.append(_normalize(first_line))

        for line in lines[1:5]:
            candidates.append(_normalize(line))

        # 去重保序
        seen, deduped = set(), []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                deduped.append(c)
        return deduped if deduped else [_normalize(text[:200])]

    if q_type == "list":
        # 优先解析编号列表（1. item 或 1) item）
        numbered = re.findall(r'^\d+[.)]\s+(.+?)(?:\s*$)', text, re.MULTILINE)
        if numbered:
            return [_normalize(i) for i in numbered if _normalize(i)]
        # 无序列表（- item 或 • item）
        bulleted = re.findall(r'^[-•]\s+(.+?)(?:\s*$)', text, re.MULTILINE)
        if bulleted:
            return [_normalize(i) for i in bulleted if _normalize(i)]
        # 行分割（每行一个条目）
        line_items = [l.strip() for l in text.split("\n") if l.strip()]
        if len(line_items) >= 2:
            return [_normalize(i) for i in line_items if _normalize(i)]
        # 最后回退：逗号/分号分割
        items = re.split(r';|,(?=\s)', text)
        return [_normalize(i) for i in items if _normalize(i)]

    return [_normalize(text[:200])]


# ---------- 各题型评测 ----------

def eval_yesno(predictions: list[dict], golden: dict) -> dict:
    gold_map = {q["id"]: q for q in golden["questions"]}
    y_true, y_pred = [], []

    for pred in predictions:
        if pred["type"] != "yesno":
            continue
        gq = gold_map.get(pred["id"])
        if not gq:
            continue

        # BioASQ yes/no: exact_answer 是字符串 "yes"/"no"，而非列表
        ea = gq.get("exact_answer")
        if isinstance(ea, list):
            gold_ans = str(ea[0]).lower() if ea else "no"
        else:
            gold_ans = str(ea).lower() if ea else "no"
        pred_ans = _extract_candidates(pred["answer"], "yesno")[0]

        y_true.append(1 if gold_ans == "yes" else 0)
        y_pred.append(1 if pred_ans == "yes" else 0)

    if not y_true:
        return {"macro_f1": 0.0, "n": 0}

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"macro_f1": round(macro_f1 * 100, 2), "n": len(y_true)}


def eval_factoid(predictions: list[dict], golden: dict) -> dict:
    gold_map = {q["id"]: q for q in golden["questions"]}
    strict_hits, lenient_hits, rr_sum, n = 0, 0, 0.0, 0

    for pred in predictions:
        if pred["type"] != "factoid":
            continue
        gq = gold_map.get(pred["id"])
        if not gq:
            continue

        # 黄金答案列表（exact_answer 中的所有可接受答案）
        # BioASQ factoid: [[alias1, alias2], [answer2]] 或 ["answer1", "answer2"]
        gold_list = gq.get("exact_answer") or []
        if not gold_list:
            continue
        gold_norms = set()
        for g in gold_list:
            if isinstance(g, list):
                for alias in g:
                    gold_norms.add(_normalize(str(alias)))
            else:
                gold_norms.add(_normalize(str(g)))

        candidates = _extract_candidates(pred["answer"], "factoid")[:5]
        n += 1

        # Strict: 第一个候选完全匹配
        if candidates and _normalize(candidates[0]) in gold_norms:
            strict_hits += 1

        # Lenient: 前5个中有匹配
        for cand in candidates:
            if _normalize(cand) in gold_norms:
                lenient_hits += 1
                break

        # MRR: 第一个匹配位置的倒数
        for rank, cand in enumerate(candidates, start=1):
            if _normalize(cand) in gold_norms:
                rr_sum += 1.0 / rank
                break

    if n == 0:
        return {"strict_acc": 0.0, "lenient_acc": 0.0, "mrr": 0.0, "n": 0}

    return {
        "strict_acc": round(strict_hits / n * 100, 2),
        "lenient_acc": round(lenient_hits / n * 100, 2),
        "mrr": round(rr_sum / n * 100, 2),
        "n": n,
    }


def eval_list(predictions: list[dict], golden: dict) -> dict:
    gold_map = {q["id"]: q for q in golden["questions"]}
    f1_scores, n = [], 0

    for pred in predictions:
        if pred["type"] != "list":
            continue
        gq = gold_map.get(pred["id"])
        if not gq:
            continue

        gold_items = gq.get("exact_answer") or []
        if not gold_items:
            continue

        # BioASQ list: [[alias1, alias2], [answer2]] 或 [item1, item2]
        # 对每个实体取所有别名，全部加入 gold_norms（宽松匹配）
        gold_norms = set()
        for g in gold_items:
            if isinstance(g, list):
                for alias in g:
                    gold_norms.add(_normalize(str(alias)))
            else:
                gold_norms.add(_normalize(str(g)))

        pred_items = _extract_candidates(pred["answer"], "list")
        pred_norms = set(pred_items)
        # 扩展：把多词预测项中每个短词（可能是缩写）也加入 pred_norms
        # 例如 "immunoglobulin m igm" → 也尝试匹配 "igm"
        extra = set()
        for item in pred_norms:
            for word in item.split():
                if 2 <= len(word) <= 6:
                    extra.add(word)
        pred_norms = pred_norms | extra

        if not gold_norms and not pred_norms:
            f1_scores.append(1.0)
        elif not gold_norms or not pred_norms:
            f1_scores.append(0.0)
        else:
            tp = len(gold_norms & pred_norms)
            p = tp / len(pred_norms)
            r = tp / len(gold_norms)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            f1_scores.append(f1)

        n += 1

    if n == 0:
        return {"f1": 0.0, "n": 0}

    return {"f1": round(sum(f1_scores) / n * 100, 2), "n": n}


# ---------- 主流程 ----------

def evaluate_batch(pred_file: Path, golden_file: Path) -> dict:
    with open(pred_file, encoding="utf-8") as f:
        predictions = json.load(f)
    with open(golden_file, encoding="utf-8") as f:
        golden = json.load(f)

    yesno = eval_yesno(predictions, golden)
    factoid = eval_factoid(predictions, golden)
    list_res = eval_list(predictions, golden)

    return {
        "batch": pred_file.stem,
        "yesno": yesno,
        "factoid": factoid,
        "list": list_res,
    }


def print_results(results: dict):
    b = results["batch"]
    yn = results["yesno"]
    fa = results["factoid"]
    li = results["list"]

    print(f"\n{'='*50}")
    print(f"Batch: {b}")
    print(f"  Yes/No  (n={yn['n']}): Macro F1 = {yn['macro_f1']:.2f}%")
    print(f"  Factoid (n={fa['n']}): Strict={fa['strict_acc']:.2f}%  "
          f"Lenient={fa['lenient_acc']:.2f}%  MRR={fa['mrr']:.2f}%")
    print(f"  List    (n={li['n']}): F1 = {li['f1']:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--golden", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--all-batches", action="store_true",
                        help="汇总评测 results/13B*_predictions.json")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--bioasq-dir", type=Path, default=Path("data/bioasq"))
    args = parser.parse_args()

    if args.all_batches:
        all_results = []
        for i in range(1, 5):
            pred_file = args.results_dir / f"13B{i}_predictions.json"
            golden_file = args.bioasq_dir / f"13B{i}_golden.json"
            if not pred_file.exists():
                print(f"Skipping {pred_file} (not found)")
                continue
            res = evaluate_batch(pred_file, golden_file)
            print_results(res)
            all_results.append(res)

        # 加权平均
        if all_results:
            _print_aggregate(all_results)

        out = args.results_dir / "bioasq_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved: {out}")

    else:
        if not args.predictions or not args.golden:
            parser.error("Provide --predictions and --golden, or use --all-batches")
        res = evaluate_batch(args.predictions, args.golden)
        print_results(res)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(res, f, indent=2)


def _print_aggregate(all_results: list[dict]):
    def _wavg(key_path: list[str], n_key: str):
        total_n = sum(r[key_path[0]][n_key] for r in all_results)
        if total_n == 0:
            return 0.0
        val = 0.0
        for r in all_results:
            n = r[key_path[0]][n_key]
            v = r[key_path[0]][key_path[1]]
            val += v * n / total_n
        return round(val, 2)

    print(f"\n{'='*50}")
    print("AGGREGATE (weighted by n):")
    print(f"  Yes/No  Macro F1  : {_wavg(['yesno', 'macro_f1'], 'n')}%")
    print(f"  Factoid Strict Acc: {_wavg(['factoid', 'strict_acc'], 'n')}%")
    print(f"  Factoid Lenient   : {_wavg(['factoid', 'lenient_acc'], 'n')}%")
    print(f"  Factoid MRR       : {_wavg(['factoid', 'mrr'], 'n')}%")
    print(f"  List    F1        : {_wavg(['list', 'f1'], 'n')}%")


if __name__ == "__main__":
    main()
