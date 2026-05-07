#!/usr/bin/env python3
"""
BioBERT RE 微调训练脚本（论文3.4.3节）

用法：
    # ChemProt
    python -m src.re.train --dataset chemprot --output-dir models/re_chemprot

    # DDI-2013
    python -m src.re.train --dataset ddi --output-dir models/re_ddi

    # GAD（10折交叉验证）
    python -m src.re.train --dataset gad --output-dir models/re_gad --cv-folds 10
"""
import argparse
import json
import logging
import math
from pathlib import Path


class PathEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Path):
            return str(o)
        return super().default(o)

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score

from src.re.dataset import (
    REDataset, DATASET_INFO, get_special_tokens,
    load_re_examples, _load_gad,
)
from src.re.model import BioBERTRelationClassifier, build_class_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DEFAULT_BIOBERT = "dmis-lab/biobert-base-cased-v1.1"


def evaluate(model, loader, device, labels_list, eval_label_names: list[str] | None = None):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                e1_pos=batch["e1_pos"],
                e2_pos=batch["e2_pos"],
            )
            preds = out["logits"].argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].cpu().tolist())

    # 使用 DATASET_INFO 中显式指定的 eval_labels，避免将冷门标签混入 Micro-F1
    if eval_label_names is not None:
        pos_labels = [labels_list.index(l) for l in eval_label_names if l in labels_list]
    else:
        pos_labels = list(range(1, len(labels_list)))

    micro_f1  = f1_score(all_labels, all_preds, labels=pos_labels, average="micro", zero_division=0)
    precision = precision_score(all_labels, all_preds, labels=pos_labels, average="micro", zero_division=0)
    recall    = recall_score(all_labels, all_preds, labels=pos_labels, average="micro", zero_division=0)

    # 每类 F1，便于诊断哪类学不到
    per_class = f1_score(all_labels, all_preds, labels=pos_labels, average=None, zero_division=0)
    per_class_info = {
        (eval_label_names or labels_list[1:])[i]: round(float(per_class[i]), 4)
        for i in range(len(pos_labels))
    }

    return {"f1": micro_f1, "precision": precision, "recall": recall,
            "per_class_f1": per_class_info}


def train_one_fold(
    dataset_name: str,
    train_examples,
    dev_examples,
    output_dir: Path,
    args,
) -> dict:
    info = DATASET_INFO[dataset_name]
    labels_list = info["labels"]
    eval_label_names = info.get("eval_labels")   # 显式评估类别，None 则用所有非NONE类
    num_labels = len(labels_list)
    special_tokens = get_special_tokens(dataset_name)

    biobert_model = getattr(args, "model", _DEFAULT_BIOBERT)
    tokenizer = AutoTokenizer.from_pretrained(biobert_model)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    neg_ratio = None if args.neg_ratio < 0 else args.neg_ratio
    train_ds = REDataset(train_examples, tokenizer, args.max_length, neg_ratio=neg_ratio)
    dev_ds   = REDataset(dev_examples,   tokenizer, args.max_length, neg_ratio=neg_ratio)

    if len(train_ds) == 0 or len(dev_ds) == 0:
        logger.warning("Empty dataset after sampling — skip %s (train=%d dev=%d)",
                       dataset_name, len(train_ds), len(dev_ds))
        return None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = BioBERTRelationClassifier(biobert_model, num_labels=num_labels)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # 类别权重（论文3.4.3节）；支持数据集级别的额外乘数（如 Substrate 1.5×）
    all_train_labels = [e.label for e in train_ds.examples]
    class_boost = info.get("class_boost", {})
    class_weights = build_class_weights(
        all_train_labels, num_labels, device, labels_list, class_boost
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = math.ceil(total_steps * 0.06)   # 6% warmup，全量数据收敛更稳

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_f1 = 0.0
    best_state = None
    history = []
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                e1_pos=batch["e1_pos"],
                e2_pos=batch["e2_pos"],
                labels=batch["label"],
                class_weights=class_weights,
            )
            loss = out["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if step % 100 == 0:
                logger.info("Epoch %d step %d/%d loss=%.4f",
                            epoch, step, len(train_loader), total_loss / step)

        metrics = evaluate(model, dev_loader, device, labels_list, eval_label_names)
        avg_loss = total_loss / len(train_loader)
        logger.info(
            "Epoch %d | loss=%.4f | P=%.4f R=%.4f F1=%.4f | per-class: %s",
            epoch, avg_loss, metrics["precision"], metrics["recall"], metrics["f1"],
            metrics["per_class_f1"],
        )
        history.append({"epoch": epoch, "loss": avg_loss,
                        "f1": metrics["f1"], "precision": metrics["precision"],
                        "recall": metrics["recall"], "per_class_f1": metrics["per_class_f1"]})

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)",
                            epoch, args.patience)
                break

    # 保存最优模型
    output_dir.mkdir(parents=True, exist_ok=True)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.cpu()
    torch.save(model.state_dict(), output_dir / "model.pt")
    tokenizer.save_pretrained(output_dir / "tokenizer")

    # --- 最终用完整 dev set（neg_ratio=None）再跑一次，得到可与 SOTA 对比的指标 ---
    full_dev_ds = REDataset(dev_examples, tokenizer, args.max_length, neg_ratio=None)
    full_dev_loader = DataLoader(full_dev_ds, batch_size=args.batch_size * 2,
                                 shuffle=False, num_workers=4, pin_memory=True)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.to(device)
    full_metrics = evaluate(model, full_dev_loader, device, labels_list, eval_label_names)
    logger.info(
        "FULL dev (SOTA-comparable) | P=%.4f R=%.4f F1=%.4f | per-class: %s",
        full_metrics["precision"], full_metrics["recall"], full_metrics["f1"],
        full_metrics["per_class_f1"],
    )

    meta = {
        "dataset": dataset_name,
        "num_labels": num_labels,
        "labels": labels_list,
        "best_f1_balanced": best_f1,           # balanced dev（用于 early stopping）
        "best_f1_full": full_metrics["f1"],    # full dev（SOTA 可比口径）
        "full_dev_metrics": full_metrics,
        "history": history,
        "args": vars(args),
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, cls=PathEncoder)

    logger.info("Balanced dev best F1=%.4f  |  Full dev F1=%.4f  →  saved to %s",
                best_f1, full_metrics["f1"], output_dir)
    return meta


def run_standard(dataset_name: str, output_dir: Path, args):
    train_examples = load_re_examples(dataset_name, "train")
    dev_examples = load_re_examples(dataset_name, "dev")
    logger.info("%s: train=%d dev=%d", dataset_name, len(train_examples), len(dev_examples))
    return train_one_fold(dataset_name, train_examples, dev_examples, output_dir, args)


def run_cv(dataset_name: str, output_dir: Path, args):
    """GAD 十折交叉验证（论文3.4.2节）。"""
    from sklearn.model_selection import KFold

    all_examples = _load_gad("train", fold=0)
    test_examples = _load_gad("test", fold=0)   # 独立测试集，用于 final model 评估
    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_examples)):
        logger.info("=== GAD Fold %d/%d ===", fold_idx + 1, args.cv_folds)
        train_ex = [all_examples[i] for i in train_idx]
        val_ex = [all_examples[i] for i in val_idx]
        fold_dir = output_dir / f"fold_{fold_idx}"
        meta = train_one_fold(dataset_name, train_ex, val_ex, fold_dir, args)
        fold_results.append(meta["best_f1_balanced"])   # 修复：key 与 train_one_fold 一致
        logger.info("Fold %d F1=%.4f", fold_idx + 1, meta["best_f1_balanced"])

    avg_f1 = sum(fold_results) / len(fold_results)
    logger.info("GAD 10-fold mean F1=%.4f", avg_f1)

    # 用全量训练集训练 final model，用独立 test split 评估（无数据泄露）
    logger.info("Training final model on all GAD data (eval on test split)...")
    train_one_fold(dataset_name, all_examples, test_examples, output_dir / "final", args)

    with open(output_dir / "cv_results.json", "w") as f:
        json.dump({"folds": fold_results, "mean_f1": avg_f1}, f, indent=2, cls=PathEncoder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["chemprot", "ddi", "gad"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=_DEFAULT_BIOBERT,
                        help="HF model ID 或本地路径，默认 dmis-lab/biobert-base-cased-v1.2")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience（连续 N epoch 无提升则停止）")
    parser.add_argument("--neg-ratio", type=float, default=-1.0,
                        help="负样本下采样比（相对正样本数）；-1 表示不下采样，使用全量数据（SOTA口径）")
    parser.add_argument("--cv-folds", type=int, default=10,
                        help="GAD交叉验证折数")
    args = parser.parse_args()

    if args.dataset == "gad":
        run_cv(args.dataset, args.output_dir, args)
    else:
        run_standard(args.dataset, args.output_dir, args)

if __name__ == "__main__":
    main()
