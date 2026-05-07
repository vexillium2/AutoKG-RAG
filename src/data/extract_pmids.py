#!/usr/bin/env python3
"""
从 BioASQ JSON 文件中提取所有 PMID，输出到文本文件。

用法：
    python -m src.data.extract_pmids \\
        --bioasq-dir data/bioasq \\
        --output data/pmids/all_pmids.txt
"""
import argparse
import json
import re
from pathlib import Path


def extract_pmids(bioasq_dir: Path) -> set[str]:
    pmids: set[str] = set()
    pattern = re.compile(r"/pubmed/(\d+)")

    for fpath in sorted(bioasq_dir.glob("*.json")):
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        for q in data.get("questions", []):
            for url in q.get("documents", []):
                m = pattern.search(url)
                if m:
                    pmids.add(m.group(1))

    return pmids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bioasq-dir", type=Path, default=Path("data/bioasq"))
    parser.add_argument("--output", type=Path, default=Path("data/pmids/all_pmids.txt"))
    args = parser.parse_args()

    pmids = extract_pmids(args.bioasq_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for p in sorted(pmids):
            f.write(p + "\n")

    print(f"Extracted {len(pmids)} unique PMIDs → {args.output}")


if __name__ == "__main__":
    main()
