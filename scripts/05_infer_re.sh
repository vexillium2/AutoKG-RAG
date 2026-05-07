#!/bin/bash
# BioBERT RE 全量推理：生成 Triple JSONL
set -e
cd "$(dirname "$0")/.."

python -m src.re.infer \
    --mention-file data/mentions/mentions.jsonl \
    --doc-dir data/pubmed \
    --output data/triples/triples.jsonl \
    --chemprot-dir models/re_chemprot \
    --ddi-dir models/re_ddi \
    --gad-dir models/re_gad/final \
    --batch-size 256

echo "Triple count:"
wc -l data/triples/triples.jsonl
