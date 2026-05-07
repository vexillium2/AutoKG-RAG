#!/bin/bash
# Kazu NER + 归一化
# 需要设置 KAZU_MODEL_PACK 环境变量
set -e
cd "$(dirname "$0")/.."

if [ -z "$KAZU_MODEL_PACK" ]; then
    echo "ERROR: KAZU_MODEL_PACK environment variable not set"
    echo "Download from: https://github.com/AstraZeneca/KAZU/releases"
    exit 1
fi

JSONL_COUNT=$(find data/pubmed -name "*.jsonl" 2>/dev/null | wc -l | tr -d ' ')
if [ "$JSONL_COUNT" -eq 0 ]; then
    echo "ERROR: data/pubmed/ has no .jsonl files."
    echo "  Run step 02 first: bash scripts/02_fetch_pubmed.sh"
    exit 1
fi
echo "Found $JSONL_COUNT JSONL files in data/pubmed/, starting NER..."

python -m src.ner.kazu_pipeline \
    --input-dir data/pubmed \
    --output data/mentions/mentions.jsonl \
    --batch-size 32

echo "NER done. Mention count:"
wc -l data/mentions/mentions.jsonl
