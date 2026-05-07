#!/bin/bash
# 构建 FAISS 向量索引（SapBERT 编码所有证据句）
set -e
cd "$(dirname "$0")/.."

NEO4J_PASSWORD=${NEO4J_PASSWORD:-"your_password"}

python -m src.retrieval.faiss_builder \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password "$NEO4J_PASSWORD" \
    --output-dir data/faiss \
    --batch-size 256

echo "FAISS index built: data/faiss/index.bin"
ls -lh data/faiss/
