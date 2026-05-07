#!/bin/bash
# 三元组批量入库 Neo4j
set -e
cd "$(dirname "$0")/.."

NEO4J_PASSWORD=${NEO4J_PASSWORD:-"your_password"}

python -m src.kg.neo4j_ingestor \
    --triple-file data/triples/triples.jsonl \
    --mention-file data/mentions/mentions.jsonl \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password "$NEO4J_PASSWORD" \
    --batch-size 500

echo "KG build complete."
