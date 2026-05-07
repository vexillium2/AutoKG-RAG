#!/bin/bash
# BioASQ Task 13b 端到端评测（4个batch）
set -e
cd "$(dirname "$0")/.."

NEO4J_PASSWORD=${NEO4J_PASSWORD:-"your_password"}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
TOP_K=${TOP_K:-10}
HOPS=${HOPS:-2}

mkdir -p results

# 对四个测试 batch 分别推理
for i in 1 2 3 4; do
    GOLDEN="data/bioasq/13B${i}_golden.json"
    OUTPUT="results/13B${i}_predictions.json"

    if [ ! -f "$GOLDEN" ]; then
        echo "Skipping batch $i (golden file not found)"
        continue
    fi

    echo "=== Batch $i ==="
    python -m src.rag.pipeline \
        --faiss-dir data/faiss \
        --neo4j-uri bolt://localhost:7687 \
        --neo4j-user neo4j \
        --neo4j-password "$NEO4J_PASSWORD" \
        --model "$MODEL" \
        --top-k $TOP_K \
        --hops $HOPS \
        --bioasq-file "$GOLDEN" \
        --output "$OUTPUT"
done

# 汇总评测指标
echo ""
echo "=== Computing Metrics ==="
python -m src.eval.run_bioasq \
    --all-batches \
    --results-dir results \
    --bioasq-dir data/bioasq

echo ""
echo "Results saved to results/bioasq_results.json"
