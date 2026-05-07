#!/bin/bash
# BioBERT RE 微调：三个数据集并行训练
# 预计耗时：ChemProt ~1.5h，DDI ~1h，GAD 10-fold ~2h（RTX4080）
set -e
cd "$(dirname "$0")/.."

# 国内服务器若无法访问 huggingface.co，设置镜像站
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

BATCH_SIZE=${BATCH_SIZE:-32}
EPOCHS=${EPOCHS:-8}
LR=${LR:-1e-5}
MAX_LEN=${MAX_LEN:-512}
NEG_RATIO=${NEG_RATIO:--1}      # -1 = 不下采样（SOTA口径）；改为 2.0 可加速训练
PATIENCE=${PATIENCE:-3}

BIOBERT_MODEL=${BIOBERT_MODEL:-dmis-lab/biobert-base-cased-v1.1}

echo "=== Training ChemProt model ==="
python -m src.re.train \
    --dataset chemprot \
    --output-dir models/re_chemprot \
    --model "$BIOBERT_MODEL" \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --max-length $MAX_LEN \
    --neg-ratio $NEG_RATIO \
    --patience $PATIENCE

echo "=== Training DDI model ==="
python -m src.re.train \
    --dataset ddi \
    --output-dir models/re_ddi \
    --model "$BIOBERT_MODEL" \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --max-length $MAX_LEN \
    --neg-ratio $NEG_RATIO

echo "ChemProt and DDI training done."

echo "=== Training GAD model (10-fold CV) ==="
python -m src.re.train \
    --dataset gad \
    --output-dir models/re_gad \
    --model "$BIOBERT_MODEL" \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --max-length $MAX_LEN \
    --neg-ratio $NEG_RATIO \
    --patience $PATIENCE \
    --cv-folds 10

echo "=== All RE models trained ==="
echo "Results:"
for d in models/re_chemprot models/re_ddi models/re_gad/final; do
    if [ -f "$d/meta.json" ]; then
        echo "$d: $(python3 -c "import json; m=json.load(open('$d/meta.json')); print(f'balanced={m[\"best_f1_balanced\"]:.4f}  full={m[\"best_f1_full\"]:.4f}')")"
    fi
done
