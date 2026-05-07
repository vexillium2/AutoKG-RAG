#!/bin/bash
# 批量 EFetch 下载 PubMed 摘要
set -e
cd "$(dirname "$0")/.."

if [ -z "$NCBI_API_KEY" ]; then
    echo "WARNING: No NCBI_API_KEY set, rate limited to 3 req/s"
    WORKERS=2
else
    echo "Using API key, rate limit 10 req/s"
    WORKERS=4
fi

python -m src.data.fetch_pubmed \
    --pmid-file data/pmids/all_pmids.txt \
    --output-dir data/pubmed \
    --api-key "$NCBI_API_KEY" \
    --batch-size 100 \
    --workers $WORKERS

echo "Download complete."
ls data/pubmed/ | wc -l
