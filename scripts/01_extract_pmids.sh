#!/bin/bash
# 从 BioASQ JSON 提取所有 PMID
set -e
cd "$(dirname "$0")/.."

python -m src.data.extract_pmids \
    --bioasq-dir data/bioasq \
    --output data/pmids/all_pmids.txt

echo "PMIDs saved to data/pmids/all_pmids.txt"
wc -l data/pmids/all_pmids.txt
