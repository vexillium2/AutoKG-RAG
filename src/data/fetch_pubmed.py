#!/usr/bin/env python3
"""
通过 NCBI EFetch API 按 PMID 批量下载 PubMed 摘要，输出 Document JSONL。

速率：无 API key → 3 req/s；有 API key → 10 req/s（免费申请：
https://www.ncbi.nlm.nih.gov/account/）

用法：
    python -m src.data.fetch_pubmed \\
        --pmid-file data/pmids/all_pmids.txt \\
        --output-dir data/pubmed \\
        --api-key YOUR_KEY          # 可选，但强烈建议
        --batch-size 500 \\
        --workers 4
"""
import argparse
import gzip
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from lxml import etree
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
# 每批上限 100（NCBI 规定），有 API key 时 10 req/s，无则 3 req/s
BATCH_SIZE = 100


@dataclass
class Document:
    doc_id: str
    title: str
    body: str
    meta_tags: list
    publication_date: str


def _fetch_batch(pmids: list[str], api_key: Optional[str], retries: int = 3) -> bytes:
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "xml",
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key

    for attempt in range(retries):
        try:
            resp = requests.get(EFETCH_URL, params=params, timeout=60)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                logger.warning("Batch failed (%s), retry in %ds", e, wait)
                time.sleep(wait)
            else:
                raise


def _parse_xml(xml_bytes: bytes) -> list[Document]:
    docs = []
    root = etree.fromstring(xml_bytes)

    for article in root.findall(".//PubmedArticle"):
        citation = article.find("MedlineCitation")
        if citation is None:
            continue
        art = citation.find("Article")
        if art is None:
            continue

        pmid = (citation.findtext("PMID") or "").strip()
        title = (art.findtext(".//ArticleTitle") or "").strip()

        texts = [
            "".join(at.itertext()).strip()
            for at in art.findall(".//AbstractText")
        ]
        body = " ".join(t for t in texts if t)
        if not body:
            continue

        mesh_terms = [
            (d.text or "").strip()
            for d in citation.findall(
                ".//MeshHeadingList/MeshHeading/DescriptorName"
            )
        ]

        pub_date = ""
        pd = art.find(".//PubDate")
        if pd is not None:
            parts = [
                pd.findtext(f, "").strip()
                for f in ("Year", "Month", "Day")
            ]
            pub_date = "-".join(p for p in parts if p)

        docs.append(Document(
            doc_id=pmid,
            title=title,
            body=body,
            meta_tags=mesh_terms,
            publication_date=pub_date,
        ))

    return docs


def _process_batch(
    batch_pmids: list[str],
    output_dir: Path,
    api_key: Optional[str],
    delay: float,
) -> int:
    """下载并解析一批PMID，写入独立JSONL文件，返回实际写入条数。"""
    batch_id = batch_pmids[0]
    out_file = output_dir / f"batch_{batch_id}.jsonl"

    # 幂等：已存在则跳过
    if out_file.exists() and out_file.stat().st_size > 0:
        with open(out_file) as f:
            return sum(1 for _ in f)

    time.sleep(delay)
    xml_bytes = _fetch_batch(batch_pmids, api_key)
    docs = _parse_xml(xml_bytes)

    with open(out_file, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")

    return len(docs)


def run(
    pmid_file: Path,
    output_dir: Path,
    api_key: Optional[str],
    batch_size: int,
    workers: int,
) -> None:
    pmids = [line.strip() for line in pmid_file.read_text().splitlines() if line.strip()]
    logger.info("Total PMIDs: %d", len(pmids))

    output_dir.mkdir(parents=True, exist_ok=True)

    batches = [pmids[i:i + batch_size] for i in range(0, len(pmids), batch_size)]
    logger.info("Batches: %d (size=%d)", len(batches), batch_size)

    # 有 API key 时 10 req/s，无则 3 req/s；多worker时需除以worker数
    base_delay = 0.1 if api_key else 0.34
    per_worker_delay = base_delay * workers

    total_docs = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_batch, b, output_dir, api_key, per_worker_delay): b
            for b in batches
        }
        with tqdm(total=len(batches), desc="Fetching batches") as pbar:
            for fut in as_completed(futures):
                try:
                    total_docs += fut.result()
                except Exception as e:
                    logger.error("Batch failed: %s", e)
                pbar.update(1)

    logger.info("Done. Total documents written: %d", total_docs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmid-file", type=Path, default=Path("data/pmids/all_pmids.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/pubmed"))
    parser.add_argument("--api-key", default=None, help="NCBI API key（推荐，免费申请）")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=3,
                        help="并发线程数；有API key可设4，无则设2")
    args = parser.parse_args()

    run(args.pmid_file, args.output_dir, args.api_key, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
