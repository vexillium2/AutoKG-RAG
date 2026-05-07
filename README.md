# AutoKG-RAG：面向检索增强生成的领域知识图谱自动化构建系统研究

本科毕业设计 · 2026

端到端流水线：PubMed 摘要 → 命名实体识别 → 关系抽取 → Neo4j 知识图谱 → 混合 KG-RAG → BioASQ 问答评测

---

## 系统概览图

![系统概览](./system_overview.svg)

---

## 实验结果

### 关系抽取（RE）模型性能

| 数据集 | BERT-base | PubMedBERT | **BioBERT（本实验）** |
|--------|-----------|--------|-------|
| ChemProt | 73.74 | **77.24** | 75.81 |
| DDI-2013 | 75.63 | 82.36 | **82.51** |
| GAD（10折均值）| 79.33 | 83.96 | **84.25** |

### 知识图谱统计（实际构建结果）

**节点统计**

| 节点类型 | 数量 |
|----------|-----:|
| Gene | 9,291 |
| Disease | 9,977 |
| Chemical | 4,353 |
| **合计** | **23,621** |

**关系统计**

| 关系类型 | 数量 |
|----------|-----:|
| DOWNREGULATES | 3,761 |
| INTERACTS_WITH | 2,299 |
| UPREGULATES | 1,624 |
| SUBSTRATE_OF | 267 |
| INHIBITS | 239 |
| ASSOCIATED_WITH | 235 |
| ACTIVATES | 189 |
| **合计** | **17,376** |

FAISS 向量索引：**14,262 条**证据向量，768 维（SapBERT 编码）

---

## 环境要求

| 项目 | 要求 |
|------|------|
| GPU | RTX 3090（24GB VRAM）或同等显存显卡 |
| CUDA | 12.x（已在 CUDA 12.8 上验证） |
| 内存 | 32GB+ |
| 存储 | 约 60GB（模型 + 数据） |
| Python | 3.10 |
| Neo4j | 5.x（需 Java 17+） |

---

## 安装

### 1. 创建 Conda 环境

```bash
conda create -n biokg-rag python=3.10 -y
conda activate biokg-rag

# PyTorch（CUDA 12.8，适配 RTX 3090/4090/5090）
pip install torch==2.7.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# 其余依赖
conda env create -f environment.yml
```

### 2. 安装 Neo4j

```bash
# Ubuntu/Debian
sudo apt install -y openjdk-17-jdk
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable 5' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt update && sudo apt install -y neo4j

sudo neo4j-admin dbms set-initial-password <你的密码>
sudo systemctl enable --now neo4j
```

### 3. 下载 Kazu 模型包（约 8GB）

```bash
wget https://github.com/AstraZeneca/KAZU/releases/download/v2.3.0/kazu_model_pack_public_v2.3.0.tar.gz
tar -xzf kazu_model_pack_public_v2.3.0.tar.gz
export KAZU_MODEL_PACK=$(pwd)/kazu_model_pack_public_v2.3.0
```

### 4. 准备 DDI Corpus 解析工具

```bash
# 克隆 bigbio/biomedical 仓库，用于 BRAT 格式解析
git clone https://github.com/bigbio/biomedical.git /path/to/biomedical
export DDI_BIGBIOHUB_DIR=/path/to/biomedical
```

---

## 环境变量配置

```bash
export NEO4J_PASSWORD="your_neo4j_password"

export KAZU_MODEL_PACK="/path/to/kazu_model_pack_public_v2.3.0"

export DDI_BIGBIOHUB_DIR="/path/to/biomedical"

# NCBI API Key（免费注册，可将 PubMed 抓取速度提升约 10 倍）
export NCBI_API_KEY="your_ncbi_api_key"

# HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 主要依赖与致谢

本项目构建于以下开源工具与模型之上：

| 组件 | 来源 | 用途 |
|------|------|------|
| [Kazu v2.3.0](https://github.com/AstraZeneca/KAZU) | AstraZeneca | 生物医学 NER + 实体归一化 |
| [BioBERT v1.1](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1) | DMIS Lab, Korea University | 关系抽取骨干模型 |
| [SapBERT](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext) | Cambridge LTL | 证据句向量编码 |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Alibaba Cloud | 答案生成 |
| [Neo4j 5.x](https://neo4j.com) | Neo4j Inc. | 图数据库存储与查询 |
| [FAISS](https://github.com/facebookresearch/faiss) | Meta AI Research | 高效向量相似度检索 |
| [ChemProt](https://huggingface.co/datasets/bigbio/chemprot) | Kringelum et al. | 化学蛋白质关系 RE 数据集 |
| [DDI-2013](https://github.com/isegura/DDICorpus) | Herrero-Zazo et al. | 药物相互作用 RE 数据集 |
| [GAD](https://huggingface.co/datasets/bigbio/gad) | Bravo et al. | 基因疾病关联 RE 数据集 |
| [BioASQ](http://www.bioasq.org) | Tsatsaronis et al. | 生物医学问答评测基准 |
