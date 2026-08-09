# 🫀 PhysioRAG: Offline-First Multi-Modal RAG for Clinical Time-Series Data

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Powered by pyturboquant](https://img.shields.io/badge/Powered%20by-pyturboquant-ff69b4.svg)](https://github.com/jorgebmann/pyturboquant)

**PhysioRAG** is an open-source, air-gapped Retrieval-Augmented Generation (RAG) framework designed specifically for high-frequency physiological sensor data and MedTech machine logs. 

It bridges the gap between raw intensive care unit (ICU) waveforms (Ventilator Pressure/Flow, SpO2, ECG) and natural language semantic search.

## 🚀 The Problem & Our Solution
Hospitals, CROs, and MedTech R&D departments produce terabytes of physiological time-series data. Traditional RAG systems are entirely text-bound and fail to interpret the complex "grammar" of machine signals. 

**PhysioRAG** solves this by leveraging state-of-the-art **Time-Series Foundation Models** (e.g., *MorphologyFM*, *PaPaGei*, *Chronos*) to map multi-modal sensor waveforms into the same semantic vector space as text.

**The result:** You can query 50,000 hours of ventilator data using natural language—completely offline.

> **Example Query:** *"Find all ventilator pressure curves from last year showing patients with ARDS breathing spontaneously against the machine."* -> Returns the exact 12-second waveform snippets and associated clinical metadata in milliseconds.

## ✨ Key Features
* 🔒 **100% Offline / Air-Gapped:** Zero data leaves the local machine. No OpenAI API required. Fully compliant with GDPR and HIPAA for sensitive R&D and patient IP.
* 🧠 **Time-Series to Vector:** Encodes raw 1D/2D signals (ECG, PPG, respiratory flow) into rich semantic embeddings.
* ⚡ **Ultra-Low Hardware Footprint:** High-frequency waveform embeddings usually lead to RAM explosions. PhysioRAG integrates [pyturboquant](https://github.com/jorgebmann/pyturboquant) to aggressively compress and quantize vectors, allowing the entire pipeline to run on standard hospital/R&D edge servers.
* 📊 **Multi-Modal Retrieval:** Searches simultaneously across text (clinical PDFs, manuals) and machine waveforms.

## 🏗️ Architecture Pipeline
1. **Ingestion:** Reads raw waveform data (e.g., WFDB format from MIMIC-IV).
2. **Epoching & Encoding:** Slices time-series data into windows (e.g., 10-second epochs) and processes them through a physiological foundation model.
3. **Quantization:** Compresses the resulting high-dimensional embeddings using `pyturboquant`.
4. **Vector Storage:** Stores compressed vectors in a local vector database (e.g., Weaviate / pgvector).
5. **Retrieval & Synthesis:** Matches natural language queries to waveform vectors and uses a local LLM (e.g., Llama-3) to synthesize the results.

## 💻 Quickstart

*Note: This is a proof-of-concept pipeline. Real waveforms come from the credentialed MIMIC-IV Waveform Database; no proprietary MedTech data is included. A fully synthetic `mimic_demo` dataset is available for offline smoke tests.*

### 1. Clone & Install

```bash
git clone https://github.com/jorgebmann/PhysioRAG.git
cd PhysioRAG
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Start local infrastructure (Weaviate + Ollama)

```bash
# Weaviate (vector DB) — local Docker instance on :8080 (+ gRPC :50051)
docker run -d --name weaviate -p 8080:8080 -p 50051:50051 \
  -e ENABLE_MODULES="" -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  cr.weaviate.io/semitechnologies/weaviate:1.34.0

# Ollama (local LLM synthesis)
ollama pull llama3.1
```

The text encoder (`sentence-transformers/all-MiniLM-L6-v2`) downloads to your local
Hugging Face cache on first run. For an air-gapped host, pre-cache it once online,
then set `HF_HUB_OFFLINE=1`.

### 3. Configure PhysioNet credentials (never commit these)

You need a [PhysioNet](https://physionet.org/) account that has signed the
[MIMIC-IV Waveform Database](https://physionet.org/content/mimic4wdb/0.1.0/) DUA.

```bash
cp .env.example .env
# edit .env and set PHYSIONET_USERNAME / PHYSIONET_PASSWORD
```

### 4. Download a bounded WFDB subset

```bash
python scripts/download_mimic_wdb.py --max-records 3
```

### 5. Ingest → encode → quantize → index

```bash
# Real WFDB (default):
python scripts/ingest_waveforms.py --dataset mimic_wdb --modality ventilator

# Or fully offline synthetic demo:
python scripts/ingest_waveforms.py --dataset mimic_demo --modality ventilator
```

### 6. Search and view plottable evidence

```bash
uvicorn api.main:app --reload

# Natural-language, cross-modal search (text vector + BM25):
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"ARDS spontaneous breathing against the ventilator","top_k":3}'

# Each hit includes a plot_url; open the PNG of a returned epoch:
#   http://127.0.0.1:8000/waveforms/<epoch_id>?format=png
# Or fetch raw samples as JSON:
#   http://127.0.0.1:8000/waveforms/<epoch_id>?format=json
```

With Ollama running, `/search` also returns a grounded `answer` citing the epoch
ids it used. Interactive API docs live at `http://127.0.0.1:8000/docs`.

## 🎯 Primary Use Cases
* **MedTech R&D:** Finding edge-cases in historical sensor logs to improve machine algorithms and alarm systems.
* **Pharma & Clinical Trials:** Discovering physiological anomalies in historic trial data without relying on manual text logs.
* **ICU Analytics:** Empowering doctors to search massive patient waveform histories using natural language.

## 👨‍🔬 About the Author
Built by **[Jörg Bahlmann, PhD](https://www.linkedin.com/in/joergbahlmann)**. 
Senior AI Engineer and Neuroscientist specializing in high-dimensional data analysis, scalable GenAI, and Enterprise RAG architectures.

## ⚠️ Disclaimer
*PhysioRAG is a research prototype and software demonstration. It is NOT a certified medical device and must not be used for direct clinical decision-making or patient diagnosis.*
