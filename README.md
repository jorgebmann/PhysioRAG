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

## 💻 Quickstart (Demo)

*Note: This repository contains a proof-of-concept pipeline using open-access MIMIC-IV waveform samples. No proprietary MedTech data is included.*

### 1. Clone & Install

```bash
git clone https://github.com/jorgebmann/PhysioRAG.git
cd PhysioRAG
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Data Ingestion (Demo Dataset)

```bash
python scripts/ingest_waveforms.py --dataset mimic_demo --modality ventilator
```

### 3. Start the Retrieval API

```bash
uvicorn api.main:app --reload
```

## 🎯 Primary Use Cases
* **MedTech R&D:** Finding edge-cases in historical sensor logs to improve machine algorithms and alarm systems.
* **Pharma & Clinical Trials:** Discovering physiological anomalies in historic trial data without relying on manual text logs.
* **ICU Analytics:** Empowering doctors to search massive patient waveform histories using natural language.

## 👨‍🔬 About the Author
Built by **[Jörg Bahlmann, PhD](https://www.linkedin.com/in/joergbahlmann)**. 
Senior AI Engineer and Neuroscientist specializing in high-dimensional data analysis, scalable GenAI, and Enterprise RAG architectures.

## ⚠️ Disclaimer
*PhysioRAG is a research prototype and software demonstration. It is NOT a certified medical device and must not be used for direct clinical decision-making or patient diagnosis.*
