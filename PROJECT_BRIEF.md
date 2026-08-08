# Project Context & Architecture Brief: PhysioRAG

## 1. Project Overview & Objective
**Project Name:** PhysioRAG
**Goal:** Build an offline-first, multi-modal Retrieval-Augmented Generation (RAG) prototype that enables natural language semantic search over high-frequency physiological time-series data (sensor logs, waveforms) combined with clinical text.
**Target Audience:** R&D departments in MedTech companies (e.g., Drägerwerk) and Pharma/CROs. 

**The Core Problem:**
MedTech companies produce terabytes of machine logs (ventilator pressure curves, SpO2, ECG). Currently, finding specific physiological anomalies in historical data requires manual scripting and metadata filtering. Traditional RAG only works on text. Furthermore, patient data and R&D intellectual property are highly sensitive, meaning cloud-based APIs (like OpenAI) are strictly prohibited.

**The Solution:**
A 100% local, air-gapped system that translates raw time-series sensor data into semantic embeddings using **Physiological Time-Series Representation Learning**. The system compresses these embeddings using `pyturboquant` to allow massive vector databases to run on local edge hardware.

---

## 2. The MedTech Use-Case (The "Dräger" Prototype)
To build a compelling Minimum Viable Product (MVP), we will focus on the **Intensive Care Unit (ICU) Ventilator Use-Case**.

*   **Data Source:** Open-access MIMIC-IV Waveform Database (from PhysioNet).
*   **Modality Focus:** High-frequency waveforms, specifically Respiratory Pressure/Flow curves, SpO2 (Photoplethysmogram/PPG), and ECG.
*   **User Persona:** An R&D Engineer looking for edge-cases to train a new machine alarm algorithm.
*   **Example Query:** *"Show me 12-second waveform windows from last year where patients with ARDS breathed spontaneously against the ventilator, causing a pressure spike."*
*   **Expected Output:** The system returns the exact time-series snippets (plottable arrays) and synthesizes the related metadata/text using a local LLM.

---

## 3. Core Tech: Physiological Time-Series Representation Learning
Instead of extracting statistical features manually, we use Foundation Models to encode raw waveforms into high-dimensional semantic vectors. The AI agent should design the ingestion pipeline to accommodate these types of models:

*   **General Time-Series Foundation Models (TSFMs):** e.g., *Chronos* (Amazon), *MOMENT*, or *TimeGPT*. These can be fine-tuned on ventilator flow/pressure curves.
*   **SpO2 / PPG Models:** e.g., *PaPaGei* or *MorphologyFM*.
*   **ECG Models:** e.g., *HuBERT-ECG* or *X-ECG*.

**The Embedding Concept (Contrastive Learning / Multi-modal Alignment):** 
The ultimate goal of the encoding layer is to place the time-series vector in the same semantic vector space as the text describing that physiological event, enabling Cross-Modal Retrieval.

---

## 4. System Architecture & Pipeline Requirements

The AI Coding Agent should structure the repository to support the following pipeline:

### Phase 1: Data Ingestion & Preprocessing
*   **Input:** `WFDB` (Waveform Database) files (standard for MIMIC-IV).
*   **Epoching:** Slicer module to cut continuous waveforms into fixed windows (e.g., 10-second epochs).
*   **Signal Processing:** Basic artifact removal, resampling, and normalization.

### Phase 2: Encoding & Quantization (The USP)
*   **Model Inference:** Pass epochs through a local HuggingFace Foundation Model to extract `[CLS]` token equivalents as dense vectors (e.g., 768 dimensions).
*   **Compression (`pyturboquant`):** **CRITICAL STEP.** High-frequency epoching leads to vector explosion. The agent must integrate the `pyturboquant` library to quantize the raw embeddings, radically reducing the RAM footprint for the vector database.

### Phase 3: Storage & Retrieval
*   **Vector DB:** A local, open-source vector store (e.g., Weaviate, pgvector, or Milvus) capable of handling quantized vectors.
*   **Multi-Vector Strategy:** The DB should link the time-series embedding, the raw numpy array (for plotting in the UI), and the associated text metadata.

### Phase 4: Synthesis & API
*   **Local LLM:** Integration with a local LLM (e.g., Llama-3 via `Ollama` or `llama.cpp`) to process the retrieved context and generate a natural language response.
*   **Backend:** FastAPI to serve the retrieval endpoints.

---

## 5. Constraints & Directives for the AI Agent
1.  **Strictly Offline:** Do not use `openai`, `anthropic`, or `pinecone` SDKs. Use `transformers`, `sentence-transformers`, `ollama`, and local vector stores.
2.  **Modularity:** Keep the Time-Series Encoder abstract. We might start with a simple 1D-CNN baseline or a generic Chronos model before moving to complex mult-modal models.
3.  **Memory Efficiency:** Always design data loading (MIMIC-IV can be huge) using generators or chunking. Never load the entire dataset into RAM.
4.  **pyturboquant Integration:** Treat vector compression not as an afterthought, but as a core component of the embedding pipeline.

## 6. Next Steps for Code Generation
AI Agent, please begin by:
1. Creating the basic directory structure (e.g., `src/ingestion`, `src/embeddings`, `src/retrieval`, `api/`).
2. Defining the abstract base classes for the `WaveformProcessor` and `TimeSeriesEncoder`.
3. Setting up a robust `requirements.txt` / `pyproject.toml` including `wfdb`, `transformers`, `fastapi`, and `pyturboquant`.
