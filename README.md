# 🫀 PhysioRAG: Offline-First Multi-Modal RAG for Clinical Time-Series Data

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Powered by pyturboquant](https://img.shields.io/badge/Powered%20by-pyturboquant-ff69b4.svg)](https://github.com/jorgebmann/pyturboquant)

**PhysioRAG** is an open-source, air-gapped Retrieval-Augmented Generation (RAG) framework designed specifically for high-frequency physiological sensor data and MedTech machine logs. 

It bridges the gap between raw intensive care unit (ICU) waveforms (Ventilator Pressure/Flow, SpO2, ECG) and natural language semantic search.

## 🚀 The Problem & Our Solution
Hospitals, CROs, and MedTech R&D departments produce terabytes of physiological time-series data. Traditional RAG systems are entirely text-bound and fail to interpret the complex "grammar" of machine signals. 

**PhysioRAG** maps multi-modal sensor waveforms into the same semantic vector space as text, so signals and clinical language are searchable together. It ships a lightweight **1D-CNN baseline encoder** (`baseline_cnn`) for the ventilator track and a **reused open ECG–language dual-encoder** (MERL) for CLIP-style ECG search (see [ECG semantic search](#-ecg-semantic-search-phase-b--reused-merl-dual-encoder) below). Further modality-specific models (*PaPaGei*, *Chronos*) remain on the roadmap (see `PROJECT_BRIEF_PhysioRAG.md`, Phase B+).

**The result:** You can query large archives of ventilator data using natural language—completely offline.

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

Requires **Python >= 3.12** and **PyTorch >= 2.4** (needed by
[pyturboquant](https://github.com/jorgebmann/pyturboquant)).

```bash
git clone https://github.com/jorgebmann/PhysioRAG.git
cd PhysioRAG
python3.12 -m venv .venv
source .venv/bin/activate
# Include the real compressor (pyturboquant) and dev tools:
pip install -e ".[dev,quant]"
```

Without the `quant` extra, quantization falls back to a `float16_stub` (little
real compression). Phase A smoke / strict runs require the real TurboQuant codec
from `pyturboquant.core` (installed via the `quant` extra: `pyturboquant>=0.1.1`).
Weaviate still indexes the **dequantized float32 reconstruction** for ANN search;
ingest reports compressed-byte / fidelity stats so “TurboQuant on” is measurable.

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
# Real WFDB (default collection: WaveformEpochV2):
python scripts/ingest_waveforms.py --dataset mimic_wdb --modality ventilator

# Or fully offline synthetic demo (auto-uses collection WaveformEpochDemo):
python scripts/ingest_waveforms.py --dataset mimic_demo --modality ventilator
```

`mimic_demo` is written to `WaveformEpochDemo` so curated scenarios are not
drowned out by a larger `mimic_wdb` index. Real WFDB stays on `WaveformEpochV2`
(see `configs/default.yaml`).

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

### 7. Try the search widget

A tiny static demo UI (no build step) is served by the same FastAPI app:

```
http://127.0.0.1:8000/
```

Type a natural-language query (or click one of the example chips), see the
matched waveform epochs rendered as plots alongside the LLM-synthesized,
citation-grounded answer. Great for a quick screen-recorded walkthrough.

## 🫁 Ventilator retrieval (Phase C)

The synthetic ventilator demo covers labeled asynchrony scenarios (double
triggering, air trapping, ineffective effort, flow starvation, delayed cycling,
reverse triggering, low compliance, normal) as 2-channel `Paw`/`Flow` windows
with **bilingual (EN + DE) templated captions** and structured metadata
(`asynchrony_type`, `vent_mode`, `peep_cmh2o`, `diagnosis`, `finding`,
`pairing_tier`).

Retrieval is `hybrid_text`: a vent-only German→English glossary (applied only for
ventilator / unfiltered queries) rewrites the query, MiniLM searches the caption
text vector, and BM25 scores the caption plus promoted metadata properties
(in-memory: reciprocal rank fusion of dense + keyword). Captions store real
umlauts so untranslated German still matches. This is caption/metadata retrieval,
**not** CLIP-style text→signal — see [`docs/VENT_RETRIEVAL.md`](docs/VENT_RETRIEVAL.md)
for the pairing tiers and why ICU notes are never used as captions.

```bash
# Re-create the demo collection (new searchable metadata properties) and ingest:
python scripts/ingest_waveforms.py --dataset mimic_demo --modality ventilator --reset-collection

# German Dräger-style query (glossary rewrites it to English under the hood):
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Double Triggering unter Druckunterstützung","modality":"ventilator","top_k":3}'

# Each hit carries a metadata block (asynchrony_type, vent_mode, channels, …)
# next to its Paw/Flow plot_url.
```

Track retrieval quality with a frozen EN/DE query set (labeled Recall@k vs
chance; hybrid / MiniLM-only / keyword-only). Queries are split into a "caption"
subset and a harder "paraphrase" subset (which avoids the caption's own words),
and the corpus carries unlabeled distractor epochs so recall is not trivially 1:

```bash
python scripts/eval_vent_retrieval.py --variants 3
# writes data/processed/vent_recall.json (with a by_subset block)
# add --backend weaviate to score on the live product fusion
```

## 🫀 ECG semantic search (Phase B — reused MERL dual-encoder)

PhysioRAG can search **12-lead ECG** by natural language using an open ECG–language
dual-encoder ([MERL](https://github.com/cheliu-computation/MERL-ICML2024),
Liu et al., ICML 2024) — **no training required**. The ECG signal tower produces
the index vector; the matched Med-CPT text tower embeds queries into the *same*
256-d space, so retrieval runs CLIP-style (text → ECG signal) instead of the
ventilator MiniLM/BM25 hybrid. This lives in its own config and collection; the
ventilator default (`configs/default.yaml`) is untouched.

### 1. Get the MERL checkpoint (once, online)

Download the **full** `*_ckpt.pth` (both towers — the `*_encoder.pth` is ECG-only
and can't serve queries) from the [MERL release](https://github.com/cheliu-computation/MERL-ICML2024)
and place it locally (git-ignored):

```bash
mkdir -p data/models/merl
# copy res18_best_ckpt.pth into data/models/merl/
# (path is set in configs/ecg_merl.yaml -> embeddings.merl.checkpoint)
```

Pre-cache the Med-CPT text model into your Hugging Face cache:

```bash
python -c "from transformers import AutoModel, AutoTokenizer; \
AutoModel.from_pretrained('ncbi/MedCPT-Query-Encoder'); \
AutoTokenizer.from_pretrained('ncbi/MedCPT-Query-Encoder')"
```

> The wrapper is MERL's ResNet18 `ECGCLIP` path (no stem max-pool, `downconv` +
> attention pool, Med-CPT `pooler_output` → `proj_t`) and loads with `strict=True`.
> A ViT `*_ckpt.pth` or an ECG-only `*_encoder.pth` fails at load with a sample
> of checkpoint keys. Confirm the MERL license before redistributing weights.

### 2. Download a bounded PTB-XL subset (open access)

```bash
python scripts/download_ptbxl.py --max-records 200
```

### 3. Ingest ECG (256-d signal vectors → isolated collection)

```bash
python scripts/ingest_waveforms.py --config configs/ecg_merl.yaml \
  --dataset ptbxl --modality ecg --reset-collection
```

`--reset-collection` is needed the first time because `WaveformEpochEcg` uses
256-d vectors (vs the 128-d ventilator index). Hits include a 3×4 12-lead PNG
(`/waveforms/{epoch_id}?format=png`).

### 4. Search ECG by natural language

```bash
# Point the API at the ECG config:
PHYSIORAG_CONFIG=configs/ecg_merl.yaml uvicorn api.main:app --reload

curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"atrial fibrillation with rapid ventricular response","modality":"ecg","top_k":5}'
```

`signal_aligned` search applies the same PTB-XL DE/SV→EN glossary as eval when
the query contains those tokens (`vorhofflimmern` → `atrial fibrillation`).
English queries are left unchanged. This ECG glossary is separate from the
ventilator vent-only glossary; the two never share vocabulary.

### 5. Report Recall@k vs the baseline

```bash
python scripts/eval_ecg_retrieval.py --config configs/ecg_merl.yaml \
  --dataset ptbxl --max-records 200
```

This runs text → ECG Recall@{1,5,10} on a frozen, patient-level query set.
Queries are the **raw diagnostic report** (product metric), the same reports after
a longest-match DE/SV→EN glossary (`merl_report_en_to_ecg`; no extra encoder),
and, separately, the English **SCP caption** (closer to MERL zeroshot prompts).
MERL ECG input is min-max scaled to [0, 1] with aVL/aVF swapped to MIMIC lead
order; queries are lowercased. Report queries are also scored after ingest
quantization. MiniLM is a text→text reference only (report → SCP caption) plus
chance `k / corpus_size`. Writes `data/processed/ecg_recall.json`.

### Offline note

Everything above runs air-gapped once the MERL checkpoint and Med-CPT are cached
(`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`). PTB-XL is open access (no MIMIC
DUA), but review its PhysioNet terms.

## 🔒 Air-Gapped Install & Smoke Test

PhysioRAG is designed to run with no outbound network at query time. Do the
one-time downloads while online, then flip the offline switches.

### Prepare once (online)

```bash
# 1. Python deps incl. the real compressor
pip install -e ".[dev,quant]"

# 2. Pre-cache the local text encoder into your Hugging Face cache
python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# 3. Pull the local LLM and the Weaviate image
ollama pull llama3.1
docker pull cr.weaviate.io/semitechnologies/weaviate:1.34.0
```

### Run offline

```bash
# Force libraries to use local caches only (no network calls)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Weaviate + Ollama run locally as shown in the Quickstart above.
```

Strict Phase A smoke requires these flags (or pass `--no-strict`).

### Strict mode

Set strict mode so a broken offline setup fails loudly instead of silently
degrading to keyword-only search or skipping synthesis:

```bash
export PHYSIORAG_STRICT=1
# or per-run: python scripts/ingest_waveforms.py --strict ...
# or in configs/default.yaml: runtime.strict: true
```

Under strict mode, ingest/serve raise a clear error if the text encoder can't
load, the vector store is unreachable, `pyturboquant.core` isn't installed, or
Ollama isn't healthy. `/search` also returns HTTP 503 if synthesis is requested
but Ollama fails while strict.

### Verify (Phase A acceptance)

With Weaviate and Ollama running, one command proves the whole path
(ingest → Weaviate → `/health` → `/search` → PNG plot → grounded Ollama answer):

```bash
# Synthetic demo (default):
python scripts/smoke_demo.py

# Bounded WFDB (after download_mimic_wdb.py):
python scripts/smoke_demo.py --dataset mimic_wdb
```

It exits non-zero with an actionable message if any step fails. You can also
sanity-check the service directly:

```bash
curl -s http://127.0.0.1:8000/health
# expect: {"status":"ok","text_encoder":true,"llm":true,"store":"weaviate",
#          "store_ok":true,"quant_available":true,...}
```

ECG (Phase B), after the MERL checkpoint + PTB-XL subset from above:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/smoke_demo.py --dataset ptbxl --max-records 20
```

That ingest → `WaveformEpochEcg` → `/search` (`sinus rhythm` and `vorhofflimmern`)
→ landscape 12-lead PNG (`data/processed/ecg_smoke.png`) → cited Ollama answer.

## 🎯 Primary Use Cases
* **MedTech R&D:** Finding edge-cases in historical sensor logs to improve machine algorithms and alarm systems.
* **Pharma & Clinical Trials:** Discovering physiological anomalies in historic trial data without relying on manual text logs.
* **ICU Analytics:** Empowering doctors to search massive patient waveform histories using natural language.

## 👨‍🔬 About the Author
Built by **[Jörg Bahlmann, PhD](https://www.linkedin.com/in/joergbahlmann)**. 
Senior AI Engineer and Neuroscientist specializing in high-dimensional data analysis, scalable GenAI, and Enterprise RAG architectures.

## ⚠️ Disclaimer
*PhysioRAG is a research prototype and software demonstration. It is NOT a certified medical device and must not be used for direct clinical decision-making or patient diagnosis.*
