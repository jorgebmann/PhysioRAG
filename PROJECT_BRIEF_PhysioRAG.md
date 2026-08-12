# Project Context & Architecture Brief: PhysioRAG

## 1. Naming & Scope

| Name | Role |
|------|------|
| **PhysioRAG** | Product / system: offline-first multi-modal RAG over physiological waveforms |
| **Physio-CLTP** | Method / subproject: Contrastive Language–Time-series Pretraining of modality-specific dual encoders |

**Do not** brand the stack as “Physio-CLTP-RAG.” In papers and demos say: *PhysioRAG with modality-specific CLTP (or reused ECG–CLIP) encoders.* Training details live in [`docs/PROJECT_BRIEF_Physio-CLTP.md`](docs/PROJECT_BRIEF_Physio-CLTP.md). Older drafts: [`docs/PROJECT_BRIEF_PhysioRAG_first_version.md`](docs/PROJECT_BRIEF_PhysioRAG_first_version.md).

**Goal:** Natural-language semantic search over high-frequency physiological time-series (ventilator logs, SpO2/PPG, ECG), with plottable evidence and local LLM synthesis — fully air-gapped.

**Audience:** MedTech R&D (e.g. Drägerwerk), Pharma/CROs. Research prototype only — **not** a medical device.

**Core problem:** Waveform archives are searchable only via scripts and metadata. Text-only RAG cannot interpret signal morphology. Cloud APIs are prohibited for patient data and IP.

**Solution:** Local ingest → modality-specific encoding into a text-aligned vector space → **`pyturboquant` post-hoc compression** → local vector DB → retrieve epochs + synthesize with a local LLM.

---

## 2. MedTech MVP (Ventilator / “Dräger” Prototype)

- **Persona:** R&D engineer hunting edge cases for alarm / algorithm development.
- **Example query:** *“Show me waveform windows where patients with ARDS breathed spontaneously against the ventilator, causing a pressure spike.”*
- **Output:** Ranked plottable epochs (arrays/PNG), linked metadata, citation-grounded local LLM answer.
- **Demo data today:** Bounded `mimic4wdb` subsets + synthetic `mimic_demo` (see `configs/default.yaml`). Continuous ICU waveforms alone are **not** CLIP-quality caption pairs — see §4.

ECG and SpO2 are first-class modalities in the architecture, but the **product story** leads with ventilator search; ECG is the strongest place to **reuse** existing dual-encoders.

---

## 3. Encoding Strategy (What Changed)

### 3.A Principle: modality-specific towers + shared retrieval pattern

Do **not** train one mega multi-channel model (Pressure+Flow+SpO2+ECG). Each modality has its own signal encoder (and, when contrastively trained, a matched text tower). PhysioRAG indexes and queries per modality (or separate collections), with a common pipeline: encode → quantize → store → retrieve → synthesize.

### 3.B ECG — reuse, don’t retrain by default

Open ECG–language dual-encoders (e.g. **MERL**, ECG-CLIP-style, FG-CLEP, C-MELT, ECG-Chat’s contrastive stage) already align **10 s 12-lead ECG ↔ report text**. That largely fulfills the *scientific* CLTP goal for ECG.

**PhysioRAG default for ECG:**
1. Wrap a usable open dual-encoder (prefer **MERL** if weights + both towers are available offline).
2. Index ECG epochs with the **signal** tower; embed user queries with the **same** text tower.
3. Evaluate as **corpus RAG** (text→ECG Recall@k), not only zero-shot PTB-XL classification.
4. Retrain / Physio-CLTP on MIMIC-IV-ECG only if reuse fails packaging, licensing, or retrieval quality.

### 3.C Ventilator — product track; captions are the bottleneck

No large public CLIP-scale set of `(pressure/flow window ↔ free-text asynchrony caption)`.

| Approach | Role |
|----------|------|
| Synthetic / model-based PSV with ground-truth asynchrony labels → templated captions | Best CLIP-like pairs for vent CLTP |
| Expert-labeled asynchrony (if accessible) | Supervised or contrastive fine-tune |
| `mimic4wdb` / continuous ICU waveforms | Domain adaptation / unimodal SSL — **not** primary pairs with ICU notes |
| Chart-derived templates (`mode=PSV, PEEP=…`) | Weak metadata; OK as hybrid signal, not morphology captions |

**Near-term PhysioRAG vent path:** keep `baseline_cnn` (or SSL vent encoder) + hybrid retrieval (signal vector + BM25/metadata) while CLTP vent pairs are built. Do not pretend same-day progress notes are CLIP captions.

### 3.D PPG / SpO2 — second multimodal track

- Prefer **PaPaGei**-style unimodal init + **PulseLM** / label→caption templates for text alignment.
- ICU free-text notes are a poor PPG morphology caption source.

### 3.E Physio-CLTP (when we train)

Train dual-encoders only where reuse is insufficient (primarily **vent**, optionally **PPG**). Leonardo (A100 **64 GB**, Slurm) is for those tracks — not to redo MERL-scale ECG CLIP without a clear gap. Details, pairing tiers, and HPC roadmap: [`docs/PROJECT_BRIEF_Physio-CLTP.md`](docs/PROJECT_BRIEF_Physio-CLTP.md).

---

## 4. Pairing Quality (CLIP Analogy)

CLIP needs text that describes **the same window** as the signal.

| Tier | Example | PhysioRAG use |
|------|---------|----------------|
| Strong | ECG study ↔ its machine/cardiologist report | Reuse MERL / index ECG–report corpora |
| Medium | PPG window ↔ templated SpO2/AF caption; vent breath ↔ `"double triggering"` | CLTP or supervised alignment |
| Weak | ICU vent epoch ↔ same-day note / loose chart join | Exclude from primary contrastive train; optional metadata only |

Public **`mimic4wdb` v0.1.0** is a small technical preview — fine for pipeline demos, not a “foundation model” corpus.

---

## 5. System Architecture (Repository Truth)

The repo already implements the RAG skeleton. Agents **extend** it; they do not rebuild FastAPI from scratch.

```
WFDB / demo waveforms
    → ingest & epoch (modality-specific preprocessing)
    → TimeSeriesEncoder (+ optional matched text encoder for queries)
    → pyturboquant (post-hoc, at index time)
    → Weaviate / memory store + raw array store (plottable evidence)
    → search (dense ± BM25) → Ollama / local LLM synthesis
    → FastAPI (`api/main.py`)
```

### Components

| Stage | Responsibility | Notes |
|-------|----------------|-------|
| Ingestion | WFDB / demo processors, windowing, resample, normalize | Generators/chunking; never load full corpus into RAM |
| Encoders | Abstract `TimeSeriesEncoder`; `baseline_cnn` today; plugs for MERL / CLTP / PaPaGei | Match `embedding_dim` in config or document breaking changes |
| Text queries | Local text encoder; for CLTP/MERL paths use the **paired** text tower | Avoid OpenAI / cloud embed APIs |
| Quantization | **`pyturboquant` after** float encode, before/at vector insert | Not QAT-in-training by default |
| Storage | Weaviate (or memory); link vector ↔ epoch id ↔ numpy/plot URL | Modality-specific collections OK |
| Synthesis | Ollama / llama.cpp; cite epoch ids | Air-gapped model weights |
| API | FastAPI search + waveform plot/JSON | Existing demo UI |

### Constraints

1. **Strictly offline** at runtime: no `openai` / `anthropic` / hosted vector DBs. Pre-cache HF/Ollama weights; `HF_HUB_OFFLINE=1` when air-gapped.
2. **Modular encoders** — swapping MERL or a CLTP checkpoint must not rewrite ingest or API.
3. **Memory-efficient** I/O for MIMIC-scale data.
4. **PhysioNet DUA** — credentials via env only; no “proprietary foundation model” claims without compliance review.
5. Prefer **post-hoc TurboQuant**; optional quantization-aware research must not block the RAG path.

---

## 6. Evaluation

- **Product:** text→waveform Recall@k on a held-out epoch index; qualitative Dräger-style queries with plottable hits; synthesis grounded in epoch ids.
- **ECG reuse:** MERL (or similar) vs current `baseline_cnn` + MiniLM on an ECG index.
- **CLTP vent/PPG:** patient-level splits; beat non-contrastive baseline on agreed query set before declaring success.
- Splits must be **patient-level** (no subject leakage).

---

## 7. Roadmap for the AI Agent

Work against the **existing** tree (`src/physiorag/`, `api/`, `scripts/`, `configs/default.yaml`). Prioritize integration and modality gaps over greenfield training.

### Phase A — Product hardening (ongoing)

1. Keep ingest → encode → quantize → Weaviate → search → Ollama reliable on `mimic_demo` / bounded `mimic_wdb`.
2. Preserve abstract encoder + config-driven backend selection.
3. Document air-gap steps (weight pre-cache, offline flags).

### Phase B — ECG dual-encoder backend (reuse)

1. Add a `TimeSeriesEncoder` (+ query text path) wrapping MERL or another open ECG–CLIP dual-encoder.
2. Ingest a small ECG set (e.g. MIMIC-IV-ECG or PTB-XL subset); index signal embeddings; query with the matched text tower.
3. Report Recall@k vs `baseline_cnn` + MiniLM.
4. **Exit:** ECG semantic search works offline in PhysioRAG without training from scratch.

### Phase C — Ventilator retrieval quality

1. Improve vent demo: synthetic/labeled asynchrony captions, hybrid dense+keyword retrieval, clearer metadata.
2. Optional: unimodal SSL vent encoder; optional Physio-CLTP vent dual-encoder on Leonardo when pair index exists (see CLTP Phase 0 gate).
3. **Exit:** Dräger-style queries return plausible pressure/flow epochs with plots.

### Phase D — PPG / SpO2 (optional)

1. PaPaGei-style or baseline PPG encoder; templated / PulseLM-style text alignment.
2. Separate collection or named vectors; same API shape.

### Phase E — Scale & HPC (as needed)

1. Physio-CLTP training only for modalities without a good open dual-encoder.
2. Export checkpoints into PhysioRAG; remapping plan if dims/windows differ from `configs/default.yaml`.

---

## 8. Acceptance Criteria (summary)

| Milestone | Done when |
|-----------|-----------|
| A | Demo path works air-gapped on synthetic + bounded WFDB; TurboQuant on; citations in answers |
| B | Open ECG dual-encoder plugged in; text→ECG retrieval beats or matches baseline on a fixed eval set |
| C | Vent queries return plottable evidence; pairing/hybrid strategy documented (no fake note–CLIP claims) |
| D/E | Extra modalities or CLTP weights load via the same encoder interface without API rewrite |

---

## 9. Out of Scope

- Cloud LLM / embedding / vector APIs.
- Claiming a single shared “foundation model” across all ICU signals without modality-specific towers.
- Primary contrastive training on weak ICU note↔waveform joins.
- Replacing the existing FastAPI/Weaviate stack as part of encoder work.
- QAT-inside-training as a hard dependency (post-hoc `pyturboquant` is the default).

---

## 10. Pointers

- CLTP training brief: [`docs/PROJECT_BRIEF_Physio-CLTP.md`](docs/PROJECT_BRIEF_Physio-CLTP.md)
- Prior PhysioRAG draft: [`docs/PROJECT_BRIEF_PhysioRAG_first_version.md`](docs/PROJECT_BRIEF_PhysioRAG_first_version.md)
- Runtime config: `configs/default.yaml`
- Quickstart: `README.md`
