# Project Context & Architecture Brief: Physio-CLTP

## 1. Project Overview & Computing Context

**Project Name:** Physio-CLTP (Contrastive Language-Time-series Pretraining)
**Parent Product:** PhysioRAG (offline-first multi-modal RAG for physiological waveforms)
**Compute Environment:** EuroHPC Leonardo Booster — custom NVIDIA A100 GPUs with **64 GB** HBM2e (not 80 GB). Nodes typically provide 4× A100 + 512 GB host RAM. Jobs run via **Slurm** on the Linux cluster.

**Goal:** Train modality-specific dual-encoder models that align physiological time-series windows with clinical text in a shared embedding space. Trained encoders become drop-in `TimeSeriesEncoder` backends for PhysioRAG’s existing ingest → quantize → retrieve → synthesize pipeline.

**Relationship to PhysioRAG (source of truth):**
- Product brief: [`PROJECT_BRIEF_PhysioRAG.md`](../PROJECT_BRIEF_PhysioRAG.md). Naming: **PhysioRAG** = system; **Physio-CLTP** = training method — not “Physio-CLTP-RAG.”
- Physio-CLTP **produces encoder weights** (and optionally a matched text tower) where open dual-encoders are missing or insufficient.
- **ECG default in PhysioRAG:** reuse open ECG–CLIP models (e.g. MERL), not retrain on Leonardo unless packaging, licensing, or retrieval quality forces it.
- **CLTP compute priority:** ventilator (and optionally PPG/SpO2), where public CLIP-quality pairs/models are weak.
- PhysioRAG **keeps** FastAPI, Weaviate/local vector store, post-hoc `pyturboquant` at index time, Ollama/local LLM synthesis, and the modular encoder interface.
- Do **not** rebuild a FastAPI wrapper as part of CLTP. Embedding dim, window length, and sample rate must match PhysioRAG config or be documented as a deliberate breaking change with a remapping plan.

**What “from scratch” means (when we train):**
- **Waveform tower:** trained (random init or SSL init), modality-specific — not a generic off-the-shelf TSFM as the primary CLTP path.
- **Text tower:** **initialized from a pretrained clinical LM** (e.g. `emilyalsentzer/Bio_ClinicalBERT`), with top layers unfrozen. This is *not* training the text encoder from scratch.

**Legal / product note:** MIMIC / PhysioNet DUAs constrain redistribution and commercial use of derived artifacts. Prefer “research prototype encoder” over unqualified “proprietary foundation model” until compliance is reviewed. PhysioRAG remains a research prototype, not a medical device.

---

## 2. Scientific Risk: Pairing Quality (CLIP Analogy)

CLIP works because captions describe **the same object** as the image. For waveforms we need:

`(signal window) ↔ (text that names what is in that window)`

| Pairing quality | Example | Use |
|-----------------|---------|-----|
| **Strong** | 10 s 12-lead ECG + machine/cardiologist report for *that* study | Primary CLTP pretrain |
| **Medium** | PPG/SpO2 window + templated clinical caption or PulseLM-style text/QA | Second multimodal track |
| **Weak** | Continuous ICU vent window + same-day progress note / loose `chartevents` join | Avoid as primary InfoNCE pairs |

**Critical flaw to avoid:** Treating MIMIC-IV continuous bedside waveforms + ICU free-text notes / `chartevents` as CLIP-style pairs. Notes are usually written later, summarize hours of care, and rarely pin events (e.g. double-triggering) to a 30 s window. `chartevents` are mostly structured values, not narrative captions. `noteevents` is MIMIC-III naming; MIMIC-IV notes live in **`mimic-iv-note`**.

**Scale caveat:** Public **MIMIC-IV Waveform Database (`mimic4wdb`) v0.1.0** is a small technical preview (~200 records / ~198 patients), not a Tier-0-scale corpus. Do not plan a “foundation model” narrative on that release alone.

---

## 3. Datasets & Modality Strategy

**Prefer modality-specific encoders and datasets.** Do not start with one mega multi-channel model (Pressure + Flow + SpO2 + ECG). Separate towers keep sampling rates, window lengths, caption schemas, and in-batch negatives honest. Later, project into a shared text space or maintain modality-specific indexes in PhysioRAG.

### 3.A ECG — best public CLIP analogue (reuse first; train only if needed)

| Resource | Role | Notes |
|----------|------|--------|
| **MIMIC-IV-ECG** (`mimic-iv-ecg`) | Pretrain | ~800k × 10 s, 12-lead @ 500 Hz; machine summary reports; optional links toward cardiologist notes via `waveform_note_links` / MIMIC-IV-Note |
| **PTB-XL** | Eval / supplemental | SCP statements → short English captions; standard zero-shot / retrieval benchmark |
| Chapman–Shaoxing / CSN / SPH | Optional pool | Common in ECG–language literature |

**Why first:** Study-level alignment (one recording ↔ one report), diagnostic language about *that* ECG, scale suitable for InfoNCE. Caveats: reports are templated; many lack fine morphology detail (optional later: LLM-augmented tags as in FG-CLEP-style work). Prefer ECG↔its own report; avoid fragile joins to other EHR tables for the core pair set.

**Success target:** Text→ECG and ECG→text Recall@k on held-out patients; zero-shot / retrieval transfer to PTB-XL vs a non-contrastive baseline.

### 3.B PPG / SpO2 — second multimodal track

| Resource | Role | Notes |
|----------|------|--------|
| **PulseLM**-style PPG–text / QA | Contrastive or instruction pairs | Standardized windows + clinical attribute text (HR, AF, SpO2 bins, etc.) |
| **PaPaGei**-style unlabeled PPG (VitalDB, MIMIC waveforms, MESA, …) | Signal-encoder SSL init | Strong unimodal pretrain, then lighter text alignment |
| Label→caption templates | Medium-strength pairs | e.g. `"SpO2 mild hypoxemia (90–94%)"` |

Do **not** rely on ICU free-text notes to describe PPG morphology at second resolution.

### 3.C Ventilator pressure / flow — product track (Dräger-style), caption-starved

No large public dataset of `(pressure/flow window ↔ free-text asynchrony caption)` at CLIP scale.

| Resource | Role | Notes |
|----------|------|--------|
| Synthetic / model-based PSV waveforms with ground-truth asynchrony types | Primary CLIP-like pairs | Captions like `"pressure support; double triggering"` |
| Expert-labeled asynchrony corpora (often restricted) | Supervised / contrastive if accessible | Breath-level labels, not narratives |
| `mimic4wdb` / continuous ICU waveforms | Domain adaptation / SSL | **Not** primary InfoNCE pairs with notes |
| Chart-derived templates | Weak metadata captions | e.g. `"mode=PSV, PEEP=8, FiO2=0.4"` — settings, not morphology |

**Product path:** labeled/synthetic asynchrony → templated captions for contrastive or supervised retrieval; optional unimodal vent SSL + hybrid retrieval in PhysioRAG. Modality-specific vent encoders help engineering; they do **not** create captions.

### 3.D Pairing-quality acceptance (Phase 0 gate)

Before full training, produce a short pairing audit:
- Pair counts, temporal offset distribution (where timestamps exist), caption length/diversity, fraction of empty/boilerplate text.
- Explicit rank of strategies used (strong → weak) and which are allowed into the InfoNCE training set.
- **Patient-level** train/val/test splits (no subject leakage).

---

## 4. Core Architecture: Dual-Encoder (CLIP-style)

Implement a contrastive dual-encoder with a shared projection/loss interface per modality. **Default first *training* build: ventilator** (synthetic/labeled captions) or **PPG**, once a Phase 0 pair index exists. For ECG, prefer wrapping an open dual-encoder in PhysioRAG; only run ECG CLTP if reuse is insufficient.

### 4.A Text branch

- Base: pretrained medical LM (e.g. Bio_ClinicalBERT).
- Unfreeze top layers; `[CLS]` (or pooled) output → linear projection to shared dim (e.g. 768 for ECG track; document any PhysioRAG dim mismatch).
- Contrastive recipe: **bidirectional** InfoNCE (text→signal and signal→text), learnable temperature / logit scale, **all-gather negatives across GPUs** under DDP.

### 4.B Signal branch (modality-specific)

- Per modality: custom 1D-ResNet or time-series Transformer; **fixed channel layout**, documented sample rate and window length.
- ECG example: 12-lead, 10 s @ 500 Hz (dataset-native).
- Vent example: pressure ± flow, **breath-segmented** or short fixed windows — not a forced merge with SpO2/ECG channels.
- Missing-channel / resampling policy must be explicit; do not silently stack misaligned rates into one tensor.
- Projection head → same dim as text tower.

### 4.C Loss & batch size (Leonardo-realistic)

- Optimize InfoNCE (or SigLIP-style variant if justified).
- **Do not hard-code global batch 4096 on one 64 GB GPU.** Measure max per-GPU batch under AMP; reach a large **effective** batch via gradient accumulation and/or multi-GPU negative gathering.
- Stack: prefer **native PyTorch + DDP** for HPC clarity (Lightning optional if standardized). Use `torch.amp`, checkpointing where useful. Checkpointing does **not** fix huge waveform activation memory by itself — size windows and per-GPU batch deliberately.
- Require a short **memory budget** note (bytes/sample × batch × activations) before scaling jobs.

---

## 5. `pyturboquant` Integration (default: post-hoc)

`pyturboquant` is designed for **data-oblivious post-hoc compression of stored embeddings** (index/RAM path), not classic QAT of projection heads.

**Default (required for PhysioRAG parity):** train float embeddings → quantize at **index time** when inserting into the vector store.

**Optional research track (do not block Phases 0–3):** distortion-regularized or quantization-aware embeddings only if a clear loss and metric are defined (e.g. Recall@k after TurboQuant at N bits). Do not assume projection heads “learn TurboQuant-compressibility” without that protocol.

---

## 6. Evaluation Protocol (must exist before large runs)

- **Splits:** patient-level holdout on the pretrain corpus; fixed external sets (e.g. PTB-XL for ECG).
- **Metrics:** text→waveform and waveform→text Recall@k; comparison to PhysioRAG baseline (e.g. `baseline_cnn` + MiniLM) and/or non-contrastive signal encoder.
- **MVP success (ECG track):** beat agreed baseline on retrieval / zero-shot diagnosis-style probes on the eval set — not “foundation model” marketing.
- **Export test:** load CLTP weights through PhysioRAG’s encoder interface; confirm dim and retrieval behavior on a smoke index.

---

## 7. Roadmap for the AI Agent

Keep code modular. Assume Leonardo + Slurm (scratch for checkpoints, parallel FS for data — not home for large I/O).

### Phase 0 — Data access, pairing audit, eval set (gate)

1. PhysioNet credentialing / DUA awareness; download bounds and on-cluster layout.
2. Inventory hours, channels, and overlap with reports/labels per modality.
3. Build offline **pair index** (e.g. parquet/arrow): `(path_or_array_ref, text, subject_id, split, pairing_tier)`.
4. Pairing-quality report; freeze patient-level splits and eval queries.
5. **Exit:** Strong-tier pairs available for ECG (or documented fallback); weak note-joins excluded from primary train set.

### Phase 1 — Data engineering

1. PyTorch `Dataset` / `DataLoader` consuming the Phase 0 pair index (not ad-hoc on-the-fly EHR joins in the hot path).
2. Yield `(waveform_tensor, text_string)` with correct modality preprocessing.
3. **Exit:** Sustained samples/sec on CPU workers; deterministic split filtering; no full-corpus RAM load.

### Phase 2 — Model architecture

1. Implement `models/dual_encoder.py` with a **modality-specific** signal encoder (start ECG), ClinicalBERT text tower, projection heads, forward → both embeddings.
2. Unit-test tiny batches on CPU/GPU.
3. **Exit:** Shapes, device moves, and freeze/unfreeze flags verified.

### Phase 3 — Loss & training loop

1. `train.py`: bidirectional InfoNCE, AMP, DDP, accumulation, checkpoint/resume, experiment logging.
2. Single-GPU smoke train → multi-GPU scale; discover per-GPU batch experimentally.
3. Slurm job templates for 1–4+ GPUs.
4. **Exit:** Val retrieval metrics logged; checkpoint exportable.

### Phase 4 — PhysioRAG integration

1. Export encoder; wire into PhysioRAG encode path; **post-hoc `pyturboquant`** at index time.
2. Measure retrieval vs previous baseline on a bounded index.
3. Only then invest in RAG/LLM polish for demo queries.

### Phase 5 (optional) — Additional modalities

1. PPG track (PulseLM-style / templated pairs ± PaPaGei init).
2. Vent track (synthetic/labeled asynchrony captions + SSL on real waveforms).
3. Shared text space or separate collections — decide based on PhysioRAG product needs.

---

## 8. Acceptance Criteria (summary)

| Phase | Done when |
|-------|-----------|
| 0 | Pairing audit written; patient splits fixed; strong/medium pairs identified; weak note-joins demoted |
| 1 | DataLoader serves batched pairs from the pair index without loading all waveforms into RAM |
| 2 | Dual-encoder forward pass returns aligned dims; ECG path works end-to-end on a mini-batch |
| 3 | Train run on Leonardo with DDP/AMP; val Recall@k tracked; reproducible checkpoint |
| 4 | Encoder loads in PhysioRAG; post-hoc TurboQuant indexing works; retrieval ≥ baseline on agreed eval |

---

## 9. Out of Scope (for now)

- Replacing PhysioRAG’s API / Weaviate / Ollama stack.
- QAT-inside-training as a hard dependency.
- Single multi-channel “all signals” foundation model as the first milestone.
- Treating `mimic4wdb` preview + ICU notes as sufficient CLIP-scale pretrain data.

*Keep implementations modular and HPC-friendly. Prefer correctness of pairs and eval over premature scale.*
