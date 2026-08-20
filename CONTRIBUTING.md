# Contributing to PhysioRAG

Thanks for your interest in PhysioRAG. This is a research and engineering
prototype for offline, multi-modal search over physiological time-series data.
Bug reports, reproducible benchmarks, and focused pull requests are all welcome.

## Ground rules

- **No patient data, no proprietary device data, no credentials** in issues,
  PRs, or commits. The demo is fully synthetic; keep it that way in examples.
- **Stay offline-first.** Do not add dependencies on hosted APIs (OpenAI,
  Anthropic, Pinecone, etc.). Retrieval and synthesis must be able to run
  air-gapped.
- **Be honest about capabilities.** The ventilator track is caption/metadata
  hybrid retrieval, not CLIP-style text→signal alignment; the ECG (MERL) track
  is the true cross-modal one. Please keep docs and claims precise (see
  [`docs/VENT_RETRIEVAL.md`](docs/VENT_RETRIEVAL.md)).

## Getting set up

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ,quant for real pyturboquant
```

Run the synthetic demo:

```bash
python scripts/serve_demo.py     # http://127.0.0.1:8000/
```

## Before opening a pull request

- Run the tests: `pytest` (the full-stack smoke tests are skipped unless
  `PHYSIORAG_SMOKE=1` with Weaviate + Ollama running).
- Run the linter: `ruff check .`
- Keep changes focused and describe *why*, not just *what*.
- If you touch retrieval quality, include eval numbers
  (`scripts/eval_vent_retrieval.py` / `scripts/eval_ecg_retrieval.py`).

## Model weights and datasets

External model weights (e.g. MERL) and datasets (MIMIC-IV, PTB-XL) carry their
own licenses and terms. Do **not** commit them or redistribute them through this
repository. They are git-ignored on purpose.

## Questions

Open a GitHub Discussion for questions, "I got it running" reports, or ideas.
For anything security-related, see [`SECURITY.md`](SECURITY.md).
