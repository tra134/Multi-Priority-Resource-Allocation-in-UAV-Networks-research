# Multi-Priority-Resource-Allocation-in-UAV-Networks-research

UAV Multi-Priority Resource Allocation with **SAC + Ensemble Critic + Async LLM Arbiter**.

Reproducible simulation study testing the claims in the PBL3 mid-term report.
See `plan.md` for the full implementation plan (Phases 0-10).

**Status:** Phases 0-7 implemented and tested. Phase 8 (real LLM integration) is
implemented; the Groq key is configured locally in `.env`. The current machine
is blocked by Groq/Cloudflare from this region (`403 Access denied`), so the
live smoke test must be run from an allowed network (VPN/another country) or
with Ollama / another OpenAI-compatible provider.

## Quick start (copy-paste)

```bash
# 0. Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 1. Install the package (dev extras include pytest + ruff)
pip install -U pip
pip install -e ".[dev]"

# 2. Run the test suite (58 tests)
pytest -q

# 3. Phase 1: channel & traffic plots
python scripts/plot_channel.py

# 4. Phase 2: baseline comparison
python scripts/run_baselines.py

# 5. Phase 3: train Pure SAC
python scripts/train.py --config configs/default.yaml --agent sac --seed 0 --steps 300000

# 6. Phase 8: real-LLM smoke test (Groq) + Claim E inference benchmark
python scripts/llm_smoke_test.py --n 20
python scripts/benchmark_inference.py
```

## LLM access (Phase 8)

Secrets live in `.env` (gitignored). Copy the template and fill in the key:

```bash
cp .env.example .env
# edit .env
```

```dotenv
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=gsk_...
LLM_MODEL=openai/gpt-oss-120b
```

`configs/llm.yaml` contains the non-secret defaults (provider, model,
temperature, retry count, cache). Environment variables override the YAML, so
the same code works locally, on Colab, and in tests.

Groq deprecated `llama-3.3-70b-versatile` on 2026-08-16; the default is now
`openai/gpt-oss-120b` (the smaller/faster alternative is
`qwen/qwen3.6-27b`).

### Smoke test

```bash
# real Groq endpoint: JSON validity, diagnosis accuracy, latency, tokens
python scripts/llm_smoke_test.py --n 20

# no network / CI: exercises the same pipeline with MockLLM
python scripts/llm_smoke_test.py --n 20 --mock

# also fit a lognormal latency model for the simulator
python scripts/llm_smoke_test.py --n 50 --write-latency-config
```

Results are written to `results/phase8/smoke_test.{json,csv}`. The script exits
non-zero if fewer than 95% of responses are schema-valid (plan.md Phase 8
acceptance criterion).

### Troubleshooting

- **403 `Access denied. Please check your network settings.`** — Groq blocks
  API access from some regions (this machine currently resolves through Hong
  Kong). The key is fine; run from an allowed network or switch provider.
- **Use Ollama instead (offline):** `ollama pull qwen2.5:3b`, then set
  `LLM_BASE_URL=http://localhost:11434/v1`, `LLM_API_KEY=ollama`,
  `LLM_MODEL=qwen2.5:3b`.
- **Never commit `.env`.** It is already in `.gitignore`.

## Repository layout

See `plan.md` Section 3. Source lives in `src/uavrl/`, configs in `configs/`,
scripts in `scripts/`, tests in `tests/`, notebooks in `notebooks/`, and
results in `results/` (gitignored).

## Claims tested (plan.md Section 1)

| ID | Claim |
|----|-------|
| A | Ensemble uncertainty gives early OOD warning |
| B | LLM arbiter adds value beyond a rule-based shield |
| C | Hybrid trigger beats periodic / reactive / always-on at lower cost |
| D | Safety layer keeps the system safe when the LLM fails |
| E | Real-time feasibility (actor + ensemble inference < 5 ms/slot) |
