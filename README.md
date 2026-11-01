# Multi-Priority-Resource-Allocation-in-UAV-Networks-research

UAV Multi-Priority Resource Allocation with **SAC + Ensemble Critic + Async LLM Arbiter**.

Reproducible simulation study testing the claims in the PBL3 mid-term report.
See `plan.md` for the full implementation plan (Phases 0-10).

**Status:** Phases 0-8 implemented and tested. The live Groq smoke test passed
on 2026-11-01: **100% schema-valid**, 92% strict diagnosis accuracy, 50 calls,
median latency 1159 ms. Prompt `v2-final` is frozen. Next: train checkpoints
(Phase 3/4), calibrate `tau_unc`, then run the Phase 9 grid.

## Quick start (copy-paste)

```bash
# 0. Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 1. Install the package (dev extras include pytest + ruff)
pip install -U pip
pip install -e ".[dev]"

# 2. Run the test suite (64 tests)
pytest -q

# 3. Phase 1: channel & traffic plots
python scripts/plot_channel.py

# 4. Phase 2: baseline comparison
python scripts/run_baselines.py

# 5. Phase 3: train Pure SAC (N_critics=2)
python scripts/train.py --config configs/default.yaml --agent sac --n-critics 2 \
    --seed 0 --steps 300000

# Phase 4: REDQ-SAC (N_critics=10)
python scripts/train.py --config configs/default.yaml --agent sac --n-critics 10 \
    --seed 0 --steps 300000

# Resume after the latest checkpoint (model, optimizers and step are restored;
# the replay buffer is rebuilt)
python scripts/train.py --agent sac --n-critics 10 --steps 300000 --resume

# 6. Phase 8: real-LLM smoke test (Groq) + Claim E inference benchmark
python scripts/llm_smoke_test.py --n 20
python scripts/benchmark_inference.py
```

## Colab training

`notebooks/colab_train.ipynb` trains either agent on a Colab GPU and keeps
checkpoints on Google Drive:

1. Open the notebook in Colab and choose `Runtime -> Change runtime type -> T4 GPU`.
2. For a private clone, add a Colab secret named `GITHUB_TOKEN`.
3. In the configuration cell set `N_CRITICS` (`2` = Pure SAC, `10` = REDQ-SAC),
   `SEED` and `STEPS`; then run all cells.

The notebook clones over HTTPS (the token is removed from `.git/config`
afterwards), installs the package, trains with `--resume`, verifies the
checkpoint format, and copies TensorBoard logs/results to Drive. Model,
optimizer, entropy and step state survive a disconnect; the replay buffer does
not and is rebuilt on resume.

Key training flags:

| Flag | Meaning |
|------|---------|
| `--n-critics N` | `2` = twin-critic SAC, `10` = REDQ-SAC (overrides `sac.N_critics`) |
| `--ckpt-every K` | checkpoint interval; use `5000` on flaky Colab sessions |
| `--resume` | continue from `<ckpt-dir>/<run>_latest.pt` |
| `--eval-every K` | evaluation interval in environment steps |

Run names are `sac_seed<SEED>_N<N_CRITICS>`, so Pure SAC and REDQ-SAC never
overwrite each other.

**Training troubleshooting**

- `ModuleNotFoundError: No module named 'uavrl'` — the notebook puts
  `<repo>/src` on `PYTHONPATH` and launches training with `sys.executable`.
  Rerun the clone/install cell after a runtime restart. `scripts/train.py`
  also adds `src` to `sys.path`, so it works even without `pip install -e .`.
- `CalledProcessError ... returned non-zero exit status 1` — that is only the
  wrapper; the actual Python traceback is printed immediately above it.



Secrets live in `.env` (gitignored). Copy the template and fill in the key:

```bash
cp .env.example .env
# edit .env
```

```dotenv
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=gsk_...
LLM_MODEL=qwen/qwen3.8-27b
```

`configs/llm.yaml` contains the non-secret defaults (provider, model,
temperature, retry count, cache). Environment variables override the YAML, so
the same code works locally, on Colab, and in tests.

The default is `qwen/qwen3.8-27b`: it is non-reasoning, uses fewer output
tokens, and scored best on the smoke test. `openai/gpt-oss-120b` is also
tested and still works (`--model openai/gpt-oss-120b`, with
`reasoning_effort: low`). Groq retired the old Llama 3.x models on
2026-08-16.

### Smoke test

```bash
# real Groq endpoint: JSON validity, diagnosis accuracy, latency, tokens
python scripts/llm_smoke_test.py --n 20

# no network / CI: exercises the same pipeline with MockLLM
python scripts/llm_smoke_test.py --n 20 --mock

# also fit a lognormal latency model for the simulator
python scripts/llm_smoke_test.py --n 50 --write-latency-config

# full controller + safety-mask path with the real model (short episode)
python scripts/llm_controller_smoke.py
```

Results are written to `results/phase8/smoke_test.{json,csv}` and the fitted
latency model to `configs/llm_latency.yaml`. The script exits non-zero if fewer
than 95% of responses are schema-valid (plan.md Phase 8 acceptance criterion).

The Groq free tier allows 8000 tokens/minute. The script paces calls with
`--tpm-limit 7000` and the client retries HTTP 429 using `Retry-After`;
`--no-throttle` is available for paid tiers. Numeric prompt features are
rounded to 2 significant figures so repeated situations hit the disk cache.

### Troubleshooting

- **403 `Access denied. Please check your network settings.`** — Groq blocks
  API access from some regions. Change network/VPN or switch provider.
- **429 rate limit** — the free tier is 8000 tokens/minute. Keep the default
  pacing, use the cache, or upgrade to a paid tier.
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
