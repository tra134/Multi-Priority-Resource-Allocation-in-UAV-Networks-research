# Multi-Priority-Resource-Allocation-in-UAV-Networks-research

UAV Multi-Priority Resource Allocation with **SAC + Ensemble Critic + Async LLM Arbiter**.

Reproducible simulation study testing the claims in the PBL3 mid-term report.
See `plan.md` for the full implementation plan (Phases 0-10).

## Quick start (copy-paste)

```bash
# 0. Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 1. Install the package (dev extras include pytest + ruff)
pip install -U pip
pip install -e ".[dev]"

# 2. Run the test suite
pytest -q

# 3. Phase 1: channel & traffic plots
python scripts/plot_channel.py

# 4. Phase 2: baseline comparison
python scripts/run_baselines.py

# 5. Phase 3: train Pure SAC
python scripts/train.py --config configs/default.yaml --agent sac --seed 0 --steps 300000
```

## LLM access (Phase 8+)

Set one OpenAI-compatible endpoint via environment variables (never commit keys):

```bash
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_API_KEY="sk-..."
export LLM_MODEL="gpt-4o-mini"
```

For local offline runs, use Ollama: `ollama pull qwen2.5:3b`, then
`export LLM_BASE_URL="http://localhost:11434/v1"` and
`export LLM_API_KEY="ollama"`.

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
