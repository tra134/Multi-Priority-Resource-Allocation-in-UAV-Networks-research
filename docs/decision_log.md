# Decision Log

Deviations from the plan.md mid-term slides and other implementation decisions.
Format: date, what, why. (Rule 9 of plan.md Section 0.)

## Inherited from plan.md Section 5 (mid-term slide fixes)

| ID | Issue in slides | Decision |
|----|-----------------|----------|
| D1 | Slide 11 says decision cycle < 10 ms, RQ1 says < 5 ms | Use slot = 5 ms and require measured actor + ensemble inference < 5 ms |
| D2 | TTL = 40 slots (200 ms) is shorter than LLM latency (200-500 ms) | TTL = 600 slots; cooldown >= 50 slots |
| D3 | "Fallback if LLM > 200 ms" would fire almost always | Fallback fires only on timeout (> 1000 ms), invalid JSON, or schema/range violation |
| D4 | `reward_weight_scale` changes the reward, but a frozen SAC policy does not adapt | Drop `eta_llm`; LLM controls only masks (bandwidth floors, power priority) |
| D5 | Training without masks then testing with masks adds a distribution shift | Train SAC with randomized masks applied with probability 0.3 |
| D6 | "0% violations" for budget constraints is trivial after projection | Treat as a unit test; scientific claims are about SOS delay violations and JSON validity |
| D7 | Slide diagram shows UAV trajectory in the action | Trajectory is out of scope (fixed hover) |
| D8 | 5-8% LLM budget vs. non-concurrent calls | Calls may overlap (async); budget = calls initiated / slots |

## Phase 0 decisions

| Date | Decision | Why |
|------|----------|-----|
| 2026-10-28 | Python 3.13.5 used instead of 3.11 | 3.11 is not installed on this machine; 3.13 is available and all target libraries support it |
| 2026-10-28 | `.gitignore` ignores `results/`, `checkpoints/`, `venv/`, `.env` | Per plan.md Section 2: large outputs and secrets are not committed |
| 2026-10-28 | `uv` available but using standard `python3 -m venv` + pip | Simplest, most portable option; uv can be used later |
| 2026-10-28 | Observation is 17-dim, not 16-dim as plan.md Section 4 states | The listed components (3+3+3+1+1+6) sum to 17; following the explicit component list |
| 2026-10-28 | `reset()` seeds `action_space` and `observation_space` | gymnasium does not auto-seed spaces; needed for reproducible trajectories |
| 2026-10-28 | Bare-env speed is ~6k-8k steps/s, below the 20,000 steps/s target in plan.md rule 8 | Pure Python + NumPy per-step overhead (~150µs/step) on this CPU; profiling done. 300k training steps still take < 1 min per seed, sufficient for CPU development. Long runs target Colab. Noted as a known gap; revisit with a compiled hot path if needed. |
| 2026-10-28 | Mask projection `b = f + (1-sum(f))*r` is not fixed-point idempotent | The plan's test lists "projection idempotent", but the specified formula only has `b = f/sum(f)` as a fixed point. The map always yields a feasible share (b>=f, sum=1), which is the safety-relevant invariant. Test adjusted to feasibility preservation. |

## Phase 8 decisions (real LLM integration)

| Date | Decision | Why |
|------|----------|-----|
| 2026-10-31 | Default provider is Groq at `https://api.groq.com/openai/v1` | HUMAN supplied a Groq key; the endpoint is OpenAI-compatible, so no provider-specific code is needed |
| 2026-10-31 | `max_tokens` raised from 200 to 512 | GPT-OSS emits reasoning tokens before the JSON; 512 avoids truncating the answer while staying small |
| 2026-10-31 | Default model is `openai/gpt-oss-120b` | Groq deprecated `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` on 2026-08-16; gpt-oss-120b (or `qwen/qwen3.6-27b`) is Groq's replacement |
| 2026-11-01 | Switched default to `qwen/qwen3.8-27b` | Live comparison: both models were 100% schema-valid, but Qwen was non-reasoning (~80 completion tokens/call vs ~170-400), had higher strict diagnosis accuracy on the smoke set (92% vs 75% in the 4-case check), and lower median latency. GPT-OSS remains available via `--model` / `LLM_MODEL` with `reasoning_effort: low` |
| 2026-11-01 | Prompt v2 adds CSI scale hints and a diagnosis guide; all numeric features rounded to 2 significant figures | v1 let the model call a pure fading case SOS_SURGE because it did not know the CSI scale; rounding was documented but not implemented, so cache hits were near zero |
| 2026-11-01 | Free-tier pacing: `--tpm-limit 7000`, rate-limit retries with `Retry-After` | Groq free tier is 8000 tokens/minute; the first 20-call run hit 429 after 8 calls. Pacing reduced a 50-call run to zero rate-limit waits |
| 2026-11-01 | Measured latency model: median 1159 ms, sigma 0.705, timeout 5000 ms; S3 stress median 4000 ms | The plan's 350 ms assumption does not hold for the hosted Groq model from this network. 5000 ms accepts ~95% of nominal calls; 4000 ms S3 remains a clear degradation |
| 2026-11-01 | Live Groq smoke test completed on a network where the API is reachable: 100% schema-valid, 92% diagnosis accuracy, n=50 | Phase 8 acceptance criterion (>=95% valid) is met. The earlier 403 HKG block is resolved and retained in history below |
| 2026-11-01 | Added `scripts/llm_controller_smoke.py`; real run installed 3 valid masks in a 600-slot surge episode (0 invalid/timeouts) | Parser-level validity is not enough: this verifies the controller, schema, mask projection and fallback wiring with the real endpoint |
| 2026-10-31 | API key stored only in `.env`; `configs/llm.yaml` keeps `api_key: null` | Rule: never commit keys. `.env` is gitignored; `.env.example` documents the variables |
| 2026-10-31 | `RealLLMClient` uses the official `openai` SDK instead of importing `httpx` directly | `openai` is already a project dependency and handles retries/timeouts/base URLs; the old code imported `httpx`, which was not installed (Python 3.13 venv has `httpx2`) |
| 2026-10-31 | `RealLLMClient.call()` never raises; network/JSON/schema errors return `ok=False` | The controller must always reach the safety fallback; a provider outage must not crash the fast loop |
| 2026-10-31 | Added tolerance `1e-9` to the `sum(floors) <= 0.95` validator | `0.6 + 0.2 + 0.15` evaluates to `0.9500000000000001` in floating point and was wrongly rejected |
| 2026-10-31 | Fixed MockLLM and prompt few-shot floors to sum to <= 0.95; fixed controller validation calling `.to_mask()` on a dict | Both bugs made every "valid" LLM response fall through to fallback, so the successful mask path had never been exercised. Regression tests added |
| 2026-10-31 | Smoke test defaults to 20 synthetic triggered situations (10 surge, 10 deep fading) | Cheap Phase 8 check without a trained checkpoint; uses the same prompt builder and schema as the controller |
| 2026-10-31 | ~~Groq live calls return 403 from this machine (Cloudflare, HKG edge)~~ | Resolved on 2026-11-01 after changing networks; see the live smoke-test row above |

## Phase 3/4 training decisions (Colab-ready)

| Date | Decision | Why |
|------|----------|-----|
| 2026-11-01 | Added `--n-critics` to `scripts/train.py` | One config can train both phases (2 = Pure SAC, 10 = REDQ-SAC) without hand-editing `configs/default.yaml` between runs |
| 2026-11-01 | Checkpoints now store optimizer state, entropy optimizer and the step counter; `load()` returns the payload | Real resume after a Colab disconnect. Previously only weights and `total_it` were saved, so Adam moments and the training position were lost |
| 2026-11-01 | Checkpoint filenames include `N_critics` (`sac_seed<seed>_N<n>`) | Pure SAC and REDQ-SAC runs for the same seed must not overwrite each other |
| 2026-11-01 | `--ckpt-every` overrides `training.ckpt_every` | Colab sessions can be pre-empted; a 5000-step interval bounds lost work without touching the default config |
| 2026-11-01 | Notebook clones over HTTPS with an optional `GITHUB_TOKEN` Colab secret, then resets `origin` to the token-free URL | Colab has no SSH key setup by default; resetting the remote avoids persisting the token in `.git/config` |
| 2026-11-01 | TensorBoard logs stay local and are copied to Drive at the end | Avoids per-scalar Drive I/O during training; checkpoints still write to Drive so they survive a disconnect |
| 2026-11-01 | Resume rebuilds the replay buffer | Buffer snapshots would add ~1 GB per checkpoint. Documented in the notebook and README; the first post-resume updates wait until one batch is collected |
| 2026-11-01 | Every `scripts/*.py` adds `<repo>/src` to `sys.path`, and the Colab notebook launches with `sys.executable` + an explicit `PYTHONPATH` | Colab raised `ModuleNotFoundError: uavrl` when the editable install was missing or belonged to another interpreter. Source-tree import makes the scripts independent of `pip install -e .` |
| 2026-11-01 | The training cell self-heals: if `import uavrl` fails it reinstalls the checkout, then rechecks before launching | After a Colab disconnect the runtime restarts; users often rerun only the training cell |
| 2026-11-01 | `.gitignore` virtualenv patterns are root-anchored (`/env/`, `/venv/`, `/.venv/`); all `src/uavrl/env/*.py` files are tracked | The bare `env/` pattern silently ignored `src/uavrl/env/`, so GitHub/Colab clones had no environment package (`ModuleNotFoundError: uavrl.env`). `tests/test_package_layout.py` now fails if any `src/**/*.py` becomes gitignored |
