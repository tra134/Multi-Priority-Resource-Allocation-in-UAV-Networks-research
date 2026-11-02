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
| 2026-11-01 | Phase 5 calibration on the trained N=10 checkpoint: `tau_unc = 0.260117...` (99th percentile of raw U on nominal validation episodes) | First calibrated value from a real model rather than an untrained agent. Written to `configs/default.yaml`; `trigger_calibrated.yaml` remains the raw artifact |
| 2026-11-01 | `llm_controller_smoke.py` gained `--trigger-mode`, `--tau-unc`, `--n-critics`, and config fallbacks for `theta_queue`/`cooldown` | The smoke test previously hard-coded the queue trigger, so it could not verify the calibrated uncertainty path or a checkpoint whose critic count differs from the config |
| 2026-11-01 | `llm_controller_smoke.py` exits immediately with a clear message when `LLM_API_KEY` is missing (unless `--mock`), and prints the client's `last_error` | On Colab the gitignored `.env` does not exist; without this the run ended as a bare `CalledProcessError` with `valid=0` |

## Phase 9 experiment-grid decisions

| Date | Decision | Why |
|------|----------|-----|
| 2026-11-02 | `scripts/run_grid.py` writes one row per (method, scenario, seed, episode) to `results/final/metrics.csv` and skips rows already present | The full grid is 14 methods x 6 scenarios x 5 seeds x 20 episodes; Colab sessions are interrupted, so resumability is mandatory |
| 2026-11-02 | Grid defaults to MockLLM; `--llm real` is opt-in | Mock runs are fast and reproducible for the main comparison. Real calls are reserved for a cost/latency validation subset |
| 2026-11-02 | S3/S4 use a 4000 ms mock latency; real calls get a latency-inflation wrapper instead of sleeping | Reproduces the degraded-API scenario without blocking the simulation, and keeps the real call's measured latency visible |
| 2026-11-02 | `HybridController.call_log` records trigger reason, OOD label, validity, timeout, diagnosis and latency per call | Phase 9 needs JSON-valid rate, timeout rate, diagnosis accuracy, trigger precision/recall and lead time, not just episode averages |
| 2026-11-02 | `M12b` (`interim_fallback=True`) applies the safe fallback from trigger time until the response arrives; the fallback preempts any active mask | The original controller stored the flag but ignored it. Preemption is intentional: an in-flight response may invalidate the current mask |
| 2026-11-02 | A call counts as a timeout when the provider reports `reason == "timeout"` or when a valid response exceeds `llm_timeout_ms` | Mock timeout injection returns `ok=False, reason="timeout"`; without this, injected timeouts were misclassified as schema failures |
| 2026-11-02 | M5 DR-SAC trains with `--dr-wide` (arrival 0.8-1.2x, shadow <= +10 dB, surge <= 3x) and writes the `_drwide` checkpoint | Implements the plan's "just train on more OOD" baseline with a distinct filename, so it cannot overwrite M3 |
| 2026-11-02 | Checkpoint names come from `uavrl.config.checkpoint_run_name`; the grid resolves both flat and per-run-folder Drive layouts | Training and evaluation previously duplicated the naming rule, and the Colab notebook writes `ckpt/<run>/<run>_latest.pt` while the grid expected a flat directory |
| 2026-11-02 | `--slots` overrides `timing.episode_slots` in the loaded config | The first runner version only changed the reported row length, so smoke tests still ran the full 2000-slot episode |
| 2026-11-02 | Calibration recheck at step ~300k: raw p99 = 0.2541 vs the stored 0.2601 (n=20 vs n=50); keep 0.2601 | The 2.3% difference is within sampling noise, so the Phase 9 grid uses the existing `configs/default.yaml` value. Recheck again only if the policy is retrained |
| 2026-11-02 | `step` and `total_it` differ by `batch - 1` (255 for batch=256), and the gap grows by 255 per resumed session | The first update waits until the replay buffer holds one batch. After a resume the buffer is rebuilt, so those warmup steps count in `step` but not in `total_it`. This is expected, not corruption |
| 2026-11-02 | Recorded that raw-U AUROC is 0.299 on surge but 0.801-0.997 on shadow/compound/unseen | The ensemble agrees under a pure load surge, so uncertainty is not a surge detector. Claim A is reported for the fading/compound scenarios, and the hybrid trigger relies on `theta_queue` for surge |
| 2026-11-02 | Added `scripts/calibrate_theta.py`; it sweeps `theta_queue` with the queue-only trigger on validation seeds (20000+) | `trigger.theta_queue` was `null`, so `run_grid.py` silently fell back to 50000 while Rule-Shield used 20000. Phase 9 needs one tuned, documented value before running M7/M12 |
| 2026-11-02 | Chose `theta_queue = 20000` bits after the validation sweep | Lowest SOS violation (1.87%) and 1.04% call rate. 35k/50k halve the calls but raise violations to ~3%; 10k doubles calls for 2.6%. The value also matches the Rule-Shield derivation, so M7 and M12 use the same threshold |
