# Findings

Per-plan notes on each figure/experiment. One to two sentences per plot.

## Phase 1 — Channel and traffic models

- `f1_p_los_vs_elevation.png`: P_LoS rises monotonically from near 0 at low elevation to ~1 at broadside (90 deg), matching the Al-Hourani urban curve (a=9.61, b=0.16).
- `f2_path_loss_vs_distance.png`: mean path loss grows with distance and lies between the LoS-only (1 dB env loss) and NLoS-only (20 dB env loss) bounds, weighted by P_LoS.
- `f3_snr_sample.png`: received SNR per class is stable under nominal conditions and drops by roughly the +15 dB shadow during the shaded window (slots 150-350), as expected from the shadowing hook.
- `f4_queue_overload.png`: with service below arrivals, all three queues grow roughly linearly; Emergency grows slowest (lowest arrival rate), confirming the fluid queue update direction.

## Phase 7 — Hybrid controller + MockLLM (async LLM arbiter)

- llm/schema.py: pydantic LLMResponse (operating_mode, min_bandwidth_ratio,
  power_boost_emergency, ttl_duration_slots) with validators (floors in [0,1],
  sum <= 0.95, emergency <= 0.9, TTL in [50,1000]).
- llm/mock.py: MockLLM with rule-based diagnosis + failure injection
  (invalid JSON, out-of-range, wrong diagnosis, timeout) and lognormal latency.
- llm/prompt.py: structured prompt (CSI mean/var/slope per class, queues,
  dQ_emg, U_t, trigger reason, current mask, schema, few-shot). Never contains
  ood_flag.
- llm/cache.py: disk cache keyed by hash(model + prompt).
- llm/client.py: LLMClient interface; MockLLMClient (Phases 1-7); RealLLMClient
  stub for OpenAI-compatible endpoints (Phase 8).
- control/hybrid_controller.py: per-slot async loop; trigger modes
  uncertainty/queue/hybrid/periodic/reactive/always/random; fast loop never
  blocks on the LLM.
- tests/test_schema.py, tests/test_controller.py: schema validation,
  end-to-end surge run, timeout/invalid-json fallback, cooldown, one decision
  per slot at 2 s latency, failure-injection sweep (no crash, budget holds).
- Verified: budget constraints hold under all failure-injection modes.

## Phase 6 — Safety layer (mask, projection, fallback, rule shield)

- control/mask.py: Mask dataclass (bw_floor[3], power_floor_emg, expires_at);
  project() guarantees b>=f, sum(b)=1, requires sum(f)<=1, rejects invalid masks.
- control/fallback.py: fixed safe allocation [0.8, 0.12, 0.08] for bw and power.
- baselines/rule_shield.py: queue-triggered mask (theta_queue, m consecutive,
  TTL, cooldown), zero-latency and delayed (350 ms) variants.
- tests/test_mask.py: projection feasibility/floors, invalid-mask rejection,
  expiry, fallback shares, Rule-Shield surge behavior, budget constraints over
  1e5 random steps.
- Verified: budget constraints hold (sum b = sum p = 1) over 1e5 random steps.

## Phase 5 — Uncertainty analysis (Claim A)

- `src/uavrl/uncertainty/calibration.py`: U_t = ensemble std of Q(s,a); normalized U/(|meanQ|+c); percentile calibration; AUROC/precision/recall/lead-time; rank-based AUROC.
- `scripts/calibrate_tau.py`: tau_unc = q-th percentile of U on nominal validation episodes; OOD evaluation on surge/shadow/compound/unseen; writes trigger_calibrated.yaml.
- `scripts/plot_uncertainty.py`: U_t timelines with OOD windows shaded.
- Illustrative local run (untrained agent): raw-U AUROC ≈ 0.85 (shadow), 0.88 (compound), 0.90 (unseen) — supports Claim A direction; real numbers require the Colab-trained agent (deferred).

## Phase 2 — Gymnasium environment + baselines

- `results/phase2/baselines.csv`: nominal load calibrated to 53.9% of mean achievable capacity (target 50-60%).
- `results/phase2/baseline_delay.png`: Emergency delay over one nominal episode per baseline.
- Acceptance claims verified:
  - Nominal: `strict_priority` SOS violation 0.1% vs `greedy_sum_rate` 99.9% (greedy starves Emergency because it sits at the worst channel, 400 m).
  - Surge (4x Emergency): `fair_share` 0.4% vs `strict_priority` 0.1% (fair share fails to protect Emergency under surge).
- This validates the mid-term slide 5 claim that greedy sum-rate harms delay-critical Emergency traffic.

## Phase 0 — Tooling

- `pip install -e ".[dev]" && pytest -q` passes on a clean venv. Python 3.13.5 used (plan specified 3.11, not installed; logged in docs/decision_log.md).

## Phase 8 — Real LLM integration (Groq)

- Provider: Groq OpenAI-compatible endpoint, default model **`qwen/qwen3.8-27b`**
  (50 live calls, 2026-11-01). The key is read from `.env` (gitignored).
- `src/uavrl/llm/client.py` `RealLLMClient`: OpenAI SDK, JSON mode with
  automatic fallback, schema validation, retry with rate-limit backoff,
  `reasoning_effort=low` for GPT-OSS, wall-clock latency + token accounting,
  and no exceptions escaping to the controller.
- `src/uavrl/config.py`: dependency-free `.env` loader; environment variables
  override `configs/llm.yaml`.
- `scripts/llm_smoke_test.py`: 20/50 triggered situations (surge + deep fading),
  reports schema-valid rate, diagnosis accuracy, latency percentiles and
  tokens; writes `results/phase8/smoke_test.json` / `.csv`.
- **Live smoke test (n=50, cache disabled): 100% schema-valid** (Phase 8 target
  >=95%), **92% strict diagnosis accuracy**, **1159 ms median latency**
  (p5 508 ms, p95 3773 ms), 42,156 prompt + 4,024 completion tokens. Artifacts:
  `results/phase8/smoke_test.json`, `configs/llm_latency.yaml`.
- The fitted lognormal is `mu_log=7.055`, `sigma_log=0.705`. `configs/default.yaml`
  now uses these measured values, `llm_timeout_ms=5000`, and a degraded-API
  scenario median of 4000 ms (S3). This replaces the plan's assumed 350 ms.
- Groq's free tier enforces 8000 tokens/minute. The smoke script paces calls
  (`--tpm-limit 7000`) and the client retries HTTP 429 using `Retry-After`;
  prompt features are rounded to 2 significant figures so the cache absorbs
  repeated situations.
- Mock smoke test (`--mock --n 6`): 100% schema-valid, 83% strict diagnosis
  accuracy, median latency 351 ms. The successful mask path is covered by
  `test_controller.py`.
- End-to-end controller smoke test (`scripts/llm_controller_smoke.py`, 600-slot
  surge episode, real Qwen): 3 LLM calls, 3 valid masks installed, 0 invalid,
  0 timeouts, mask active 70% of slots. This confirms the real response path
  reaches the safety layer, not just the parser. Artifact:
  `results/phase8/controller_smoke.json`.
- `scripts/benchmark_inference.py` (Claim E): CPU, 10 critics, 500 timed slots
  after warm-up gives median **0.35 ms/slot** actor + ensemble inference (p95
  0.49 ms), versus the 5 ms budget. Claim E is **supported on this laptop CPU**
  for inference only; environment stepping and the (asynchronous) LLM call are
  excluded. Artifact: `results/phase8/inference_benchmark.json`.
