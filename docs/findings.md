# Findings

Per-plan notes on each figure/experiment. One to two sentences per plot.

## Phase 1 — Channel and traffic models

- `f1_p_los_vs_elevation.png`: P_LoS rises monotonically from near 0 at low elevation to ~1 at broadside (90 deg), matching the Al-Hourani urban curve (a=9.61, b=0.16).
- `f2_path_loss_vs_distance.png`: mean path loss grows with distance and lies between the LoS-only (1 dB env loss) and NLoS-only (20 dB env loss) bounds, weighted by P_LoS.
- `f3_snr_sample.png`: received SNR per class is stable under nominal conditions and drops by roughly the +15 dB shadow during the shaded window (slots 150-350), as expected from the shadowing hook.
- `f4_queue_overload.png`: with service below arrivals, all three queues grow roughly linearly; Emergency grows slowest (lowest arrival rate), confirming the fluid queue update direction.

## Phase 0 — Tooling

- `pip install -e ".[dev]" && pytest -q` passes on a clean venv. Python 3.13.5 used (plan specified 3.11, not installed; logged in docs/decision_log.md).
