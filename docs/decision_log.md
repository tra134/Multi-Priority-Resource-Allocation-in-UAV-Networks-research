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
