# Multi-Priority-Resource-Allocation-in-UAV-Networks-research

Reproduction of the system model of **Lin et al. 2025**, "Deep Reinforcement Learning-Based Intelligent Resource Management in Multi-UAVs-Assisted MEC Emergency Communication System", *IET Communications* 19:e70063 ([DOI 10.1049/cmu2.70063](https://doi.org/10.1049/cmu2.70063)). The paper has no public code, so the environment is rebuilt from its text.

The plan has two parts:
1. Reproduce Lin's environment and baselines, and check them against the paper's numbers (stages R0-R3).
2. Extend that environment with multi-priority tasks, an ensemble critic and an LLM arbiter (stages R4-R8).

The extension is **not started**. Start here: `CLAUDE.md` (rules), `plan.md` (stages), `docs/lin2025_spec.md` (environment contract).

## Status (2026-11-18)

- **Environment `lin-env-v1` (v1.0)** was frozen on 2026-11-13. It has K = 2 UAVs and M = 4 subcarriers, and uses NOMA uplink with SIC. Every value the paper does not state is recorded as an assumption with an ID (A-n) in spec Section 6. The unstated values were calibrated only against Local_Computing.
- **Learning baselines**: MADDPG (Gumbel-Softmax) and MAPPO, each in POS and FOS mode, with and without SIC. In total 75 runs of 20 000 episodes each: Table 2 at N = 8, and the Fig. 5 sweep over N.
- **Reproduction not achieved.** All four pre-registered gate items fail.
  - Local_Computing matches the paper: 62.7% success here vs 61.25% in the paper.
  - Every learned method saturates at about 99% success; the paper reports 34.6-85.6%.
  - The trained policies offload everything at near-full power, which is no better than a simple heuristic.
  - The paper does not state the rule for bandwidth sharing on a carrier, and the results are sensitive to that rule.
  - Full write-up: `docs/lin2025_reproduction_report.md`; findings: `docs/findings.md`.
- **Environment `lin-env-v1.1`** is v1.0 plus assumption A-42: sensors on the same (UAV, subcarrier) share its bandwidth in proportion to their requested fractions. A-42 is our own assumption; the paper does not confirm it. It is selected with `--env-config configs/lin2025_v1_1.yaml`, and v1.0 stays the default everywhere. A pilot training run on v1.1 is in progress (spec Section 12).
- The milestones `v1.0`, `gate_fix` and `v1.1` are kept side by side. Old results are never overwritten.

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -U pip
venv/bin/pip install -e ".[dev]"
```

The shell may deactivate the venv when you change directory, so call `venv/bin/...` directly.

## Tests and lint

```bash
# the machine lags at 100% CPU: run niced, 2 threads, one pytest at a time
nice -n 10 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 venv/bin/python -m pytest -q
venv/bin/python -m ruff check src scripts tests
```

## Main scripts

Every script has `--help`. All output folders must be given explicitly, and most scripts refuse an existing folder. Test seeds (5000-5999) are used only for final evaluation; analyses use validation seeds (2000-2999).

| Script | Purpose |
|---|---|
| `scripts/calibrate_local.py` | Calibrate c_cyc and kappa on Local_Computing (spec Section 5) |
| `scripts/evaluate_baselines.py` | Non-learning baselines (Local, Random, heuristics) on validation seeds |
| `scripts/train.py` | Train MADDPG or MAPPO (`--algo`, `--mode POS/FOS`, `--sic/--no-sic`, `--n-sensors`, `--seed`, `--env-config`, `--resume`) |
| `scripts/check_pilot.py` | Apply the A-39 pilot rule (T1/T2 training checks) |
| `scripts/evaluate_learned.py` | Greedy evaluation of trained runs, plus the reproduction gate (spec Section 7) |
| `scripts/recompute_gate.py` | Recompute `gate.json` from a stored `per_run.csv` |
| `scripts/run_s3.py` | S-3 sensitivity analysis of unstated assumptions (`configs/s3_variants.yaml`) |
| `scripts/s4_equivalence.py` | S-4 equivalence test (TOST) between learned methods |
| `scripts/diagnose_policies.py` | S-5 diagnostics of what the learned policies do |
| `scripts/run_heuristics.py` | Non-learning policies on v1.0 vs v1.1 |

Example (smoke run into a scratch folder):

```bash
venv/bin/python scripts/train.py --algo maddpg --n-sensors 2 --episodes 3 --set eval_episodes=1 --out /tmp/smoke
```

Long training runs on Colab (`notebooks/colab_train.ipynb`) or on the laptop inside a resource-limited systemd unit. See `trialAndError.md` and `problemsAndFixes.md` (P-016) for the exact commands.

## Repository layout

```
configs/      lin2025.yaml (v1.0), lin2025_v1_1.yaml (v1.1), train_lin2025.yaml, s3_variants.yaml
src/uavmec/   env/ (environment: channel, NOMA, tasks, energy), agents/ (MADDPG, MAPPO),
              baselines/ (non-learning policies), eval/ (metrics, runner), config.py
scripts/      entry points (table above)
tests/        pytest suite
docs/         spec, audit, targets, findings, reproduction report
notebooks/    Colab training notebook
results/      outputs (gitignored, never committed)
```

## Project records

- `docs/lin2025_spec.md`: environment contract, assumptions (A-n), gate, and the registrations of every secondary analysis.
- `docs/findings.md`: scientific results (F-n). Every number in it comes from a script and a stored artifact.
- `trialAndError.md`: append-only work log (TE-n).
- `problemsAndFixes.md`: problems and their fixes (P-n).
- `reference.md`: references (R-n).
