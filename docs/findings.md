# Findings: Lin et al. 2025 reproduction

Scientific results of the Lin et al. 2025 reproduction (Rule 3). Every number comes from a stored artifact named next to it. The old env-v2 findings were not restored: they are kept only in git tag `archive/env-v2` and are not results of this project (Rule 4).

## F-1 | 2026-11-18 | Completeness of the 75 training runs (Stage R3)

Source: `results/completeness_2026-11-18/completeness.csv` (TE-063).

- 75/75 runs (Table 2: 30; Fig. 5 sweep, A-41: 45) are complete: 20000 episodes, `train_log.csv` covers episodes 0-19999 exactly once, commit 4dee668 with `git_dirty` false in every segment, same config hash across seeds of a configuration.
- T1/T2 (A-39 definitions, reported only): T1 passes in 73/75 runs, T2 in 74/75; no NaN/inf. All 30 Table 2 runs pass both, including MAPPO at 20000 episodes (not covered by the A-39 pilot).
- The two failures are both FOS_MADDPG_NOMA sweep runs: `fos_maddpg_sic_n4_seed2` fails T1 (Delta -8.292 pp); `fos_maddpg_sic_n6_seed0` fails T1 and T2 (Delta -11.417 pp, best - last validation success 15.25 pp). Per A-41 they stay in the A-38 test.

## F-2 | 2026-11-18 | Test-seed evaluation: reproduction gate NOT passed

Source: `results/eval_test_2026-11-18/` (`summary.csv` sha256 440b0ff9af28..., `gate.json` 9059637a964c..., `per_run.csv` a52acd3e7fbf...); TE-065. Test seeds 5000-5099, 100 episodes per run, greedy policy, `final.pt`, Local_Computing on the same seeds (A-36).

Success rate at N = 8 (mean over 5 training seeds; paper target from `summary.csv`):

| Method | Ours (mean +/- std) | Paper |
|---|---|---|
| POS_MADDPG_NOMA | 0.9927 +/- 0.0025 | 0.8563 |
| FOS_MADDPG_NOMA | 0.9900 +/- 0.0033 | 0.6088 |
| POS_MAPPO_NOMA | 0.9911 +/- 0.0050 | 0.714 (A-35) |
| POS_MADDPG_noSIC | 0.9904 +/- 0.0082 | 0.5982 |
| FOS_MADDPG_noSIC | 0.9882 +/- 0.0047 | 0.4211 |
| POS_MAPPO_noSIC | 0.9933 +/- 0.0061 | 0.3462 |
| Local_Computing | 0.6271 | 0.6125 (A-32) |

Mean latency at N = 8: learned methods 1.23-1.56 ms (paper 7.48-13.5 ms); Local_Computing 18.72 ms (paper 18.76 ms).

Reproduction gate (spec Section 7), as computed by `gate.json`:

| Item | Status | Detail |
|---|---|---|
| 1. Success ranking | FAIL | POS_MAPPO_NOMA 0.99113 < POS_MAPPO_noSIC 0.99328; the other three comparisons hold |
| 2. SIC helps | FAIL | same POS_MAPPO pair; FOS_MADDPG and POS_MADDPG hold |
| 3. Fig. 5 trend (A-38) | FAIL | first run: not_evaluable (script defect P-021); after the fix, recomputed from the same per_run.csv (`results/eval_test_2026-11-18_gate_fix/gate.json`, TE-066; reference copy from the clean commit 96215d0, git_dirty false, identical items: `results/eval_test_2026-11-18_gate_clean_96215d0/gate.json`, TE-069): mean success slope -0.145 / -0.100 / -0.227 pp per sensor (POS_MADDPG / FOS_MADDPG / POS_MAPPO; threshold <= -1.0, FOS seed 0 slope +0.022), mean latency slope 0.098 / 0.104 / 0.107 ms per sensor (threshold >= +0.3); "FOS best" holds at N = 2 (tie 0.9975) and N = 4 |
| 4. POS_MADDPG within +/- 10 pp of 85.63% | FAIL | 0.9927, difference 13.64 pp |

Reading (plain): Local_Computing matches the paper, so the calibrated environment part holds, but every learned method saturates near 99% success with about 1-2 ms latency, far above the paper. All N = 8 differences between learned methods are below 0.5 pp, smaller than the noise of 100 test episodes, so the ranking and SIC items cannot separate the methods. This agrees with P-003 (offloading is easier in our environment than in the paper). The Fig. 5 trend mostly has the right sign (mean success slope negative, latency slope positive; one FOS seed has a positive success slope) but is about 4-10 times flatter than the success threshold and about 3 times flatter than the latency threshold, for the same saturation reason. All four gate items fail. Conclusion: the reproduction is NOT achieved; this is reported as a negative result and is not tuned away (spec Section 7). It stays the conclusion until new evidence exists (HUMAN, 2026-11-18).

Other observations from `summary.csv`: learned success is 0.946-0.998 at every N in {2, 4, 6, 8, 10, 12} against Local_Computing 0.6158-0.6271; the lowest mean is FOS_MADDPG_NOMA at N = 6 (0.9462), pulled down by `fos_maddpg_sic_n6_seed0` (0.844), the run that failed T1/T2 in F-1.

## F-3 | 2026-11-18 | S-3 sensitivity of offloading difficulty to unstated readings (validation seeds only)

Source: `results/s3_sensitivity_2026-11-18/` (`summary.csv`, `per_seed.csv` 12 600 rows, `manifest.json`: commit 96215d0 + uncommitted S-3 code, diff sha256 76dd20fe..., untracked files hashed); registration spec Section 11 S-3; audit `docs/lin2025_audit.md` (S-2); TE-072, TE-074-TE-076. Validation seeds 2000-2099, non-learning policies only, POS. Effect = paired difference to V0 (v1.0), mean and 95% bootstrap CI over the 100 seeds. These are descriptive sensitivities, not reproduction results, and no variant is claimed to be the paper's model.

V0 (v1.0) levels: NearestUAVFullPower success 0.998 / 0.989 / 0.980 at N = 2 / 8 / 12 (latency 1.0-1.9 ms); RandomPolicy 0.770 / 0.754 / 0.749; GreedyLocalOrOffload 0.998-0.995; LocalComputing 0.609 / 0.627 / 0.625. LocalComputing was identical to V0 under every variant (control passed).

Success difference vs V0 (percentage points), NearestUAVFullPower, N = 2 / 8 / 12:

| Variant (S-2 verdict on the current reading) | N = 2 | N = 8 | N = 12 |
|---|---|---|---|
| N1-seq shared carrier, sequential depletion (current LESS literal; rule unstated) | 0.00 | -21.39 [-24.18, -18.71] | -39.05 [-40.66, -37.47] |
| N1-prop shared carrier, proportional rescaling (same) | 0.00 | -1.56 [-1.91, -1.23] | -4.45 [-4.97, -3.93] |
| N1-eq equal split, bound (same) | 0.00 | -1.56 (identical to N1-prop, as registered) | -4.45 |
| A3-total 62.5 MHz per subcarrier (current more literal, not decisive) | -5.95 [-7.1, -4.8] | -8.18 [-8.9, -7.48] | -11.87 [-12.56, -11.19] |
| A8-lit literal Eq. 5 (tie) | 0.00 | -1.15 [-1.43, -0.90] | -2.28 [-2.58, -1.98] |
| A1-const constant gain (tie) | +0.20 | +0.15 (CI includes 0) | +0.22 (CI includes 0) |
| A2-dBm / A2-psd lower noise (convention only) | +0.20 / +0.20 | +0.33 / +0.33 | +0.45 / +0.44 |
| UAV-cb UAV compute 1.4 GHz shared (bound; paper says negligible) | -46.05 [-49.3, -42.8] | -91.07 [-91.83, -90.31] | -95.27 [-95.95, -94.58] |

RandomPolicy (N = 2 / 8 / 12): A3-total -19.00 / -19.75 / -21.36; UAV-cb -25.00 / -45.26 / -49.58; N1-seq -1.00 / -5.15 / -7.78; N1-prop -0.15 / -0.99 / -2.00; N1-eq +7.75 / +6.70 / +5.83 (it ignores the requested fraction, so single-sensor groups get the whole carrier); A1-const +8.75 / +8.16 / +7.92; A2-dBm +8.40 / +7.90 / +7.58; A2-psd +7.65 / +7.39 / +7.02; A8-lit no clear change. A8-lit with SIC off vs V0 with SIC off (NearestUAVFullPower): no clear change at any N.

Conclusion per assumption (descriptive; the paper does not give enough information to confirm any of these rules):
- **N1 (shared carrier bandwidth)**: the assumption the audit found less literal in v1.0 matters only when more than 4 sensors share a UAV, and its size depends almost entirely on the unstated sharing rule: small with proportional or equal sharing (-1.6 / -4.5 pp at N = 8 / 12), large with sequential depletion (-21 / -39 pp). Which rule Lin used cannot be determined from the paper.
- **A3 (bandwidth per subcarrier)**: the total-split reading makes offloading moderately harder (-6 to -12 pp for the full-power heuristic, about -20 pp for random); the audit judged the current per-carrier reading more literal.
- **A8 (Eq. 5)**: small effect (at most -2.3 pp), only through equal-power sensors that are never cancelled.
- **A1, A2**: no clear change for the full-power heuristic (it is already near saturation); they make random offloading easier (+7 to +9 pp).
- **UAV computing**: the only variant that removes the saturation (-46 to -95 pp), but it contradicts the paper's thrice-stated "negligible computation time" and uses our local c_cyc; it is a bound, not a candidate reading.
- No variant is made a default; v1.0 and the gate_fix result are unchanged; nothing here is compared with Lin's learned-method numbers (Rule 4). A v1.1 environment would need a reading with literal support (spec Section 11) and HUMAN approval.

## F-4 | 2026-11-18 | HUMAN assessment of S-3 and decision on v1.1 (P3 not approved)

- **Shared bandwidth**: bandwidth as a shared resource is a well-founded difference between the v1.0 code and the paper's description (S-2, `docs/lin2025_audit.md`). However, the allocation rule is not determined by the paper. S-3 (F-3) shows that the results are sensitive to this rule, so there is not enough basis to build a v1.1 environment that represents the paper's reproduction.
- **N1-seq** cannot be chosen as the default sharing rule: its large drop (-21.4 / -39.1 pp at N = 8 / 12) depends on the arbitrary sensor order (earlier sensors take most or all of the carrier); it is one specific allocation policy, not a rule derivable from the paper.
- **N1-prop / N1-eq** are candidates worth studying further but are not confirmed as the correct rule; their small drop (-1.6 to -4.5 pp) suggests that a resource limit alone may not remove the offloading saturation; they are mathematically identical for NearestUAVFullPower (all fractions = 1).
- **UAV-cb** is not a candidate for the reproduction environment: it removes the saturation but contradicts the paper's statement that UAV computation time is negligible, so it is not used to justify an environment change.
- **A1 / A2**: "no clear change" for NearestUAVFullPower does not mean they are unimportant: RandomPolicy gains +7 to +9 pp, so their effect depends on the policy and the operating regime.
- **Decision (HUMAN, 2026-11-18)**: P3 (v1.1) not approved. It may be reopened only if an original source or appendix describing the allocation rule is found; no v1.1 is created just to have a new environment version.

## F-5 | 2026-11-18 | S-1 extended test evaluation (secondary, 1000 test episodes): same gate status as the primary result

Source: `results/eval_test_ext1000_2026-11-18/` (`summary.csv` sha256 9d08e1509e14..., `gate.json` 7c7e58fc4849..., manifest: commit 96215d0, git_dirty false, seeds 5000-5999); spec Section 11 S-1; TE-078. Secondary analysis: it does not replace the primary result (gate_fix, F-2).

N = 8 success (mean +/- std over 5 training seeds), 1000 test episodes: POS_MADDPG_NOMA 0.9930 +/- 0.0026, FOS_MADDPG_NOMA 0.9901 +/- 0.0037, POS_MAPPO_NOMA 0.9912 +/- 0.0052, POS_MADDPG_noSIC 0.9904 +/- 0.0078, FOS_MADDPG_noSIC 0.9884 +/- 0.0042, POS_MAPPO_noSIC 0.9935 +/- 0.0060, Local_Computing 0.6214.

Gate items 1-4: all FAIL, as in the primary result; no comparison changed sign. Items 1 and 2 fail only on POS_MAPPO_NOMA < POS_MAPPO_noSIC (0.99116 vs 0.99355, gap 0.24 pp at 1000 episodes vs 0.22 pp at 100). Item 4: |0.99297 - 0.8563| = 0.137 > 0.10. Item 3 mean success slopes -0.134 / -0.081 / -0.208 pp per sensor (POS_MADDPG / FOS_MADDPG / POS_MAPPO; threshold <= -1.0), latency slopes 0.098 / 0.104 / 0.107 ms per sensor (threshold >= +0.3); "FOS best" holds at N = 2 (tie) and N = 4.

Reading: ten times more test episodes reproduce the primary gate result. The ranking/SIC margins at N = 8 (0.16-0.29 pp, and the 0.24 pp POS_MAPPO inversion) are smaller than the spread over training seeds (std 0.26-0.78 pp), so these comparisons are dominated by training-seed variation, not by test-episode sampling. The negative reproduction result (F-2) stands.

## F-6 | 2026-11-18 | Lin's parameter sources [7], [18], [19] do not determine the open readings; P3 stays closed

Source: `docs/lin2025_audit.md` Addendum A; TE-079. None of Lin's Table 1 values (250 MHz, -50 dB, -100 dB, 1.4 GHz, ...) appears in R-6, R-7 or R-8; R-7 is off-topic (removed from the active references). R-6 literally states a per-UAV budget (bandwidth fractions sum to at most 1) in an FDMA model, without an enforcement rule for independent agents; this supports the constraint behind N1 but not any allocation rule. Decision (HUMAN, 2026-11-18): P3 (v1.1) stays closed; v1.0 remains the reference environment.

## F-7 | 2026-11-18 | S-4: the learned methods are equivalent within about 1 pp at N = 8

Source: `results/s4_equivalence_2026-11-18_final/{primary,s1_1000ep}/pairs.csv` (identical to the worker's first output); spec Section 11 S-4; TE-082. TOST with Welch t, 90% CI, margin 1 pp (chosen after F-2 was known; the margin-free smallest equivalence margin is also reported). Primary (100 test episodes; S-1 gives the same conclusions):

| Pair (A - B) | Diff (pp) | 90% CI (pp) | Equivalent within 1 pp | Smallest margin (pp) |
|---|---|---|---|---|
| POS_MADDPG_NOMA - FOS_MADDPG_NOMA | +0.27 | [-0.07, 0.62] | yes | 0.62 |
| POS_MADDPG_NOMA - POS_MAPPO_NOMA | +0.16 | [-0.33, 0.64] | yes | 0.64 |
| POS_MADDPG_NOMA - POS_MADDPG_noSIC | +0.23 | [-0.55, 1.01] | no (TOST p 0.052) | 1.01 |
| FOS_MADDPG_NOMA - FOS_MADDPG_noSIC | +0.18 | [-0.30, 0.66] | yes | 0.66 |
| POS_MAPPO_NOMA - POS_MAPPO_noSIC | -0.22 | [-0.87, 0.44] | yes | 0.87 |

Reading: in `lin-env-v1` the learned methods and their noSIC variants perform the same within about 1 pp; the one non-equivalent pair misses only because the POS_MADDPG noSIC runs vary more across training seeds (sd 0.82 pp). n = 5 per group, so the test has low power; descriptive only.

## F-8 | 2026-11-18 | S-5: the learned policies offload everything at near-full power/bandwidth, but do not pick the nearest UAV

Source: `results/s5_policy_diagnostics_2026-11-18/` (per_run.csv sha256 323f8fbedc34...); spec Section 11 S-5; TE-083. 15 runs (POS_MADDPG, FOS_MADDPG, POS_MAPPO, SIC, N = 8, seeds 0-4), validation seeds 2000-2099, greedy.

- 0/15 runs are "close to NearestUAVFullPower" by the registered rule. They offload 97-100% of tasks with mean power_frac 0.99-1.00 (MADDPG) or 0.73-0.95 (MAPPO) and mean bw_frac 0.95-1.00, but their UAV choice agrees with the nearest UAV only 47-52% of the time, about chance with two UAVs.
- Their success is within about 1 pp of the heuristic (per method +0.21 to +0.50 pp; runs -0.39 to +0.83 pp).
- Reading: the saturation near 99% does not come from learning a clever policy; with v1.0's link capacity, "offload everything at (near) full power to some UAV" already succeeds, and which UAV is chosen barely matters. This is direct evidence for P-003 (offloading too easy) and explains F-2 and F-7. An informal check on one run (not a registered metric) suggests each sensor keeps one fixed UAV regardless of position.

## F-9 | 2026-11-18 | Decision: P3 reopened under a new condition; lin-env-v1.1 = v1.0 + A-42 (our assumption)

HUMAN decision 2026-11-18, superseding the P3 part of F-4/F-6: the per-carrier constraint "sum of bandwidth fractions <= 1" (supported by Lin's text and R-6) is accepted, enforced by proportional rescaling and recorded as our assumption A-42, not as a rule confirmed by Lin (spec Sections 6 and 12). v1.1 is meant as a text-closer base for the extension, not as a reproduction of Lin's numbers; v1.0 and its results stay unchanged as the reproduction milestone. Pilot training and retraining are not approved yet.

## F-10 | 2026-11-18 | lin-env-v1.1 (A-42) with non-learning policies: offloading still close to saturation

Source: `results/v1_1_heuristics_2026-11-18/` (summary.csv; manifest: commit 7df202e + uncommitted v1.1 code, git_dirty true, diff and untracked files hashed); spec Section 12 check 7; TE-087. Validation seeds 2000-2099, POS, SIC; paired v1.1 - v1.0 difference with 95% bootstrap CI.

| Policy | N = 2 | N = 8 | N = 12 |
|---|---|---|---|
| NearestUAVFullPower v1.0 -> v1.1 | 0.998 -> 0.998 | 0.989 -> 0.973 (-1.56 pp [-1.91, -1.23]) | 0.980 -> 0.935 (-4.45 pp [-4.98, -3.93]) |
| RandomPolicy v1.0 -> v1.1 | 0.770 -> 0.769 | 0.754 -> 0.744 (-0.99 pp) | 0.749 -> 0.729 (-2.00 pp) |
| GreedyLocalOrOffload v1.0 -> v1.1 | 0.998 -> 0.998 | 0.997 -> 0.997 | 0.995 -> 0.992 |
| LocalComputing | identical | identical | identical |

v1.1 equals the S-3 N1-prop variant exactly (independent implementations). Reading: A-42 makes offloading only slightly harder (mostly at N = 12); the full-power heuristic stays at 93-99%, so learned methods on v1.1 are expected to stay near saturation (to be checked by the pilot if approved). Latency of the heuristic rises (N = 12: 1.93 -> 3.90 ms).
