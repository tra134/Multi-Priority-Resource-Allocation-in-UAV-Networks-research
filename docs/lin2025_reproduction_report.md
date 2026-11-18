# Reproduction report: Lin et al. 2025 (IET Communications, DOI 10.1049/cmu2.70063)

Status 2026-11-18. Every number below is copied from `docs/findings.md` (F-1 to F-8), which names the artifact behind it. This report adds no new number. Decisions are the HUMAN's and are dated.

## 1. Summary

We rebuilt the system model of Lin et al. 2025 (multi-UAV MEC, NOMA uplink with SIC, MADDPG/MAPPO with POS/FOS offloading) from the paper alone (no public code), calibrated the values the paper leaves out with a non-learning baseline only, trained 75 runs (Table 2: 30; Fig. 5 N-sweep: 45) and evaluated them once on test seeds.

**The reproduction is not achieved: all four pre-registered gate items fail** (spec Section 7; F-2). Local_Computing matches the paper (62.7% vs 61.25% success, 18.72 vs 18.76 ms), but every learned method saturates near 99% success with about 1-2 ms latency, far above the paper's 34.6-85.6%. In our environment the methods are equivalent within about 1 pp (F-7), and the trained policies are "offload everything at near-full power to some UAV" (F-8): offloading is too easy (P-003). The paper does not give enough information to locate the cause with certainty (S-2 audit, F-6); the most plausible lever (bandwidth shared on a carrier) has no stated allocation rule, and the results are very sensitive to that rule (F-3). We therefore report a negative result and keep the environment unchanged (v1.0; P3/v1.1 not approved).

## 2. What was built and how it was checked

- **Environment `lin-env-v1`** (frozen 2026-11-13, commit 27db5c1): spec `docs/lin2025_spec.md`; every unstated value is an assumption with an ID (Section 6). Calibration used only Local_Computing against targets written before the run (Section 5): c_cyc = 24.5 cycles/bit (A-33), kappa (A-34).
- **Learning code** (commit 4dee668): MADDPG (Gumbel-Softmax) and MAPPO, POS/FOS, SIC/noSIC; 20 000 episodes each (A-23, locked by the pre-registered A-39 pilot).
- **Runs**: Table 2 at N = 8, 6 configurations x seeds 0-4; Fig. 5 sweep N in {2, 4, 6, 10, 12} x 3 methods x seeds 0-2 (A-41). All 75 complete and clean (F-1): 20 000 episodes, no log gaps, one commit, git_dirty false. T1/T2 training checks pass in 73/75 and 74/75 runs; the two failures are FOS_MADDPG sweep runs (N = 4 seed 2; N = 6 seed 0), kept in the analysis as registered.
- **Evaluation**: greedy policy, `final.pt`, 100 test episodes (seeds 5000-5099), Local_Computing paired on the same seeds; once, on 2026-11-18 (TE-065). A script defect in gate item 3 (P-021) was fixed with a regression test and the gate recomputed from the stored results (no new test-seed evaluation); reference gate: `results/eval_test_2026-11-18_gate_clean_96215d0/` (clean commit).

## 3. Results against the paper

N = 8 success rate (mean over 5 training seeds; F-2):

| Method | Ours | Paper |
|---|---|---|
| POS_MADDPG_NOMA | 0.9927 | 0.8563 |
| FOS_MADDPG_NOMA | 0.9900 | 0.6088 |
| POS_MAPPO_NOMA | 0.9911 | 0.714 |
| POS_MADDPG_noSIC | 0.9904 | 0.5982 |
| FOS_MADDPG_noSIC | 0.9882 | 0.4211 |
| POS_MAPPO_noSIC | 0.9933 | 0.3462 |
| Local_Computing | 0.6271 | 0.6125 |

Gate (spec Section 7): item 1 (ranking) FAIL, item 2 (SIC helps) FAIL, item 3 (Fig. 5 trend) FAIL, item 4 (POS_MADDPG within 10 pp of 85.63%) FAIL (difference 13.6 pp). Items 1-2 fail only on POS_MAPPO NOMA < noSIC (0.991 vs 0.993); the trend in item 3 has mostly the right sign but is 4-10 times flatter than the threshold. With ten times more test episodes (S-1, 1000 episodes) every item and every comparison sign is unchanged (F-5), so the outcome is not a test-sampling artefact.

## 4. Why it fails: evidence

1. **Offloading is too easy in v1.0 (P-003, known since 2026-11-13).** The non-learning NearestUAVFullPower heuristic already reaches about 99% on validation seeds.
2. **The learned policies are trivial (F-8).** All 15 diagnosed runs offload 97-100% of tasks at near-full power and bandwidth; their UAV choice is unrelated to distance (agreement with the nearest UAV 47-52%, chance level with two UAVs); success is within about 1 pp of the heuristic.
3. **Methods cannot be separated (F-7).** 4 of 5 ranking/SIC pairs are equivalent within 1 pp; the fifth misses by 0.01 pp because of training-seed spread. The gate's ranking and SIC items test differences that do not exist in this environment.
4. **Where the paper is under-specified (S-2 audit, `docs/lin2025_audit.md`).** The v1.0 code gives every co-channel sensor a full carrier; five passages of the paper treat bandwidth as a shared "available" resource (less literal in v1.0). Other readings (A-2 noise, A-3 bandwidth per carrier, negligible UAV computation) are the most literal ones; A-1 (channel gain) and A-8 (Eq. 5) are ties. Lin's cited parameter sources [7], [18], [19] contain none of Lin's Table 1 values; [7] states a per-UAV budget (fractions sum to at most 1) but no allocation rule, and [18] is off-topic (F-6).
5. **Sensitivity (F-3, validation seeds, non-learning policies).** For the full-power heuristic at N = 8 / 12: shared bandwidth with proportional or equal split -1.6 / -4.5 pp; sequential depletion -21 / -39 pp (depends on an arbitrary sensor order); total-bandwidth split (A-3) -8 / -12 pp; literal Eq. 5 -1.2 / -2.3 pp; channel/noise readings no clear change; UAV computing time (contradicts the paper, bound only) -91 / -95 pp. The size of the effect depends on rules the paper does not state.

## 5. Threats to validity

- **Under-specification**: the allocation rule for shared bandwidth, the channel model (A-1) and the units of the noise power (A-2) are not determinable from the paper or its cited sources; any v1.1 would rest on new assumptions.
- **Internal conflicts in the paper**: Table 2 vs Fig. 4/5 and text (A-35); Table 4 inconsistencies (UAV assignment, swapped values).
- **Training budget**: 20 000 episodes is our value (A-23), locked by the pilot; the paper's Fig. 4 axis shows 10 000.
- **Statistics**: 5 training seeds per configuration; the training-seed spread (0.26-0.78 pp) exceeds the differences between methods.
- **Margins chosen after seeing data**: the 1 pp equivalence margin (F-7) was fixed after F-2 was known; the margin-free smallest equivalence margin is reported alongside.
- **Two FOS_MADDPG sweep runs failed the T1/T2 training checks**; they stay in the analysis as registered.

## 6. Decisions (HUMAN)

- 2026-11-18: negative result recorded; extension (`slide/research.pdf`) not started.
- 2026-11-18: P3 / v1.1 not approved; reopen only if a source describing Lin's allocation rule is found (F-4, F-6).
- Result milestones kept unchanged: v1.0 (`results/eval_test_2026-11-18/`), gate_fix (`results/eval_test_2026-11-18_gate_fix/`, `..._gate_clean_96215d0/`); v1.1 not created.
- 2026-11-18 (later): P3 reopened with a new condition. The per-carrier constraint (sum of bandwidth fractions <= 1) is accepted, and its enforcement is recorded as A-42, our assumption and not a rule confirmed by Lin. lin-env-v1.1 is specified (spec Section 12) and implemented (commit f29255c; F-10). The v1.1 pilot (step 3) is approved and running; the 75-run retraining (step 4) is NOT approved.
- 2026-11-19: the end-of-cycle handling of the learners (truncation with bootstrap) is recorded as A-43, with no code change and no retraining; termination without bootstrap is listed as future sensitivity analysis FS-1.

## 7. Lessons

- Calibrating the environment with a non-learning baseline (Local_Computing) worked; it did not constrain the offloading side, because the paper publishes no non-learning offloading number.
- A trivial heuristic should be run against the paper's best number before any training: here it exposed the saturation (P-003) on day one, and S-5 confirmed that the learned policies add nothing over it.
- Pre-registering the gate, the pilot rule and every secondary analysis kept the negative result clean: no value was tuned toward the paper.

## 8. Artifact index

| What | Where |
|---|---|
| Spec, assumptions, gate, secondary registrations | `docs/lin2025_spec.md` (Sections 5-11) |
| Findings F-1..F-8 | `docs/findings.md` |
| Assumption audit + Addendum A | `docs/lin2025_audit.md` |
| References | `reference.md` (R-1..R-6, R-8; R-7 removed) |
| Completeness check | `results/completeness_2026-11-18/` |
| Primary test evaluation (v1.0) | `results/eval_test_2026-11-18/` |
| Reference gate (clean commit) | `results/eval_test_2026-11-18_gate_clean_96215d0/` |
| S-1 (1000 test episodes) | `results/eval_test_ext1000_2026-11-18/` |
| S-3 sensitivity (+ clean rerun) | `results/s3_sensitivity_2026-11-18/`, `..._clean_7df202e/` |
| S-4 equivalence | `results/s4_equivalence_2026-11-18_final/` |
| S-5 policy diagnostics | `results/s5_policy_diagnostics_2026-11-18/` |
| Models and checkpoints | Google Drive `uavmec/table2`, `uavmec/laptop_results_backup`, `uavmec/laptop_ckpt_backup` |
| Decision and work log | `trialAndError.md` (TE-034..), `problemsAndFixes.md` (P-016..P-022) |
