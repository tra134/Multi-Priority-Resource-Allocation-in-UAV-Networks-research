# Lin et al. 2025: reproduction specification (contract for all workers)

**Source:** Y. Lin, Z. Xu, J. Li, J. Wang, C. Li, "Deep Reinforcement Learning-Based Intelligent Resource Management in Multi-UAVs-Assisted MEC Emergency Communication System", *IET Communications* 19:e70063, 2025. DOI 10.1049/cmu2.70063. PDF: `slide/Dai_driver/IET Communications - 2025 - Lin - ....pdf`.

**Code availability:** none. The paper says "Data available on request from the authors"; a web search on 2026-11-13 found no public code. This is therefore a **re-implementation from the paper text**. Exact numbers are not expected; trends and rankings are the gate (Section 7).

**Status:** v0.1, 2026-11-13 (v0 corrected after the W3 audit: A-12 reference, A-15 -> A-32, Fig. 5 has no energy panel). Every value is tagged with its source:
- `P:` stated in the paper (with location)
- `A-n:` assumption, listed in Section 6 with the reason
- `C-n:` value calibrated on a non-learning baseline against a pre-registered target (Section 5)

---

## 1. Scenario (P: Section 3, Table 1)

| Item | Value | Source |
|---|---|---|
| UAVs K | 2 | P: Table 1 |
| Sensors N | 8 (sweep 2, 4, 6, 8, 10, 12 for Fig. 5) | P: Table 1, Section 6.4 |
| Subcarriers per UAV M | 4 | P: Table 1 |
| Bandwidth per subcarrier B_m | 250 MHz | P: Table 1 ("Carrier bandwidth 250 MHz"); see A-3 |
| Area L x W | 100 m x 100 m | P: Table 1 ("L, W, H 100 m") |
| UAV altitude H | 100 m, fixed | P: Table 1, Section 3.3 |
| UAV max speed v_max | 50 m/s | P: Table 1 |
| Slot length xi | 1 s | P: Table 1 |
| Slots per cycle (episode) T | 10 | P: Table 1 ("time slots included in a cycle 10 s"); A-9 |
| Sensor max transmit power p_max | 1 W | P: Table 1 |
| Sensor CPU F_n | 1 GHz | P: Table 1 |
| UAV CPU | 1.4 GHz (unused: UAV compute time is negligible) | P: Table 1, Section 3.1 |
| Sensor battery E^U_max | U(100, 400) kJ | P: Table 1; A-10 |
| UAV battery E^V_max | U(100, 500) kJ | P: Table 1; A-10 |
| Task size C_n | 0.1-1.5 Mbit | P: Table 1; generation A-5 |
| Task latency requirement D_n | 1-50 ms (the text also says "proportional to C_n") | P: Table 1, Section 3.1; A-33 |
| Rate requirement R_0 | 10 Mbps | P: Table 1 |
| Noise power sigma^2 | -100 dB = 1e-10 W | P: Table 1; A-2 |
| Channel gain | -50 dB | P: Table 1; A-1 |

## 2. System model equations (P: Section 3)

Notation: sensor n, UAV k, subcarrier m, slot t.

**Task (Eq. context of Section 3.1).** Each sensor generates one task per slot: S_n(t) = (C_n(t), D_n(t)). C_n = C * d_n with d_n ~ Poisson(lambda) (A-5); D_n ~ U(1, 50) ms (A-33).

**Local computing (Eq. 1, 2).**
- T^l_n = C_n * c_cyc / F_n (A-4: the paper writes D_n/F_n; D_n is the latency requirement, so the CPU-cycle count C_n * c_cyc is used).
- z_n = kappa * F_n^2 (energy per CPU cycle).
- e^l_n = (1 - U_n) * min{ z_n * C_n * c_cyc , E_n(t) }.

**Access (Eq. 3, 4).** U_n in {0, 1, ..., K} (0 = local). Each offloading sensor uses exactly one subcarrier of one UAV: V_nkm in {0,1}, sum_{k,m} V_nkm = 1 if U_n > 0.

**Channel (A-1).** |h_nk|^2 = beta_0 / (H^2 + ||q_k - w_n||^2), beta_0 = -50 dB (1e-5) at 1 m, w_n = sensor ground position.

**NOMA with SIC (Eq. 5, 6).** On each (k, m), the received strengths g_n = |h_nk|^2 p_n are sorted in descending order and decoded strongest first. Sensor n's interference:
I_n = sum_{j != n on same (k,m)} [1 - iota_jn * d_j] * |h_jk|^2 * p_j
where iota_jn = 1 if g_j > g_n (j is decoded before n), and d_j = 1 if j was decoded successfully (A-7). SINR gamma_n = g_n / (I_n + sigma^2) (Eq. 6).
- The paper writes |h_nk|^2 inside the sum of Eq. 5; this is read as a typo for |h_jk|^2 (A-8).
- **noSIC variant:** no cancellation, I_n = sum_{j != n on same (k,m)} |h_jk|^2 p_j.

**Rate (Eq. 7).** R_n = B_n * log2(1 + gamma_n), with B_n = bw_frac_n * B_m (A-3). Rate requirement R_n >= R_0.

**Offloading time and energy (Eq. 8).** T^0_n = C_n / R_n (UAV compute time negligible, P: Section 3.1). e^0_n = p_n * T^0_n * U_n (A-11: the paper writes tau * p_n).

**Battery (Eq. 9, 10).** E_n(t+1) = E_n(t) - e^0_n - e^l_n. Recharged to E_max every cycle T.

**UAV movement and energy (Eq. 11, 12).** ||q_i - q_j|| >= d_s, ||q_k(t+1) - q_k(t)|| <= v_max * xi. UAV energy E^V_k = hover energy (A-37). UAV trajectory policy: A-13.

**Task success (Eq. 13, 14d, 15).** A task succeeds if, for its mode:
- local: T^l_n <= D_n and e^l_n <= E_n(t)
- offload: T^0_n <= D_n and e^0_n <= E_n(t) and R_n >= R_0 and the signal was decoded (d_n = 1 under SIC)

**Reward (Eq. 15), per sensor agent:** -1 if any failure condition holds, +3 otherwise.

## 3. Agents and actions (P: Sections 4, 5)

- Each sensor is an agent (CTDE: centralized critic, local actor).
- Sensor observation S_n = {x_n, y_n, E_n, C_n, D_n}; UAV state S_k = {x_k, y_k, H, B_k (available bandwidth per carrier), E_k}. The actor input is S_n plus all S_k (P: Section 5.2).
- Action A_n = {U_n, V_n, p_n, B_nm}: offload target (K+1 options, one-hot via Gumbel-Softmax), subcarrier (argmax of an M-vector), power fraction in [0,1], bandwidth fraction in [0,1] (P: Section 4.2, 5.3; actor heads of sizes (K+1, M, 2)).
- POS vs FOS (A-14): POS may choose local (U_n = 0); FOS must offload (U_n in 1..K).
- Hyperparameters (P: Table 1, Section 5.3): batch 320, actor lr 3e-4, critic lr 5e-5, buffer 2^17, gamma 0.9, hidden layer 128, ReLU6.

## 4. Baselines and published results (reproduction targets)

| Method | Success rate | Avg latency of successful tasks | Avg energy | Source |
|---|---|---|---|---|
| POS_MADDPG_NOMA | 85.63% | 7.48 ms | 5.68 mW | Table 2, Section 6.2 |
| FOS_MADDPG_NOMA | 60.88% | 8.11 ms | 4.195 mW | Table 2, Section 6.2 |
| POS_MAPPO_NOMA | 61.25% in Table 2, but ~71% in Fig. 4b / Fig. 5a and 85.63/1.20 = 71.4% in the text (paper conflict, see `docs/lin2025_targets.md`) | 12.50 ms | 11.41 mW | Table 2, Section 6.2 |
| Local_Computing | 61.25% (A-32) | 18.76 ms | 18.76 mW | Section 6.2 text ratio 85.63/1.398; Fig. 4b ~61% |
| POS_MADDPG_noSIC | 59.82% | 13.08 ms | - | Table 2 |
| FOS_MADDPG_noSIC | 42.11% | 13.50 ms | - | Table 2 |
| POS_MAPPO_noSIC | 34.62% | 11.38 ms | - | Table 2 |

Other targets: Fig. 4 (training curves, K=2, N=8; Fig. 4a reward is a per-episode sum over 10 slots: 10 x 2.43 = 24 at 85.63% success), Fig. 5 (success and latency only, no energy panel, vs N = 2..12; FOS best at N = 2, 4; POS_MADDPG best for N > 4; POS_MADDPG > 78% at N = 10), Tables 3-4 (example allocation). Figure values are digitized (visual reads with uncertainty) in `docs/data/lin2025_digitized.csv`; the full audit of this spec against the paper is `docs/lin2025_targets.md` (worker W3, 2026-11-13).

## 5. Calibration of unstated values (pre-registered 2026-11-13, before any run)

Only Local_Computing (no learning) is used. Seeds: validation range only.

| ID | Unstated value | Target | Rule |
|---|---|---|---|
| C-1 | c_cyc (CPU cycles per bit) | Local_Computing avg latency of successful tasks = 18.76 ms | grid search, closest value. **Result (T-1): 24.5** (18.74 ms) |
| C-2 | kappa (energy coefficient) | Local_Computing avg energy = 18.76 mW (energy per 1 s slot) | **Result (T-3): 1.0009e-27** with energy averaged over successful tasks (A-34). Closed form after C-1. Check: Fig. 4c and 4d show identical Local curves (ms = mJ), i.e. kappa * F^3 = 1 W, so kappa = 1e-27 is expected |

**Test T-1 (pre-registered 2026-11-13, before the run; approved by the HUMAN): deadline model A-6 vs alternative A-33.**
- A-6 (current): D_n = r * C_n, r ~ U(0.019, 0.042) s/Mbit (Section 3.1 "proportional to C_n" + Table 4).
- A-33 (candidate): D_n ~ U(1, 50) ms, independent of C_n (Table 1 "Maximum task latency requirement 1-50 ms" read literally). Conflicts with the text "proportional to C_n", so neither reading is more literal on its own; the numbers decide.
- Procedure, for each model: LocalComputing, N = 8, seeds 1000-1099; grid c_cyc from 10 to 40 cycles/bit in steps of 0.5; pick the c_cyc whose mean latency of successful tasks is closest to 18.76 ms (C-1); at that c_cyc, record the success rate and the mean energy with kappa = 1e-27.
- Targets and tolerance: latency 18.76 ms +/- 1 ms; success 61.25% +/- 5 percentage points.
- Decision: the model with the smaller |success - 61.25%| at its calibrated c_cyc wins. If A-33 wins, A-6 is deleted (spec, config, code) and A-33 takes its place. If neither is within tolerance, the winner is still adopted, P-003 stays open, and the gap is reported.

**Test T-2 (pre-registered 2026-11-13, before the run; approved by the HUMAN): offloading physics, A-1 x A-3.**
- Why non-learning: every offloading number in the paper comes from a learned method, which must not be used for calibration (CLAUDE.md Rule 4). The only non-learning offloading evidence is **Table 4**: fixed positions (Table 3, Table 4), task sizes, deadlines, and the exact actions (UAV, carrier, bandwidth %, power %) with the paper's success/fail outcome for each sensor.
- Candidates (A-2 noise 1e-10 W kept fixed): (a) A-1 distance gain beta_0/(H^2+d^2) + A-3 per_carrier (current); (b) distance gain + total_split (B = 250 MHz split into 4 x 62.5 MHz); (c) constant gain 1e-5 for every link + per_carrier; (d) constant gain + total_split.
- Procedure: replay the Table 4 slot in `LinMECEnv` (one step, N = 8, K = 2, SIC, POS, c_cyc 24.5), using the table's numbers, under two readings of "UAV1": R1 = Table 4 literally (every offloading sensor -> uav1 at (30, 53)); R2 = the text and Fig. 6 (UAV1 serves S0, S4; UAV0 at (77, 87) serves S3, S5, S6, S7). Score = number of sensors whose success/fail matches Table 4, summed over R1 and R2 (16 cases).
- Decision: the candidate with the strictly highest score replaces the current one (A-1 and/or A-3 deleted and replaced). Ties with the current candidate (a) mean no change. With only 8 sensors this is weak evidence; it is reported as such, and P-003 stays open unless the trivial NearestUAVFullPower heuristic also drops clearly below 100%.
- Descriptive only (not used for the decision): NearestUAVFullPower and RandomPolicy on validation seeds 2000-2099 under each candidate.

**Test T-3 (pre-registered 2026-11-13, before the run; approved by the HUMAN): which tasks the "average energy" metric averages over (A-27).**
- E1 (current): mean sensor energy over all tasks. E2: mean over successful tasks only, the same set as the paper's "average latency of successful tasks".
- Paper evidence (non-learning, Local_Computing): Fig. 4c and Fig. 4d Local curves have the same shape point by point (checked on a 250 dpi render), and Local energy 18.76 mW = Local latency 18.76 ms.
- Procedure: LocalComputing, N = 8, seeds 1000-1099, c_cyc 24.5. For each candidate, set kappa by C-2 (closed form so the mean energy = 18.76 mW); then compute per-episode latency (ms) and energy (mW) series.
- Decision: the candidate under which the per-episode Local energy series equals the latency series (max abs difference < 0.01, as in Fig. 4c/4d) wins; if both or neither do, the one whose C-2 kappa is closer to the F^3 check value 1e-27 wins. The winner defines `mean_energy_mw` for every method, and kappa is set to its C-2 value.

**Test T-4 (pre-registered 2026-11-13, before the run; approved by the HUMAN): offloading energy formula (A-11, A-17, A-26).**
- Candidates: O1 = physical e0 = p_n * T0_n (current A-11, transmit time capped at one slot by A-18); O2 = paper literal e0 = tau * p_n with tau = slot length 1 s (A-17); O3 = paper literal with tau read as the task's latency requirement, e0 = p_n * D_n (W3 audit item A-26: in the paper tau denotes the latency limit).
- Evidence (non-learning): the paper's stated mechanism in Section 6.2 that offloading costs the sensor less energy than local computing (only transmission energy is spent when offloading). Replay the Table 3/4 slot exactly as in T-2 (N = 8, SIC, POS, c_cyc and kappa from the current config) under readings R1 and R2. For each of the 6 offloading sensors in each reading (12 cases), compare that sensor's offloading energy under the candidate with the local-computing energy of the same task (kappa * F^2 * c_cyc * C_n).
- Score = number of the 12 cases where offloading energy < local energy. Decision: the strictly highest score wins; a tie involving O1 keeps O1. Then the `offload_energy` config switch is removed and only the winner's formula stays in code.

Local_Computing **success rate is not calibrated**; it is a validation check against 61.25% (A-32). The tolerance for this check is **not yet set** and must be approved by the HUMAN before the calibration run.

## 6. Assumptions (values the paper does not state)

| ID | Ambiguity | Assumption | Why |
|---|---|---|---|
| A-1 | "Channel gain -50 dB" without a path-loss model | beta_0 = -50 dB at 1 m, free-space LoS: beta_0 / (H^2 + d^2) | Common in UAV-MEC papers; positions would otherwise not matter. Not supported by Lin's references (refs [7], [18], [19] are cited only for CPU, battery and power values). Alternative: constant gain 1e-5 for all links. Supported by Diao et al. 2021 (Lin's ref [31]): g0 = -50 dB with a distance-based gain. |
| A-2 | Noise -100 dB, bandwidth not stated | sigma^2 = 1e-10 W per link | Literal reading |
| A-3 | "Carrier bandwidth 250 MHz" vs "B split into M subcarriers, B/M" | B_m = 250 MHz per subcarrier (decided 2026-11-13; comparison switch removed) | Table 1 and Section 6.1 ("4 subcarriers of 250 MHz bandwidth") say per carrier; Section 3 B/M is consistent with B = 1 GHz; only the Fig. 4 caption "B_max = 250" is ambiguous. Test T-2 could not separate the readings (TE-008, TE-009, TE-010) |
| A-4 | Eq. 1 T^l = D_n / F_n | T^l = C_n * c_cyc / F_n | Dimensional consistency |
| A-5 | C and lambda of the Poisson task model | C = 0.05 Mbit, lambda = 16, C_n clipped to [0.1, 1.5] Mbit | Mean 0.8 Mbit, range of Table 1 |
| A-33 | D_n distribution: Table 1 "1-50 ms" vs text "proportional to C_n" | D_n ~ U(1, 50) ms, independent of C_n | Replaced A-6 after test T-1 (Section 5): matches both the paper's Local latency and Local success; the text's "proportional" claim is not reproduced (2026-11-13) |
| A-34 | Which tasks the "average energy" metric averages over (was proposed A-27) | Successful tasks only, the same set as latency | Test T-3 (2026-11-13): with all tasks, C-2 needs kappa 9.60e-28 and the Local energy series differs from the latency series by up to 1.36 mW; with successful tasks, kappa 1.0009e-27 (F^3 check 1e-27) and the series match within 0.018, as in Fig. 4c/4d. Artifact `results/T3_energy_metric_2026-11-13/` |
| A-35 | Paper-internal conflicts in the targets: Table 2 POS_MAPPO 61.25% vs ~71% in Fig. 4b/5a and the text; Fig. 5a FOS at N = 8 ~81% vs Table 2 60.88% (was proposed A-28) | Use the Fig. 4/text values for MAPPO; flag the conflict; never use the FOS-vs-MAPPO ordering at N = 8 as a gate item | Paper is internally inconsistent; HUMAN decision 2026-11-13 |
| A-36 | Whether all methods see the same task/deadline streams (was proposed A-30) | All methods are evaluated on the same seeded task/deadline streams (paired evaluation), as `src/uavmec/eval/runner.py` already does | HUMAN decision 2026-11-13 |
| A-23 (status: **locked 2026-11-16**) | Training hyperparameters the paper does not state | Our values (not Lin et al.'s), exactly as in `configs/train_lin2025.yaml` and Section 10. **Episodes: 20000 for MADDPG and 20000 for MAPPO** (all Table 2 configurations). This is not a value from Lin et al.; their Fig. 4 x-axis shows 10000 episodes, which was our provisional value. Other A-23 values, unchanged since pre-registration: T = 10 steps, warm-up 1000, Adam, tau_soft 0.01, Gumbel tau 1.0 hard straight-through, exploration noise sigma 0.1 -> 0.01 over a fixed 50000 env steps (not rescaled with the episode count), evaluation every 500 episodes on 20 validation episodes, checkpoint every 500; MAPPO: clip 0.2, 10 epochs, GAE lambda 0.95, 32 episodes per rollout, minibatch 320, ent_coef 0.01. Values inherited from Lin et al. (P, never changed by A-23 or A-39): batch 320, actor lr 3e-4, critic lr 5e-5, buffer 2^17, gamma 0.9, actor 128 ReLU6, critic 64/32 (Section 10). Reported results use the last checkpoint `final.pt`; no best-validation checkpoint selection | Pre-registered 2026-11-13 before any training run (episodes 10000, provisional). Locked by the A-39 pilot (TE-030, TE-032; local recheck TE-033, artifacts `results/pilot_A39/`, `results/pilot_A39_check_local/`): POS_MADDPG T1 failed at 10000 (Delta -4.21 pp), extended once to 20000, T1/T2 pass (Delta +0.13 pp); POS_MAPPO T1/T2 pass at 10000 (Delta +0.19 pp). A-39 did not say whether the episode count is per algorithm; the HUMAN chose 20000 for both (2026-11-16) so that both algorithms get the same training budget in episodes (same episode count does not mean equal compute or equal number of gradient updates). MAPPO at 20000 was not checked by the pilot: T1/T2 are computed on the Table 2 runs and reported; a MAPPO failure is reported and does not change the protocol automatically |
| A-38 | Gate item 3 (Fig. 5 trend): how "success decreases and latency increases with N" and "FOS best at N = 2, 4" are checked | N fixed to {2, 4, 6, 8, 10, 12}; a missing N for a method makes item 3 not_evaluable. Methods: POS_MADDPG_NOMA, FOS_MADDPG_NOMA, POS_MAPPO_NOMA. Each training seed (3 per method and N) is evaluated greedily on 100 test episodes (seeds 5000-5099). For each method and each training seed, take the ordinary least-squares slope of that seed's metric vs N over the 6 points. Success (percentage points): PASS iff the mean of the seed slopes <= -1.0 pp per sensor AND every seed slope < 0. Latency (ms, over successful tasks): PASS iff the mean slope >= +0.3 ms per sensor AND every seed slope > 0. Seeds are matched across N by training seed index; the seed set must be the same at all 6 N, else not_evaluable. "FOS best at N = 2, 4": mean success of FOS_MADDPG_NOMA >= mean success of every other NOMA learned method at that N, no tolerance. Item 3 passes only if all these checks pass | Pre-registered 2026-11-13 before any training run. Thresholds are about 1/3 of the smallest learned-method slope digitized from Fig. 5 (POS_MADDPG -2.79 pp, +0.88 ms per sensor; MAPPO -3.03, +1.31; FOS -3.79, +1.07) and well away from Local_Computing's flat line (+0.01 pp, +0.16 ms), from `docs/data/lin2025_digitized.csv` |
| A-39 | A-23 pilot rule: how A-23 is locked | Pilot = 2 runs, POS_MADDPG_NOMA and POS_MAPPO_NOMA, N = 8, SIC, training seed 0, config A-23. Only the validation log (`eval_log.csv`, validation seeds) and the `train_log.csv` losses are inspected; paper numbers are never used. **T1 (convergence):** S_last = mean validation success of the last 3 evaluations, S_prev = mean of the 3 evaluations just before them (at 10000 episodes: 9000-10000 vs 7500-8500); Delta = S_last - S_prev; PASS iff \|Delta\| <= 2 percentage points. **T2 (stability):** FAIL if any NaN/inf in the logged losses or validation metrics, or if (best validation success over all evaluations) - (last validation success) > 5 percentage points; else PASS. **Decision:** T1 and T2 pass -> lock A-23 unchanged. \|Delta\| > 2 and T2 passes -> the only allowed change is episodes 10000 -> 20000 (eval/ckpt schedule unchanged), then T1/T2 are rechecked on the extended run. T2 fails -> stop and report to the HUMAN; no automatic change. Paper-given values (lr, batch, gamma, buffer, network sizes) are never changed. At most one extension; if T1 still fails at 20000 -> stop and report (confirmed by the HUMAN 2026-11-13, during the pilot run, before any pilot result was inspected) | Pre-registered 2026-11-13 before any training run; implemented by `scripts/check_pilot.py` |
| A-40 | Readings and units for gate items 1, 2, 4 (as implemented in `scripts/evaluate_learned.py`) | Success is stored as a fraction in [0, 1] in all outputs; std over training seeds is the sample std (ddof = 1), 0 for one seed. Item 1 checks at N = 8: POS_MADDPG_NOMA > FOS_MADDPG_NOMA, POS_MADDPG_NOMA > POS_MAPPO_NOMA, FOS_MADDPG_NOMA > FOS_MADDPG_noSIC, POS_MAPPO_NOMA > POS_MAPPO_noSIC (each method vs its own noSIC twin; the FOS-vs-MAPPO ordering is never used, A-35). Item 2: every `*_NOMA` method beats its `*_noSIC` twin at the same N (mean success). Item 4: mean over training seeds at N = 8, \|success - 0.8563\| <= 0.10 | Pre-registered 2026-11-13 before any training run; fixes the readings of Section 7 items 1, 2, 4 |
| A-41 | Fig. 5 N-sweep runs for gate item 3 (A-38 fixes the test, not the runs) | Methods POS_MADDPG_NOMA, FOS_MADDPG_NOMA, POS_MAPPO_NOMA (all SIC); training seeds 0, 1, 2 (matched across N); N in {2, 4, 6, 10, 12} trained new; at N = 8 the Table 2 runs with seeds 0-2 are reused (identical configuration); 20000 episodes and every other value as A-23 (locked). 45 new runs. The A-39 pilot only covered N = 8; T1/T2 of every sweep run are reported, never used to change the protocol | HUMAN decision 2026-11-16, before any sweep run (TE-037) |
| A-42 (environment lin-env-v1.1 only; our assumption, NOT a rule confirmed by Lin) | Shared bandwidth on a carrier: the paper treats bandwidth as a shared "available" resource (Sec 4.1 B_k(t) "available bandwidth of the UAV's m carriers"; Sec 5.3 "proportion of available bandwidth"; Table 3; Table 4 fractions on a shared carrier sum below 100%; Sec 6.4) and R-6 (Lin's ref [7]) states a budget sum of fractions <= 1, but no source gives the allocation rule for independent agents | For every group (UAV k, subcarrier m) in a slot: B_n = f_n * B / max(1, sum_{j in group} f_j) (proportional rescaling, identical to S-3 variant N1-prop); the constraint sum B_n <= B per carrier holds by construction; order-independent. Observation B_k unchanged (constant 1.0): all agents act simultaneously and bandwidth renews every slot, so the bandwidth available at decision time is the full carrier | HUMAN decision 2026-11-18 (P3 reopened with a new condition: accept the per-carrier constraint and enforce it by proportional rescaling, recorded as our assumption). Chosen over N1-seq (depends on an arbitrary sensor order) and N1-eq (ignores the requested fraction). Evidence: S-2 audit + Addendum A, S-3 (F-3), F-4, F-6. Replace by Lin's rule if the authors answer (CLAUDE.md "Replacing an assumption") |
| A-43 (recorded 2026-11-19; code unchanged) | What happens at the end of a cycle T (episode = 10 slots, A-9). The paper recharges batteries every cycle (Eq. 9-10) but does not say whether the learner treats the cycle end as a terminal state | Time-limit **truncation**: `terminated` is always False and `truncated` is True at t = T; the critic still bootstraps from the next state (MADDPG stores terminal = 0, `agents/maddpg.py` `observe`; MAPPO cuts the GAE trace at the episode end but keeps the value bootstrap, `agents/mappo.py`). Alternative: **termination** at t = T (no bootstrap) | Was an implicit choice of the code since R3, found by an external review (TE-095), recorded with HUMAN approval 2026-11-19. Not a confirmed bug: with gamma = 0.9 and T = 10 the effect is expected to be small, but it is unmeasured. Not changed, nothing retrained; listed as future sensitivity analysis FS-1 (Section 11) |
| A-44 (registered 2026-11-20, pending test T-5; Section 13) | Unstated extra loss in the link budget | Extra link loss L dB on every sensor-UAV link: |h_nk|^2 = beta0 * 10^(-L/10) / (H^2 + d^2); L = 0 is v1.0. Equivalent to raising the noise by L dB. beta0 stays -50 dB (Table 1 literal), A-1 distance law unchanged | NOT stated by Lin. Candidate reading (antenna or implementation loss). Motivation: P-003 and test T-2 (MU6 of Table 4 predicted to succeed, paper says it fails). Value decided by T-5 only; if 0 is feasible or the feasible set is empty, A-44 is not adopted |
| A-7 | SIC decoding failure | d_j = 1 iff R_j >= R_0 at decoding time | Paper ties success to the rate threshold |
| A-8 | Eq. 5 uses |h_nk|^2 for interferers | Use |h_jk|^2 | Physical meaning |
| A-9 | "Number of time slots in a cycle: 10 s" | Episode = 10 slots of 1 s | Literal reading |
| A-10 | Battery capacity ranges | Sampled per episode | Literal reading; with mJ-scale task energy the battery constraint is almost never active (note in report) |
| A-11 | Eq. 8 e^0 = tau * p_n (tau also means the latency limit, A-26) | e^0 = p_n * T^0_n | Test T-4 (2026-11-13): offloading energy < local energy in 12/12 cases for p*T0, 10/12 for p*D_n, 0/12 for p*slot (1 s costs 0.3-1 J vs 7-42 mJ local), so the paper's Section 6.2 mechanism (offloading is cheaper than local computing) holds only for p*T0. Config switch removed. Artifact `results/T4_offload_energy_2026-11-13/` |
| A-37 | UAV energy model | Hover only: E_UAV = P_H * time per UAV, P_H = 168.4842 W (Diao et al. 2021 Eq. 17 and simulation settings, Lin's ref [31]); flight energy ignored as in Diao | Lin cites [31] for the UAV energy model; replaces the Zeng et al. stand-in (A-12/A-22), which gave the same value at hover (P0 + Pi = 168.4842 W); metrics only (2026-11-13) |
| A-13 | Who controls the UAV trajectory (not in the actor heads) | UAVs hover at positions sampled per episode (min separation d_s, A-16) | The paper claims joint trajectory optimization but its actor has no trajectory output (heads K+1, M, 2) and Table 3/Fig. 6 give fixed positions, so trajectory optimization is not reproducible from the paper's description; decided by the HUMAN 2026-11-13 |
| A-14 | Meaning of "partial offloading" | POS = may choose local or offload per task; FOS = must offload | Actor heads contain no offload ratio |
| A-32 | Local_Computing success rate (target for validation) | 61.25%: the text's relative percentages (85.63 is 39.8%, 40.6%, 20% higher than Local, FOS, MAPPO) give Local = 61.25%, FOS = 60.90%, MAPPO = 71.4%; Fig. 4b shows Local ~61%, MAPPO ~70%. Table 2's MAPPO 61.25% is the suspect entry | Replaces A-15 (W3 audit; checked by the main agent on the rendered Fig. 4b) |
| A-16 | d_s (UAV safe distance) | 10 m | Not stated. Diao et al. 2021 uses d_min = 10 m. |

| A-18 | Transmit time in the energy formula when the rate is ~0 | Capped at one slot for energy; the outcome latency stays uncapped (may be inf) | A transmission cannot outlast its slot (W1, 2026-11-13) |
| A-19 | Fail-reason labels | Under SIC "decode" == "rate" failure (via A-7), labelled "decode"; under noSIC "rate". Priority: decode/rate > deadline > energy | Metrics only (W1, 2026-11-13) |
| A-20 | Energy spent by a task that lacks battery | spent = min(required, battery) | Battery never negative (W1, 2026-11-13) |

New ambiguities found later get the next A-number here, with the date.

**Assumption policy (HUMAN, 2026-11-13; CLAUDE.md Rule 4):** every assumption is noted here and marked `# A-n` in code. If an alternative is tested and is closer to the paper (more literal reading, or smaller gap to a paper number with a non-learning baseline against a pre-written target), the old assumption is deleted from this table, the config and the code, and the new one is added with a new ID. Config switches that exist only to compare two assumptions (e.g. `bandwidth_mode`, `offload_energy`) are removed once the comparison is decided.

**Replaced assumptions** (one line each: date, old ID -> new ID, evidence, TE entry):
- 2026-11-13, A-6 -> A-33: with D_n = r*C_n (A-6) Local success is independent of task size and no c_cyc meets both paper numbers; in test T-1, at the c_cyc that matches 18.76 ms, A-6 gives Local success 79.95% (gap 18.7 pp) vs A-33 61.94% (gap 0.69 pp, within tolerance). Artifact `results/T1_deadline_model_2026-11-13/`; TE-007.
- 2026-11-13, A-12 + A-22 -> A-37: Lin's ref [31] (Diao et al. 2021) read; its Eq. 17 hover model replaces the Zeng stand-in; identical value at hover, so no rerun needed; TE-015.
- 2026-11-13, A-15 -> A-32: A-15 offered two readings of the Local success rate (71.4% or 45.8%), both refuted by Fig. 4b (Local ~61%) and by the paper's consistent use of relative percentages; TE-005.

**Proposed, not yet adopted:** A-24..A-31 in `docs/lin2025_targets.md` Section (d) (carrier bandwidth sharing, reward-per-episode, symbol clash, energy unit, paper-internal conflicts). They need HUMAN approval before entering this table.

## 7. Reproduction gate (Stage R2)

Written before any learning run. Pass if all hold:
1. Ranking of success rate in Table 2 is reproduced: POS_MADDPG_NOMA > {FOS_MADDPG_NOMA, POS_MAPPO_NOMA} > noSIC variants of the same algorithm.
2. SIC helps: each NOMA method beats its noSIC variant.
3. Fig. 5 trend: success decreases and latency increases with N for the learned methods; FOS is best at N = 2, 4. Checked exactly as defined in A-38 (per-seed OLS slopes over N in {2, 4, 6, 8, 10, 12}, thresholds -1.0 pp and +0.3 ms per sensor, every seed slope with the right sign, FOS best at N = 2, 4 with no tolerance); readings of items 1, 2, 4 are in A-40.
4. POS_MADDPG_NOMA success within +/- 10 percentage points of 85.63%.
Anything that fails is reported as a negative result, not tuned away.

## 8. Seeds

| Use | Range |
|---|---|
| Unit tests | 0-99 |
| Calibration (C-1, C-2) | 1000-1099 |
| Training reset seeds | 10_000 + training_seed * 1_000_000 + episode |
| Validation | 2000-2999 |
| Test (final numbers only) | 5000-5999 |

## 9. Code contract (API all workers code against)

Package `uavmec` in `src/uavmec/`. Config `configs/lin2025.yaml` (every key commented with its source tag).

```python
# src/uavmec/config.py
@dataclass(frozen=True)
class LinConfig: ...                       # all values of Sections 1, 5, 6
def load_config(path: str | Path = "configs/lin2025.yaml", **overrides) -> LinConfig

# src/uavmec/env/types.py
@dataclass(frozen=True)
class SensorAction:
    target: int        # 0 = local, 1..K = UAV index + 1
    subcarrier: int    # 0..M-1 (ignored if target == 0)
    power_frac: float  # [0, 1], p_n = power_frac * p_max
    bw_frac: float     # [0, 1], B_n = bw_frac * B_m

@dataclass
class TaskOutcome:
    size_bits: float; deadline_s: float; mode: str  # "local" | "offload"
    uav: int | None; subcarrier: int | None
    sinr: float | None; rate_bps: float | None; decoded: bool | None
    latency_s: float; energy_j: float
    success: bool; fail_reason: str | None   # "deadline" | "energy" | "rate" | "decode" | None

# src/uavmec/env/lin_env.py
class LinMECEnv:
    def __init__(self, cfg: LinConfig, sic: bool = True, mode: str = "POS"): ...  # mode "POS" | "FOS"
    n_agents: int; obs_dim: int; state_dim: int
    def reset(self, seed: int | None = None) -> tuple[list[np.ndarray], dict]: ...
        # obs: one normalized vector per sensor (own S_n + all S_k); info["state"]: global vector
    def step(self, actions: list[SensorAction]) -> tuple[list[np.ndarray], np.ndarray, bool, bool, dict]: ...
        # returns obs, rewards (shape N, Eq. 15), terminated (False), truncated (t == T), info
        # info["outcomes"]: list[TaskOutcome]; info["state"]; info["uav_energy_j"]
```

Read-only public attributes of `LinMECEnv`, valid after `reset` and updated by `step` (used by non-learning policies): `cfg`, `t` (slot index), `sensor_pos` (N,2) m, `uav_pos` (K,2) m, `task_sizes` (N,) bits, `task_deadlines` (N,) s, `battery` (N,) J, `uav_battery` (K,) J, `channel_gain` (N,K) linear |h_nk|^2.

Non-learning policy interface (`src/uavmec/baselines/`): `class Policy: name: str; def act(self, env: LinMECEnv, obs: list[np.ndarray], rng: np.random.Generator) -> list[SensorAction]`.

Modules with pure functions (unit-tested one equation at a time): `env/channel.py` (A-1), `env/tasks.py` (A-5, A-33), `env/noma.py` (Eq. 5-7, SIC and noSIC), `env/energy.py` (Eq. 2, 8, 10, 12), `env/local.py` (Eq. 1).

Metrics (`src/uavmec/eval/metrics.py`): success rate = successful tasks / all tasks; average latency over successful tasks; average sensor energy per slot in mW (energy per 1 s slot in mJ); per-mode counts; fail-reason histogram.

## 10. Learning code contract (stage R3, 2026-11-13)

Training hyperparameters live in `configs/train_lin2025.yaml` (NOT in the frozen env config `configs/lin2025.yaml`). Values the paper does not state are **A-23 (locked 2026-11-16 by the A-39 pilot, Section 6)**: 20,000 episodes for both MADDPG and MAPPO (our value, not the paper's; the provisional 10,000 came from the Fig. 4 axis), L = T = 10 steps per episode, one gradient update per env step after a warm-up of 1,000 transitions, Adam, soft-update tau 0.01, Gumbel-Softmax temperature 1.0 with straight-through hard samples, Gaussian exploration noise sigma 0.1 on the two continuous heads (linearly decayed to 0.01 over the first 50,000 env steps, i.e. the first 5,000 episodes; kept fixed when the episode count changed), MAPPO: clip 0.2, 10 epochs per update, GAE lambda 0.95, rollout of 32 episodes per update, same learning rates as MADDPG. Paper values (P): batch 320, actor lr 3e-4, critic lr 5e-5, buffer 2^17, gamma 0.9, actor hidden 128 + ReLU6, critic hidden 64 then 32 (W3 audit, Fig. 3).

Package `src/uavmec/agents/`:
- `common.py` (owner W7): `decode_actions(raw: np.ndarray, n_uavs: int, n_sub: int, fos: bool) -> list[SensorAction]` where `raw` has shape (N, (K+1) + M + 2) = [target one-hot (K+1) | subcarrier scores (M) | power_frac, bw_frac in [0,1]]; in FOS mode the local entry is never chosen. `ACTION_DIM(K, M) = K + 1 + M + 2`. `Actor(obs_dim, K, M, hidden=128)` torch module: 128-unit ReLU6 trunk, three heads (K+1 logits, M logits, 2 sigmoid), `forward(obs, explore, tau, fos) -> raw` using Gumbel-Softmax (hard, straight-through) on the two discrete heads.
- `maddpg.py` (owner W7): `MADDPG(n_agents, obs_dim, state_dim, K, M, cfg: dict, fos: bool, seed: int)`.
- `mappo.py` (owner W8): `MAPPO(n_agents, obs_dim, state_dim, K, M, cfg: dict, fos: bool, seed: int)`.
- Both implement the Learner interface used by `scripts/train.py` (owner W9):
  - `name: str`
  - `act(obs: list[np.ndarray], state: np.ndarray, explore: bool) -> list[SensorAction]` (caches the raw action for `observe`)
  - `observe(obs, state, rewards: np.ndarray, next_obs, next_state, done: bool) -> None`
  - `update() -> dict[str, float]` (called after every env step; returns {} when no update happened)
  - `state_dict() -> dict` / `load_state_dict(d) -> None` (networks, optimizers, RNG, replay or rollout buffer, step counters) so `--resume` continues exactly.
- noSIC variants are the same learners on `LinMECEnv(cfg, sic=False)`; FOS variants use `mode="FOS"` and `fos=True`.

## 11. Secondary analyses and result milestones (pre-registered 2026-11-18, before any run)

**Result milestones (HUMAN, 2026-11-18). Never delete or overwrite any of them:**
- **v1.0**: environment `lin-env-v1` (commit 27db5c1), training code 4dee668, the 75 runs, and the primary test evaluation `results/eval_test_2026-11-18/` (TE-065).
- **gate_fix**: the gate recomputed after the P-021 fix: `results/eval_test_2026-11-18_gate_fix/` (TE-066) and its clean-commit copy `results/eval_test_2026-11-18_gate_clean_96215d0/` (TE-069). This is the primary reproduction result: all four items FAIL.
- **v1.1**: a new environment version is created only if the assumption audit (S-2) finds a reading of the paper that is demonstrably more literal, and the HUMAN approves it. Then every run is retrained and evaluated into new folders; v1.0 and gate_fix stay as they are.

**S-1 Extended test evaluation (secondary; HUMAN priority P2).** Purpose: check whether the gate items 1 and 2 comparisons, which differ by less than 0.5 pp at 100 episodes, keep their sign with ten times more test episodes. Design, fixed now:
- Same 75 runs (`final.pt`), same script `scripts/evaluate_learned.py` at commit 96215d0, same settings as TE-065 (greedy, CPU only, Local_Computing paired on the same seeds) except: seeds 5000-5999 (`--seeds-start 5000 --n-episodes 1000`).
- Output: a new folder `results/eval_test_ext1000_<date>/` (and a Drive copy); never written into the v1.0 or gate_fix folders.
- Reported: the same gate.json items and summary.csv, labelled "secondary (S-1), 1000 test episodes". The per-seed standard deviation over the 5 training seeds (Table 2) is the uncertainty reported for every N = 8 mean.
- Use: S-1 never replaces or changes the primary result (gate_fix). If an S-1 item differs from the primary item, both are reported side by side and the difference is attributed to sampling of test episodes. No training, environment, or protocol change may follow from S-1.

**S-2 Assumption audit (HUMAN priority P0).** Read-only re-reading of Lin et al. (R-1) for the assumptions that set how hard offloading is: A-1 (channel gain), A-2 (noise power), A-3 (bandwidth per subcarrier), and the UAV computing model (UAV CPU sharing, queueing, computation time of offloaded tasks); also any other assumption found to affect offloading latency/success. For each: the exact paper text (page, equation, table), the current reading and its code location, and every alternative reading with a literal support in the paper. Output: `docs/lin2025_audit.md`. No code, config, or result change follows from S-2 without a separate HUMAN approval (P3).

**S-3 Sensitivity analysis (HUMAN priority P1).** Registered in detail after S-2 and before running: the list of alternative readings from S-2, the non-learning policies (Local_Computing, NearestUAVFullPower, and random offloading if available), validation seeds 2000-2099 only, metrics success/latency/energy at N = 8 and N in {2, 12}. Purpose: report how offloading difficulty depends on each assumption. It is never used to choose readings by closeness to the paper's learned-method numbers (Rule 4).

### S-3 detailed registration (2026-11-18, written after S-2 and before any S-3 code or run; HUMAN process approval 2026-11-18)

**Hypothesis (from S-2, `docs/lin2025_audit.md`, TE-072).** Offloading is much easier in `lin-env-v1` than in the paper (P-003) mainly because of readings that set link capacity. Each variant below changes ONE reading; S-3 measures how much each changes offloading difficulty for non-learning policies. S-3 does not decide which reading is "correct per the paper" (the paper does not give enough information for N-1, A-1, A-8), does not make any variant the default, and is never compared with Lin's learned-method numbers to choose a variant (Rule 4).

**Variants (all off by default; v1.0 behaviour unchanged).** B = `subcarrier_bw_hz` (250 MHz), group (k, m) = sensors offloading to UAV k on subcarrier m in a slot, f_n = `bw_frac` of sensor n, p_n = transmit power, h_n = |h_nk|^2 of sensor n to its own UAV.
- **V0 (reference)**: v1.0 exactly. B_n = f_n B; interference/SIC as A-8; noise 1e-10 W; distance-law gain (A-1); zero UAV compute time.
- **N1-seq (sequential depletion)**: within each group, sensors in increasing sensor index n; remaining R starts at B; B_n = f_n R; R <- R - B_n.
- **N1-prop (proportional rescaling)**: S = sum of f_j in the group; B_n = f_n B / max(1, S).
- **N1-eq (equal split, bound)**: B_n = B / |group| (ignores f_n).
  - **Equivalence check, registered now:** when every sensor in a group requests f = 1 (NearestUAVFullPower, GreedyLocalOrOffload when it offloads), N1-prop and N1-eq give identical B_n = B / |group|, and N1-seq gives B to the lowest index and 0 to the others. These identities are tested in the regression tests; RandomPolicy (f drawn in [0, 1]) is the policy that separates N1-prop from N1-eq, and S-3 reports them separately only where they differ.
- **A1-const (constant gain)**: h_n = 1e-5 for every link (Table 1 "Channel gain -50 dB" read as a bare constant).
- **A8-lit (literal Eq. 5)**: interference of j on n = h_n p_j (the gain of the decoded sensor n times j's power); chi_jn = 1 iff p_j > p_n (since h_n cancels, the order is by transmit power only; equal powers give chi_jn = 0, i.e. never cancelled); d_j = 1 iff R_j >= R_0 (A-7). Sensors are processed in decreasing p; I_n = sum_{j != n} (1 - chi_jn d_j) h_n p_j; gamma_n = h_n p_n / (I_n + sigma^2); R_n = B_n log2(1 + gamma_n). Under noSIC, A8-lit gives I_n = sum_{j != n} h_n p_j (nothing cancelled; this differs from V0 noSIC, which uses sum_{j != n} h_j p_j).
- **A3-total (total split)**: subcarrier bandwidth B = 250 MHz / 4 = 62.5 MHz (Sec 3 "B/M").
- **A2-dBm**: sigma^2 = 1e-13 W ("-100 dBm", convention only). **A2-psd**: sigma^2 = 1e-12 W (-174 dBm/Hz x 250 MHz, convention only).
- **UAV-cb (bound only, not a literal reading)**: offloaded task latency += C_n c_cyc / (f_uav / n_k), f_uav = 1.4 GHz (Table 1), c_cyc = 24.5 cycles/bit (our calibrated LOCAL value, A-33; the paper gives no UAV value), n_k = number of sensors offloading to UAV k in the slot (equal CPU sharing). Energy unchanged.
- The observation B_k stays the constant 1.0 in every variant (non-learning policies do not read it).

**Policies**: LocalComputing, NearestUAVFullPower, RandomPolicy, GreedyLocalOrOffload (existing code in `src/uavmec/baselines/policies.py`, unchanged); mode POS; SIC on. A8-lit is also run with SIC off for NearestUAVFullPower only. LocalComputing is run under every variant as a control: no variant touches local computing, so its per-seed results must be identical to V0 in all variants (a difference is a bug).

**Review notes (2026-11-18, before code):**
- NearestUAVFullPower and GreedyLocalOrOffload assign subcarriers round-robin per UAV (`_offload_all`), so a group (k, m) has more than one sensor only when more than M = 4 sensors offload to the same UAV. N1-* variants can therefore change these two policies only in such groups (mostly N = 8, 12); at N = 2 they are expected to have no effect for them. This is a property of the policies, reported as such, not a reason to change them.
- All offloading sensors of NearestUAVFullPower/GreedyLocalOrOffload use p = 1 W, so under A8-lit no sensor is ever cancelled for them (equal powers, chi = 0); RandomPolicy has distinct powers and exercises the SIC order.
- RandomPolicy uses the `rng` passed to `act`: S-3 seeds it as `np.random.default_rng(seed)` per validation episode and the env does not consume it, so RandomPolicy takes the same actions in every variant (paired comparison). A regression test checks that its action sequence is identical between V0 and one variant.
- N1-seq depends on the arbitrary sensor index order; it is reported as one rule among three, not as a preferred rule.
- Added 2026-11-18 after the code review, before any S-3 run: the A8-lit noSIC rows (NearestUAVFullPower) need a like-with-like reference, so V0 is also run with SIC off for NearestUAVFullPower at each N. The registered effect of A8-lit noSIC is its paired difference to V0 noSIC; the difference to V0 SIC on is also listed (column `reference`) for information only.

**Seeds and N**: validation seeds 2000-2099 (100 episodes, the same task/deadline streams for every variant and policy, A-36); N in {2, 8, 12}. Test seeds (5000-5999) are never used in S-3.

**Metrics** (per episode, per seed stored): success rate, mean latency over successful tasks (ms), mean energy over successful tasks (mW, A-34), fail reasons (A-19). **Effect of a variant** = paired difference to V0 over the 100 validation seeds, same policy and N: mean and 95% bootstrap CI (10 000 resamples over seeds, resampling RNG seed 0). A variant is reported as "changes offloading difficulty" for a policy/N if the 95% CI of the success difference excludes 0 and |mean difference| >= 1 pp; otherwise "no clear change". These are descriptive labels only.

**Implementation rules**: variants are selected through a separate config (`configs/s3_variants.yaml`, one named block per variant) passed explicitly; `configs/lin2025.yaml` and every default code path stay byte-for-byte v1.0. A regression test proves that V0 through the new code path gives results identical to v1.0 on seeds 2000-2004 for all four policies at N in {2, 8, 12}; tests for each variant formula (including the equivalence check above and A8-lit on a hand-computed 3-sensor group). Code marks every variant line with its ID. The run script writes to a new folder `results/s3_sensitivity_<date>/`: config of every variant, git commit + dirty flag, seeds, per-seed CSV, aggregate CSV with CIs, manifest.

**Bug rule (HUMAN, 2026-11-18)**: if an implementation error is found in a variant during or after the run, it is fixed with a regression test, logged (problemsAndFixes + trialAndError), and that variant is rerun on ALL 100 validation seeds for every policy and N it covers, into a new folder; never only the seeds with unfavourable results.

**Not allowed in S-3**: changing v1.0; choosing a default variant because it is closer to the paper; retraining; using validation results to claim the paper is reproduced; calling any variant "the paper's model" when the paper does not state the rule.

### S-4 Equivalence analysis of the learned methods (registered 2026-11-18, before computing it; HUMAN approval "làm 123")

- **Question**: are the learned methods at N = 8 equivalent in success rate within a practically negligible margin, rather than only "not separable"?
- **Data (no new evaluation)**: primary `results/eval_test_2026-11-18/per_run.csv` (100 test episodes per run); repeated on S-1 `results/eval_test_ext1000_2026-11-18/per_run.csv` as secondary. Unit of analysis: the training run (5 training seeds per configuration at N = 8).
- **Pairs**: the five N = 8 comparisons of gate items 1-2: POS_MADDPG_NOMA vs FOS_MADDPG_NOMA, POS_MADDPG_NOMA vs POS_MAPPO_NOMA, and each NOMA method vs its noSIC variant (POS_MADDPG, FOS_MADDPG, POS_MAPPO).
- **Test**: TOST with Welch's t (independent runs, unequal variances), alpha = 0.05 per side, i.e. the 90% CI of the difference in mean success (A - B, in pp). Margin delta = 1.0 pp: the same "practically meaningful" threshold fixed for S-3 before its run (TE-074). Note: this margin is chosen after the primary results were seen (F-2), so the analysis also reports, margin-free, the smallest delta for which equivalence would hold (max |90% CI bound|).
- **Output**: new folder `results/s4_equivalence_<date>/` (pairs CSV with means, stds, n, difference, 90% CI, t-values, TOST p, "equivalent within 1 pp" yes/no, smallest equivalence margin; manifest with git commit/dirty, input sha256). Descriptive: it does not change the gate result.

### S-5 Diagnostics of the learned policies' actions (registered 2026-11-18, before any run; HUMAN approval "làm 123")

- **Question**: do the trained MADDPG/MAPPO policies behave like the trivial NearestUAVFullPower heuristic (offload to the nearest UAV at full power and bandwidth), which would explain the saturation near 99%?
- **Runs**: the 15 Table 2 runs at N = 8 with SIC: POS_MADDPG_NOMA, FOS_MADDPG_NOMA, POS_MAPPO_NOMA x training seeds 0-4 (`final.pt` from Drive, copied to a new local folder under `results/`; no training).
- **Episodes**: validation seeds 2000-2099 only (never test seeds), greedy actions (explore = False), environment v1.0 (default, no S-3 variant), the same task streams for all runs and for NearestUAVFullPower (A-36).
- **Metrics per run** (over all sensor-slots): fraction offloaded; fraction offloaded to the nearest UAV; agreement with NearestUAVFullPower's target choice (local/UAV id); mean and distribution (deciles) of power_frac and bw_frac for offloaded tasks; fraction of offloaded tasks sharing a (UAV, subcarrier) group with another sensor; success/latency of the run vs NearestUAVFullPower on the same seeds (paired difference over seeds, 95% bootstrap CI, 10 000 resamples, RNG seed 0).
- **Reading rule (fixed now)**: the policies are called "close to the heuristic" for a run if target agreement >= 90% AND mean power_frac >= 0.9 AND mean bw_frac >= 0.9 for offloaded tasks; otherwise the deviations are described. Descriptive only.
- **Output**: new folder `results/s5_policy_diagnostics_<date>/` (per-run CSV, summary, manifest with git commit/dirty and sha256 of each final.pt used). The analysis script comes with a unit test of its metric functions.

### Future sensitivity analyses (listed, NOT registered, NOT approved)

- **FS-1 (A-43)**: cycle end as termination (no bootstrap) instead of truncation. Would need a code switch, a regression test showing v1.0 unchanged by default, and retraining, so it needs its own registration and HUMAN approval.

## 12. Environment lin-env-v1.1 (registered 2026-11-18, before any v1.1 code or run)

**Decision (HUMAN, 2026-11-18)**: P3 reopened with a new condition: accept the per-carrier constraint "sum of bandwidth fractions on each carrier <= 1" and record its enforcement as our assumption A-42 (not a rule confirmed by Lin). Steps 1 (this spec) and 2 (implementation and tests) approved; step 3 (pilot training) and step 4 (75 training runs) NOT approved yet.

**Definition**: lin-env-v1.1 = lin-env-v1 (v1.0) + A-42, nothing else. Every other assumption, value, calibration (c_cyc, kappa), observation, action, reward and seed range is unchanged. Purpose: an environment closer to the paper's text, to serve as the base of the extension. It is NOT expected to reproduce Lin's numbers: S-3 measured NearestUAVFullPower at 0.973 / 0.935 success (N = 8 / 12) under N1-prop, so learned methods will probably still be near saturation; this is reported, not tuned.

**Implementation contract**:
- `LinConfig` gets a field `bandwidth_sharing` with values `"independent"` (v1.0, default) and `"proportional"` (A-42). New config file `configs/lin2025_v1_1.yaml` = `configs/lin2025.yaml` + `bandwidth_sharing: proportional`. `configs/lin2025.yaml` is not edited; v1.0 stays the default everywhere (train.py, evaluate_learned.py, run_s3.py), and v1.1 is selected only with `--env-config configs/lin2025_v1_1.yaml`.
- Note: the new field appears in `dataclasses.asdict(cfg)`, so the config hash recorded by train.py for NEW v1.0 runs will differ from the hash of the existing 75 runs although the environment is identical; existing artifacts are not touched.
- Code lines marked `# A-42`. The S-3 code path (`s3_variant`) stays as it is.

**Acceptance checks for step 2 (all must pass before step 3 is proposed)**:
1. v1.0 unchanged: with `configs/lin2025.yaml` the frozen v1.0 per-episode snapshot (`tests/data/v1_reference_s3.json`) is reproduced exactly (4 policies, N in {2, 8, 12}, seeds 2000-2004).
2. v1.1 equals the S-3 N1-prop variant per episode (same seeds/policies/N): two independent implementations of A-42 must agree exactly.
3. LocalComputing per-episode results identical under v1.0 and v1.1.
4. Unit tests of the A-42 allocation: single sensor (B_n = f_n B), group with sum f <= 1 (unchanged), group with sum f > 1 (rescaled, sums to B), independence from sensor order.
5. train.py and evaluate_learned.py run end-to-end with `--env-config configs/lin2025_v1_1.yaml` (tiny smoke into the session scratchpad, a few episodes, not a result).
6. Full test suite and ruff clean.
7. Report (descriptive, validation seeds 2000-2099 only, never test seeds): the four non-learning policies on v1.1 at N in {2, 8, 12} vs v1.0 (output `results/v1_1_heuristics_<date>/`, with manifest).

**Step 3 pilot (registered now for later approval, not to be run before approval)**: POS_MADDPG_NOMA and POS_MAPPO_NOMA, N = 8, SIC, training seed 0, A-23 settings (20 000 episodes), v1.1; T1/T2 as A-39 (reported); validation success of the final policy vs NearestUAVFullPower on v1.1 (seeds 2000-2099, paired 95% bootstrap CI). The pilot result is input for the HUMAN's step 4 decision; it never changes A-42 or any value.

**Status (2026-11-18, step 2 done; TE-087)**: A-42 implemented (`LinConfig.bandwidth_sharing`, `proportional_bandwidth` in `src/uavmec/env/lin_env.py`, `configs/lin2025_v1_1.yaml`; uncommitted). Acceptance checks 1-7 pass; full suite 231 passed (rerun by the main agent), ruff clean. Extra guard: an `s3_variant` cannot be combined with `bandwidth_sharing: proportional` (ValueError), so the S-3 path can never silently ignore A-42. Step 3 (pilot) still needs HUMAN approval.

## 13. Test T-5 and environment lin-env-v1.2 (registered 2026-11-20, before any T-5 code or run)

**Decision (HUMAN, 2026-11-20, "Nhanh A - Reproduction")**: pre-register a test of one extra parameter, the extra link loss L (our assumption A-44), against Table 4 of Lin et al. (the only non-learning offloading evidence, test T-2). The choices below were fixed before any T-5 code was written.

**Parameter (A-44)**: extra link loss L in dB, applied uniformly to every sensor-UAV link: |h_nk|^2 = beta0 * 10^(-L/10) / (H^2 + d^2). beta0 stays -50 dB (Table 1 literal), the A-1 distance law is unchanged, noise stays A-2. Reading: an unstated extra loss in the link budget (for example antenna or implementation loss). It is NOT stated by Lin. It is equivalent to raising the noise by L dB. L = 0 is lin-env-v1.0.

**Test T-5, replay**: the Table 4 slot (Table 3 positions, sizes, deadlines, exact actions: UAV, carrier, bandwidth %, power %) is replayed in `LinMECEnv` (the real env code: one step, N = 8, K = 2, SIC, POS, current c_cyc and kappa), exactly as in T-2 (Section 5), under both readings: R1 = Table 4 literal (every offloader -> UAV1 at (30, 53)); R2 = text and Fig. 6 (UAV1 serves S0, S4; UAV0 at (77, 87) serves S3, S5, S6, S7). The T-2 replay script was not kept in the repository, so it is re-implemented in `scripts/run_t5.py`; as a check, at L = 0 it must reproduce T-2's 14/16 and the per-sensor outcomes (sim_success, match) of candidate `a_dist_percarrier` in `results/T2_offload_physics_2026-11-13/table4_replay.csv`. If it does not, T-5 stops and is reported.
- Grid: L from -10 dB to +20 dB in steps of 0.25 dB (121 values). Score per reading = number of the 8 sensors (including the 2 local ones) whose success/fail matches Table 4.
- Feasible set F = {L : score 8/8 under R1 AND 8/8 under R2}.
- Decision rule (mechanical): if 0 is in F -> no change, A-44 not adopted. If F is non-empty and excludes 0 -> adopt L* = midpoint of the largest contiguous interval of F (contiguous on the 0.25 dB grid), rounded to the nearest 0.5 dB (a tie at x.25 or x.75 goes up; clarification added 2026-11-20 before the run; if two contiguous intervals have the same length, the one with the lower L is used). If F is empty -> no change, reported.
- Descriptive only (never decides): the same replay with a path-loss exponent alpha in {2.0, 2.2, ..., 3.0} instead of L (distance law (H^2 + d^2)^(-alpha/2) times beta0, L = 0).

**Disclosure (Rule 4)**: before this registration the main agent ran an exploratory standalone probe (TE-100; not the env code, not registered, no stored artifact) that found all 6 offloading outcomes matching only when every link is roughly 5 to 8.5 dB weaker. The midpoint rule was chosen after seeing that range, so T-5 is a confirmation of a range already seen, not a blind test. The evidence is the 6 offloading sensors of one slot (weak). Table 4 has known inconsistencies (UAV assignment, A-31). A-44 is a reading of the gap, not a measured paper parameter.

**Definition**: lin-env-v1.2 = lin-env-v1 (v1.0) + A-44 (A-42 OFF, bandwidth_sharing independent), nothing else. Purpose: a reproduction branch. v1.0 and all its results stay unchanged as the reproduction milestone. Learned methods are NOT retrained on v1.2 (that needs separate HUMAN approval and runs on Colab, started by the HUMAN).

**Implementation contract (only if A-44 is adopted)**: `LinConfig` gets `extra_link_loss_db: float = 0.0` (default keeps v1.0 bit-identical), applied in the channel gain, code marked `# A-44`. New config `configs/lin2025_v1_2.yaml` = `configs/lin2025.yaml` + `extra_link_loss_db: <L*>`. `configs/lin2025.yaml` and `configs/lin2025_v1_1.yaml` are not edited. Guard: `extra_link_loss_db != 0` combined with an `s3_variant` raises ValueError (like the A-42 guard).

**Acceptance checks**: (1) the frozen v1.0 snapshot `tests/data/v1_reference_s3.json` is reproduced exactly; (2) v1.1 unchanged (its existing tests pass); (3) LocalComputing per-episode results identical under v1.0 and v1.2; (4) unit tests of A-44 (L = 0 identity; gain scales by 10^(-L/10)); (5) `train.py` and `evaluate_learned.py` smoke with `--env-config configs/lin2025_v1_2.yaml` into a scratch folder only (a few episodes, not a result); (6) full test suite and ruff clean.

**Baseline report (descriptive, validation seeds 2000-2099 only, never test seeds)**: LocalComputing, RandomPolicy, NearestUAVFullPower, GreedyLocalOrOffload, POS, SIC, N in {2, 4, 6, 8, 10, 12}, on v1.2 and v1.0, with the paired v1.2 - v1.0 difference and a 95% bootstrap CI (10 000 resamples, RNG seed 0); plus NearestUAVFullPower with noSIC at N = 8. Output `results/v1_2_baselines_2026-11-20/` with a manifest (config, commit, git_dirty, seeds, timestamp, llm_client "none"), produced from a clean commit.

**Status (2026-11-20)**: registered, T-5 not yet run.
