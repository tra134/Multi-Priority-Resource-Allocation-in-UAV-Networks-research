# Lin et al. 2025: assumption audit for offloading difficulty (S-2)

Status: DRAFT for `docs/lin2025_audit.md`, written 2026-11-18 by a read-only worker (HUMAN priority P0, spec Section 11, S-2). Nothing in the paper, code, configs, results or `slide/` was changed. No simulation was run and no new simulation number is given.

## 0. Scope, method, limits

- Question: which readings of Lin et al. (R-1) set how hard offloading is, and is the current reading the most literal one? Context (P-003, F-2): NearestUAVFullPower reaches about 99% success / 1.4 ms on validation seeds (TE-007: 98.86% / 1.43 ms), and every learned method of ours saturates near 99% (`docs/findings.md` F-2), against 85.63% / 7.48 ms for Lin's best learned method.
- Sources read in full for this audit: R-1 (text extracted with `pdftotext`, and pages 4, 10, 11, 13 viewed as rendered images to check equations and tables), the code in `src/uavmec/env/`, `src/uavmec/config.py`, `src/uavmec/baselines/policies.py`, `configs/lin2025.yaml`, spec Sections 1, 2, 5, 6, 11, `docs/lin2025_targets.md`, P-003, TE-004..TE-010 and TE-014, `docs/findings.md`, `reference.md`. R-2 (Diao et al. 2021) was grepped where Lin cites it ([31]) or where it shares a parameter (g0 = -50 dB, H = 100 m).
- Page numbers are the journal's "n of 14". Equations are my plain-text transcription of the typeset equations (checked on the rendered page); text quotes are verbatim from the PDF text layer.
- Rule 4 discipline: judgements in item (4) of each section use only the paper text and R-2. Lin's learned-method numbers (Table 2, Figs. 4 and 5) were deliberately NOT used to prefer a reading. The direction notes in item (5) are hypotheses to be measured in S-3, not results. Numbers quoted from earlier work carry their source.
- "Literal support" below means a passage of R-1 that can be read as stating the alternative. Readings that only follow wireless-literature convention are listed separately as "convention only" and are not counted as literal.

## 1. Summary table

| # | Item | Current reading (spec / code) | Verdict on current reading | Direction of the best alternative (HYPOTHESIS for S-3) |
|---|---|---|---|---|
| 1 | A-1 channel gain | beta0 = -50 dB at 1 m, gain = beta0 / (H^2 + d^2) | tie (distance dependence is supported by the paper's structure; "-50 dB" is also literally a constant) | constant 1e-5 for all links: noise-limited SNR much higher, but no near-far spread among co-channel sensors: direction unclear |
| 2 | A-2 noise | sigma^2 = 1e-10 W per link ("-100 dB" read as dB re 1 W) | most literal | "-100 dBm" = 1e-13 W (convention only): easier |
| 3 | A-3 bandwidth | 250 MHz per subcarrier | most literal (not decisive) | total split B/M = 62.5 MHz: harder (already measured in T-2, TE-008) |
| 4 | N-1 (new, unlogged) bandwidth is a shared per-carrier resource | each sensor gets bw_frac x 250 MHz independently; no depletion; observed B_k is the constant 1.0 | LESS literal than the alternative | harder, possibly strongly (full-bandwidth heuristic can no longer give every co-channel sensor the whole carrier) |
| 5 | A-8 Eq. 5 / chi_jn | interferer strength g_j = \|h_jk\|^2 p_j, SIC order by g | tie: equations say \|h_nk\|^2 p_j, prose says channel gain of each sensor | literal Eq. 5: SIC order by power only, no cancellation among equal powers: probably harder for full-power heuristics |
| 6 | UAV computing | zero compute time, no sharing, no queue, no result return, UAV CPU unused | most literal (stated 3 times) | model UAV compute with 1.4 GHz and sharing: harder |
| 7 | Eq. 7 log base | log2 | most literal (Shannon, bit/s) | ln: rate x 0.69, slightly harder |
| 8 | T^0 formula | T^0 = C_n / R_n | only reading (formula not in the paper) | none with literal support |
| 9 | Time-varying channel h(t) | static within an episode (A-13 hover) | tie / less literal on the "(t)" index | fading or UAV motion: probably harder; no model in the paper, cannot be tested without inventing one |
| 10 | Units | all conversions checked, see Section 5 | consistent | n/a |

## 2. Detailed audit

### 2.1 A-1 channel gain

(1) Paper text.
- Table 1 (p. 10), last row, right half: "Channel gain | -50 dB". No other statement of a channel model or path loss exists in R-1 (`docs/lin2025_targets.md` Section (a): "No path-loss model or carrier frequency").
- Eq. (3) (p. 4): y_k^m(t) = sum_n U_n(t) V_n^{km}(t) h_n^k(t) sqrt(p_n(t)) x_n(t) + z_k. The gain h_n^k carries the index (t), a sensor index n and a UAV index k.
- Section 3.3 (p. 4): "The UAVs fly over the disaster area at a fixed altitude H." Table 1 (p. 10): "L, W, H | 100 m".
- Section 4.1 (p. 5): the sensor state is S_n(t) = {x_n, y_n, E_n, C_n, D_n} and the UAV state is S_k(t) = {x_k, y_k, H, B_k(t), E_k(t)}. Positions are part of the state.
- Abstract / Section 1 (p. 1-2): the algorithm "jointly optimizes the UAVs' flight trajectories". Section 3.3 (p. 5): "four parameters need to be adjusted for each time slot, that is, the flight trajectory, ...".
- Section 1 (p. 2): "UAVs are computationally loaded into dynamic wireless communication networks where the network environment and channel conditions change constantly."
- Fig. 6 (p. 13): 3D plot with the UAVs drawn at z = 100 (checked on the rendered page).
- R-2 (Lin's ref [31]), Section II Eq. (1): g_{k,m} = g0 / ||q_m - u_k||^2 times a Rician term, "g0 represents the channel power gain at the reference distance d0 = 1 m"; Section IV: "H = 100 m, HB = 30 m, dmin = 10 m, kappa = 10 dB, g0 = -50 dB". Lin cites [31] only for the UAV energy ("flight energy consumption and hovering energy consumption [31]", p. 4), not for the channel. Lin writes (p. 10) "We mainly refer to references [7, 18, 19], for parameters related to equipment and links, such as drone CPU frequency, drone battery capacity, sensor CPU frequency, and sensor power". [7], [18], [19] are not in `reference.md` and are not in the repo.

(2) Current reading. Spec A-1: |h_nk|^2 = beta0 / (H^2 + ||q_k - w_n||^2), beta0 = 1e-5 (-50 dB) at 1 m. Code: `src/uavmec/env/channel.py:18` (`return beta0 / (height_m**2 + d2)`), `configs/lin2025.yaml:22` (`beta0: 1.0e-5`), `src/uavmec/env/lin_env.py:67-69` (recomputed each slot from fixed positions, so it is constant within an episode). Sensors are uniform in the 100 m x 100 m area (`lin_env.py:111`), UAVs uniform in the area with separation >= 10 m (`lin_env.py:53-63`, A-13, A-16), at altitude 100 m.

(3) Alternatives with literal support.
- (a) Constant gain for every link: Table 1 lists "Channel gain -50 dB" as a bare value next to "Noise -100 dB", with no reference distance and no distance law. Spec A-1 already names this alternative (constant 1e-5). Against it: Eq. (3) indexes h by n, k and t, the state contains positions, and the paper claims trajectory optimization; none of that would matter for a constant gain.
- (b) The same distance law with a different reference distance (for example -50 dB at the nominal altitude H): no passage in R-1 states a reference distance. R-2 states 1 m. No literal support in R-1; not counted.
- (c) Small-scale fading on top of the distance law: support is only the "(t)" index of h and the Section 1 sentence "channel conditions change constantly". No fading statistics are given anywhere in R-1, so this reading cannot be implemented from the text. R-2 uses Rician fading with kappa = 10 dB, but only through its mean (Jensen bound, R-2 Eq. 2-3); that is R-2's model, not Lin's.

(4) Judgement: tie. The distance law with beta0 = -50 dB at 1 m is the form used by the paper Lin cites for UAV parameters (R-2: same g0 = -50 dB, same H = 100 m), and it is the only reading under which the state, trajectory and h(t) elements of R-1 have any effect. The constant reading is the more literal reading of the single Table 1 row. Neither is excluded by the text. The 1 m reference distance itself comes from R-2, not from R-1.

(5) Hypothesis for S-3. Arithmetic on the formula (not a simulation): with H = 100 m and horizontal distance d <= 141 m, the current gain lies between about 1e-9 and 3.3e-10, which is about 40 dB below the constant 1e-5. A constant gain would raise the noise-limited SNR by a factor of about 10^4 to 3x10^4 and remove all near-far differences between sensors. It should make single-user links easier; for sensors sharing a carrier it removes the received-power spread that SIC exploits, so the net direction for NOMA groups is unclear. In the current reading the gain varies by at most a factor of about 3 across the area, so the sensor and UAV positions matter little.

### 2.2 A-2 noise power

(1) Paper text.
- Table 1 (p. 10): "Noise | -100 dB".
- Section 3.1 after Eq. (3) (p. 4): "z_k is the Gaussian white noise at UAV k with zero mean and variance sigma^2".
- Eq. (6) (p. 4): gamma_n^{km} = g_n / (I_n + sigma^2). Eq. (7) (p. 4): R_n^{km} = B_m log(1 + gamma_n^{km}) >= R_0; "Here B_m is the bandwidth."
- No noise bandwidth, no noise density and no reference level for "dB" appear anywhere. Table 1 gives power as "Sensor power | 1 W".
- R-2 Section IV: "N0 = -174 dBm/Hz", "Pkmax = 20 dBm", SNR written as p g / (N0 B).

(2) Current reading. Spec A-2: sigma^2 = -100 dB = 1e-10 W per link, independent of bandwidth. Code: `configs/lin2025.yaml:21` (`noise_w: 1.0e-10`), used unchanged in SINR at `src/uavmec/env/noma.py:38` and `:46` (`g / (interference + noise_w)`), whatever the sensor's bandwidth B_n.

(3) Alternatives.
- Literal support: the text gives sigma^2 as a variance, that is a power, in Eq. (6), added to the interference with no bandwidth factor. So a power per link is the literal reading. A noise density times bandwidth has no literal support.
- Convention only: "-100 dB" written for "-100 dBm" = 1e-13 W (R-2 writes dBm explicitly, so a dBm/dB slip is plausible but not shown). "-174 dBm/Hz x B" (R-2's model): for B = 250 MHz this is about -90 dBm = 1e-12 W (unit arithmetic).

(4) Judgement: current is the most literal reading. "dB" has no stated reference, but the only power unit used in the model is the watt (p_max = 1 W), so 0 dB = 1 W (dBW) is the natural reading, and the variance form sigma^2 in Eq. (6) matches a plain power. The dBm reading needs a letter that is not in the paper.

(5) Hypothesis for S-3. A smaller noise (1e-13 W, or 1e-12 W) means a higher SNR, so easier; it matters most for weak links and for single-user carriers, little for interference-limited NOMA groups. Noise scaled with B_n (a density reading) would make narrow bandwidth fractions cheaper in noise, which has no support in the text.

### 2.3 A-3 bandwidth per subcarrier

(1) Paper text.
- Section 3 intro (p. 3): "each UAV allocates the frequency resource with bandwidth B into M subcarriers on average, and the transmission bandwidth on each subcarrier is B/M."
- Eq. (7) (p. 4): "B_m log(1 + gamma)"; "Here B_m is the bandwidth." (note: B_m, not a per-sensor bandwidth).
- Table 1 (p. 10): "Number of carriers | 4", "Carrier bandwidth | 250 MHZ".
- Section 6.1 (p. 9): "a system with 2 UAVs and 8 sensors, each equipped with 4 subcarriers of 250 MHz bandwidth".
- Fig. 4 caption (p. 11): "where B_max = 250, K = 2, and N = 8." (no unit).
- Section 6.4 (p. 11): "Under the same resources and the fixed total bandwidth, as N increases, the success rate ... decrease gradually".
- Section 6.4 / Fig. 5b text (p. 12): "When N = 2 and 4, there is sufficient bandwidth for sensor offloading, and transmission latency is low." and "FOS_MADDPG_NOMA algorithm offloads all tasks to UAVs, and link load increases".

(2) Current reading. Spec A-3 (decided 2026-11-13, TE-010): B_m = 250 MHz per subcarrier. Code: `configs/lin2025.yaml:8` (`bandwidth_hz: 250.0e+6`), `src/uavmec/config.py:50-53` (`subcarrier_bw_hz` returns `bandwidth_hz`), rate at `src/uavmec/env/noma.py:39,47` with bandwidth `bw_frac x subcarrier_bw_hz` (`src/uavmec/env/lin_env.py:150`).

(3) Alternatives with literal support.
- Total split: "B/M" with B = 250 MHz, giving 62.5 MHz per subcarrier. Support: the Fig. 4 caption "B_max = 250" and "fixed total bandwidth" (Section 6.4). Weakness: B_max is never defined in the paper, carries no unit, and "fixed total bandwidth" is also true of 4 x 250 MHz.
- B = 1 GHz per UAV, B/M = 250 MHz: this is the current reading seen through the Section 3 definition; it is the same numbers as the current code.
- Another reading: the text of Section 3 says B belongs to each UAV; a split of 250 MHz over both UAVs (31.25 MHz per subcarrier) has no support.

(4) Judgement: current is the most literal, though not decisive. Two statements attach "250 MHz" directly to a carrier (Table 1 "Carrier bandwidth 250 MHZ" with "Number of carriers 4"; Section 6.1 "4 subcarriers of 250 MHz bandwidth"). The B/M sentence is a definition without a number, and the only attachment of 250 to the symbol B is the unit-less "B_max = 250". This agrees with TE-009 (which corrected the earlier miscount, P-005). The earlier W3 audit counted three passages for the total reading; TE-009 showed that two of them do not discriminate.

(5) Hypothesis for S-3. Total split divides every rate by 4 at the same SINR, so harder. Existing measurement (pre-freeze T-2 descriptive run, TE-008, `results/T2_offload_physics_2026-11-13/descriptive_val2000.csv`, validation seeds 2000-2099): NearestUAVFullPower 98.86% (per carrier) vs 90.69% (total split) with the distance gain, and RandomPolicy 75.40% vs 55.65%. S-3 can repeat this on the frozen v1.0 environment.

### 2.4 N-1 (new, not in spec Section 6): is the carrier bandwidth a shared, depleting resource?

This item is the largest difference found between the paper's text and the code that is not yet an assumption with an ID. `docs/lin2025_targets.md` proposed it as A-24 and A-29; they were "not adopted (HUMAN did not approve test T-5)" (TE-013). It is therefore an implicit assumption of the code that is not in spec Section 6.

(1) Paper text.
- Section 4.1 (p. 5): the UAV state includes "B_k(t) = {B_{k,1}(t), B_{k,2}(t), ..., B_{k,m}(t)} denotes the available bandwidth of the UAV's m carriers".
- Section 4.2 (p. 5-6): the action includes "(3) how much bandwidth to allocate for data transmission" and B_nm(t).
- Section 5.3 (p. 9): "The bandwidth resource allocation and transmission power allocation actions take values in the range [0,1] to indicate the proportion of available bandwidth or transmission power that can be allocated to the task."
- Algorithm 1 / Section 5.2 (p. 8): "the UAV's system state includes location, available bandwidth, available power" (power is a battery that decreases).
- Table 3 title (p. 12): "UAVs' initial position, available bandwidth, and available power", with "Carrier 0 ... Carrier 3 | 100%" for both UAVs.
- Table 4 (p. 12): sensors on the same carrier have allocations that sum to less than 100%: carrier 2 holds MU3 47.8% + MU5 45.3% = 93.1%; carrier 1 holds MU6 30.8% + MU7 42.2% = 73.0%. (These are outputs of a learned policy, so they are evidence of how the authors' environment was used, not a calibration target.)
- Section 6.4 (p. 11-12): "When N = 2 and 4, there is sufficient bandwidth for sensor offloading"; "link load increases, which affects average latency".
- Section 3 (p. 3): "multiple sensors can share a single subcarrier for task offloading in short packets".
- Eq. (7) itself writes B_m, not B_nm.

(2) Current reading. Not in the spec as an A-ID; spec Section 2 line "Rate (Eq. 7): B_n = bw_frac_n x B_m (A-3)". Code: `src/uavmec/env/lin_env.py:150` (`bw = ... actions[n].bw_frac * cfg.subcarrier_bw_hz`): every sensor gets its own fraction of the full 250 MHz, regardless of how many other sensors use the same (UAV, subcarrier); there is no budget. `lin_env.py:88` sets the observed B_k to the constant 1.0 (`cfg.subcarrier_bw_hz / cfg.bandwidth_hz`). `src/uavmec/baselines/policies.py:91` makes every sensor ask for 100% power and 100% bandwidth (`SensorAction(k + 1, c % m, 1.0, 1.0)`), with round-robin subcarriers per UAV, so with 4 sensors per UAV and N = 8 each sensor holds a whole carrier, and with more than 4 sensors on one UAV several sensors each hold the full carrier bandwidth at once.

(3) Alternatives with literal support.
- (a) Available bandwidth of a carrier depletes as sensors allocate: B_n = bw_frac_n x (remaining bandwidth of (k, m)), allocated in some order (for example sensor index). Support: B_k(t) is a state variable of the UAV with the index (t); the action is a "proportion of available bandwidth"; Table 3 says "initial ... available bandwidth".
- (b) The fractions on one carrier are normalised so that they sum to at most 1 (for example proportional rescaling). Support: Table 4's sums; "proportion of available bandwidth". The rescaling rule itself is not stated.
- (c) Eq. (7) literal: every sensor on carrier m has B_m (the whole subcarrier bandwidth) and the bandwidth action has no effect on the rate. Support: Eq. (7) writes B_m. Against it: the action space and the Table 4 percentages.
- None of (a), (b) states the allocation order or the rescaling rule: that part is an unavoidable new assumption.

(4) Judgement: the current reading is LESS literal than (a) or (b). The current code gives every co-channel sensor an independent full carrier, which contradicts the plain meaning of "available bandwidth" that "decreases", of "proportion of available bandwidth", and of the Table 3 word "initial". It is the only item in this audit where several independent passages (Section 4.1, 5.3, Table 3, Table 4, Section 6.4) point away from the code. The exact depletion rule is not in the text, so any replacement is a new assumption with an ID, chosen only after S-3.

(5) Hypothesis for S-3. Harder, and possibly the largest lever: a heuristic that requests 100% bandwidth for every sensor would get the whole carrier only for the first sensor (depletion) or 1/n of it (rescaling) on a shared carrier, and a smart policy has to avoid sharing or to split bandwidth. Link load would grow with N as the text describes in Section 6.4, which the current code can only reproduce through interference. Note (c) would remove the effect of the bandwidth action altogether (equal to bw_frac = 1), which does not change the full-bandwidth heuristic.

### 2.5 A-8 and the SIC order (Eq. 5, chi_jn)

(1) Paper text.
- Section 3.2 (p. 4): "The UAVs sort the offloaded sensors by their signal strength, which depends on their transmission power and channel gain. The sensor with the strongest signal is decoded first. ... For sensor n that offloads to subcarrier m at UAV k, its signal strength is g_n^o = U_n(t) |h_n^k(t)|^2 V_n^{km} p_n(t) and its interference is"
- Eq. (5) (p. 4): I_n(t) = sum_{j in N, j != n} U_j(t) V_j^{km} [1 - chi_{jn}(t) d_j(t)] |h_n^k(t)|^2 p_j(t). The channel gain inside the sum carries the index n, not j.
- Following sentence (p. 4): "where chi_jn is a binary indicator that equals 1 if the signal strength of sensor j is greater than that of sensor n, that is, |h_n^k(t)|^2 p_j(t) > g_n^o, and 0 otherwise. Moreover, d_j(t) is also a binary indicator that equals 1 if sensor j is successfully decoded and 0 otherwise."

(2) Current reading. Spec A-8: use |h_jk|^2 for each interferer, and order by g_j = |h_jk|^2 p_j; A-7: d_j = 1 iff R_j >= R_0. Code: `src/uavmec/env/noma.py:41-48` (order by received power `g = |h_jk|^2 p_j`, interference = all weaker sensors plus stronger ones that failed to decode), `src/uavmec/env/lin_env.py:151` (`rx = channel_gain[off, uav_idx] * power`). Ties are broken by index (`kind="stable"`).

(3) Alternatives.
- Literal equations: interference of sensor j on n = |h_n^k|^2 p_j (n's own channel gain times j's power), and chi_jn = 1 iff |h_n^k|^2 p_j > |h_n^k|^2 p_n, that is iff p_j > p_n: the decoding order is by transmit power only and equal powers give chi = 0 (no cancellation at all). Support: Eq. (5) and the sentence defining chi_jn, which use the same h_n^k twice.
- Prose reading (current): "signal strength ... depends on their transmission power and channel gain" and g_n^o uses the sensor's own gain, so the strength of j should use h_j^k.

(4) Judgement: tie. The two equations literally say |h_n^k|^2 p_j; the prose and the definition g_n^o say the strength of a sensor uses its own gain. A-8 resolves an internal contradiction of the paper in favour of the prose and the physics. `docs/lin2025_targets.md` already noted that the chi definition also uses h_nk. Neither reading can be called more literal on the text alone.

(5) Hypothesis for S-3. With the literal equations the SINR of sensor n does not depend on its own channel gain when noise is small (gamma_n ~ p_n / sum of unremoved p_j), and a group of full-power sensors gets no SIC gain, because equal powers never satisfy p_j > p_n. This is probably harder for NearestUAVFullPower on shared carriers and makes the NOMA-vs-noSIC gap depend only on power control. Direction for learned methods is unclear. This reading could be checked with a non-learning rule on the Table 4 replay (T-2), where the powers differ, but Table 4 has only 8 sensors (weak evidence).

### 2.6 UAV computing model

(1) Paper text.
- Section 3.1 (p. 4), before Eq. (3): "We assume that the UAVs have sufficient computation resources and negligible computation time."
- Section 3.1 (p. 4), before Eq. (8): "It is assumed that the UAVs have enough computation resources and negligible computation time for the offloaded tasks."
- Section 6.2 (p. 10-11): "Because the server has enough computing resources to quickly process the successfully unloaded task, and the server computing task time can be ignored, so only the transmission energy consumption and the local computing energy consumption are considered".
- Table 1 (p. 10): "UAV CPU frequency | 1.4 GHz". Section 6 (p. 9-10): refers to [7, 18, 19] "for parameters related to equipment and links, such as drone CPU frequency".
- Latency terms: Eq. (14d) and Eq. (15) use only T_n^l and T_n^0, the transfer time ("T_n^0(t) represents the transfer time of offloading", p. 4); Section 4.3 (p. 6) writes "T_n^l(t) + T_n^0(t) <= tau".
- No queue, no sharing of the UAV CPU, no result downlink, no computing energy of the UAV appear. The only mention of a buffer is Section 4 (p. 5-6): sensors need not know "UAV flight trajectory, battery power, buffer length, transmission power" of others, with no queue model.
- Section 6.4 (p. 12): "all local tasks are offloaded to 2 UAVs, and UAVs' resources are enough to handle tasks" for N = 2 and 4, which implies that UAV resources may not be enough for larger N; given the three explicit "enough computation resources" statements this is read as communication resources (bandwidth, interference).
- R-2 (Lin's ref [31]), Section II: "Since the BS has strong computing capabilities and the amount of data of the computing results is small, the delays of the computing and result downloading are ignored." (R-2 has no UAV CPU model: its UAVs relay to a BS.)

(2) Current reading. Spec Section 1 and 2: UAV compute time is zero; T^0 = C_n / R_n; f_uav 1.4 GHz is unused. Code: `configs/lin2025.yaml:17` (`f_uav_hz`, "unused"), `src/uavmec/config.py:29`, `src/uavmec/env/lin_env.py:171` (`lat = bits / r`), no use of `f_uav_hz` anywhere under `src/`. Success = decoded and `lat <= ddl` and energy (`lin_env.py:175-183`). No downlink, no queue, no sharing.

(3) Alternatives. With literal support: only that Table 1 lists the UAV CPU frequency 1.4 GHz and Section 6 names "drone CPU frequency" among parameters taken from other works. That is support for the parameter existing, not for using it in the latency; the text states three times that the compute time is negligible. Result return and queueing: no support.

(4) Judgement: the current reading is the most literal one; it is stated directly and repeated, and R-2 makes the same simplification. The unused 1.4 GHz is a loose end in the paper, not a conflict.

(5) Hypothesis for S-3 (bounding case only). If the UAV computed each task at 1.4 GHz with the sensor's calibrated c_cyc = 24.5 cycles/bit (no value for the UAV is given in the paper, so this is not a literal reading), a 0.8 Mbit task would need 0.8e6 x 24.5 / 1.4e9 = 14 ms of compute (unit arithmetic), comparable to the deadlines of 1-50 ms; sharing the CPU among sensors on one UAV would multiply this. So this would make offloading much harder. Because it contradicts three explicit statements, it should be reported only as a bound.

### 2.7 Other assumptions that affect offloading latency or success

2.7.1 Rate formula and log base (Eq. 7).
(1) "R_n^{km} = B_m log(1 + gamma_n^{km}) >= R_0" (p. 4); "According to Shannon's formula, the offloading rate is expressed as formula (7)" (p. 4); no base printed. (2) Code: `np.log2` at `noma.py:39,47`; the spec writes log2 in Section 2 but gives it no A-ID. (3) Alternative: natural log (no support: "Shannon's formula" with bit/s units and R_0 in Mbps implies log2). (4) Current is the most literal. (5) ln would scale every rate by ln 2 = 0.693: slightly harder.

2.7.2 Transfer time T_n^0 (formula not in the paper).
(1) "T_n^0(t) represents the transfer time of offloading." (p. 4). No equation defines it. (2) Spec Section 2: T^0 = C_n / R_n, no A-ID; code `lin_env.py:171`. (3) No alternative with textual support (R-2 relates the UE's slot length to D_k / R, which gives the same C/R). (4) Only reading. It deserves an A-ID for completeness, since the latency of every offloaded task depends on it.

2.7.3 Rate threshold R_0 and the decode flag (A-7).
(1) Table 1: "Transmission rate requirement | 10 Mbps"; Eq. (7) "... >= R_0"; Eq. (15) "-1, R_n^{km} <= R_0"; Section 4.3 (p. 6): "To offload successfully, the offloading rate must meet the QoS threshold requirement: R_n^{km} >= R_0." Eq. (5): d_j = 1 "if sensor j is successfully decoded". (2) A-7: d_j = 1 iff R_j >= R_0; `noma.py:48`. (3) Alternative decode criteria (an SINR threshold) have no support. Equality: Eq. (7) says >= (success) and Eq. (15) says <= (failure): a measure-zero boundary difference. (4) Most literal. (5) The rate a deadline requires is C_n / D_n, which Table 1 puts between 2 Mbps (0.1 Mbit, 50 ms) and 1500 Mbps (1.5 Mbit, 1 ms); R_0 = 10 Mbps is the binding rate constraint only for the few combinations with C_n / D_n below 10 Mbps, otherwise the deadline binds (arithmetic from Table 1, not a simulation).

2.7.4 Transmit power range.
(1) Eq. (3) (p. 4): "p_n(t) in [0, p_max]"; Table 1 "Sensor power | 1 W"; Section 5.3 (p. 9): power action in [0,1] "to indicate the proportion of available ... transmission power". (2) `configs/lin2025.yaml:15`, `lin_env.py:149`. (3) None with support ("sensor power" could be an average, but Section 3 defines p_max). (4) Most literal.

2.7.5 Number of UAVs, subcarriers and sensors.
(1) Table 1: 2 UAVs, 4 carriers, 8 sensors; Section 6.4: N = 2, 4, 6, 8, 10, 12. Eq. (4): "sum_m sum_k V_n^{km}(t) = 1". (2) `configs/lin2025.yaml:5-7`. Note: K x M = 8 (UAV, subcarrier) pairs for N = 8 sensors, so every sensor can have its own carrier, and sharing is unavoidable only for N > 8. Eq. (3) and Eq. (5) sum only over sensors with V_n^{km} = 1 for the same (k, m), so there is no interference between different (k, m), as in the code (`noma.py:89-93`). (3) None. (4) Most literal.

2.7.6 Slot length and cycle.
(1) Table 1: "Time duration of a time slot | 1 s", "Number of time slots included in a cycle | 10 s". (2) `slot_s` 1.0 and `n_slots` 10 (A-9). The slot (1 s) is far longer than any transfer (ms), so it only enters the energy cap (A-18); it does not affect the latency or the success test. (4) Most literal; not a lever for offloading difficulty.

2.7.7 Task size units.
(1) Table 1: "Task size per time slot | 0.1-1.5 Mbits"; Section 3.1 (p. 3): "C_n(t) = C d_n(t) is the task size in bits"; Table 4: "Data size of task (Mb)"; Section 6.5 (p. 12): "task size is 0.825 Mbps" (a unit slip for Mbit). (2) Code: `task_min_bits: 100.0e+3`, `task_max_bits: 1.5e+6` (`configs/lin2025.yaml:27-28`); 1 Mbit = 1e6 bit. (3) Megabyte readings (x8 larger tasks) have no support: Table 1 says "Mbits", Table 4 uses "Mb" with a lower-case b, and Section 3.1 says "in bits". (4) Most literal. (5) A byte reading would make tasks 8 times larger, so harder.

2.7.8 Time-varying channel and UAV motion.
(1) h_n^k(t) carries "(t)" (Eq. 3); Section 3.3 (p. 4-5): the UAV positions q_k(t) change over the cycle; "four parameters need to be adjusted for each time slot, that is, the flight trajectory ...". (2) A-13: UAVs hover at positions sampled per episode, so the gain is constant within an episode (`lin_env.py:111-117`, `_new_slot`). The paper's own Table 3 and Fig. 6 give fixed positions. (3) Alternative: moving UAVs (needs a trajectory policy that the paper does not specify) or fading. (4) Tie, decided by the HUMAN on 2026-11-13 (TE-013). (5) Hypothesis: with an optimised trajectory the UAVs could move near the sensors, which would make offloading easier than with random hovering positions; fading would make it harder. Not testable without inventing the missing model.

2.7.9 Success definition for offloaded tasks.
(1) Eq. (14d): "T_n^l <= D_n, T_n^0 <= D_n, T_n^l <= tau"; Eq. (15) conditions: T^0 >= D_n, E ... >= E, R <= R_0 give -1, else +3; Section 4.3 (p. 6): "Latency: ... T_n^l(t) + T_n^0(t) <= tau". The symbol tau is the delay limit (p. 3), the soft-update factor (p. 8), and appears in Eq. (8). (2) Success = decoded and T^0 <= D_n and energy (`lin_env.py:175-183`). (3) A separate global limit tau in addition to D_n: no value for tau is given except possibly Table 1 "Maximum task latency requirement | 1-50 ms". (4) Most literal given D_n ~ U(1, 50) ms (A-33), settled by test T-1 on Local_Computing. Not an offloading-specific lever.

## 3. Cross-checks between the paper and the code (no numbers computed)

- Interference and noise are both computed per (UAV, subcarrier) group: `noma.py:89-93`. Interference does not depend on the bandwidth fractions of the sensors in the group (`noma.py:37-38`, `:45-46`), and the noise does not depend on B_n. Eq. (5)-(7) in R-1 do not make such a dependence either; if bandwidth fractions on a shared carrier were disjoint sub-bands (Table 4's sums are below 100%), the co-channel interference in Eq. (5) would not apply to disjoint sub-bands. The paper does not say whether sensors on one carrier overlap in frequency. This is part of N-1 and is another reason it needs a registered assumption.
- The observation B_k is constant 1.0 (`lin_env.py:88`), while the paper defines B_k(t) as a time-varying "available bandwidth": the observation space of our agents differs from the paper's in the same place as N-1.
- UAV positions are uniform random in the area at fixed height 100 m (A-13); Table 3 shows (77, 87) and (30, 53) at z = 100 (Fig. 6), consistent.

## 4. Unit conversions in the code, checked against the text

| Quantity | Paper | Config / code value | Conversion check |
|---|---|---|---|
| Carrier bandwidth | "250 MHZ" (Table 1) | `bandwidth_hz: 250.0e+6` | 250 MHz = 250e6 Hz: correct |
| Channel gain | "-50 dB" | `beta0: 1.0e-5` | 10^(-50/10) = 1e-5: correct, as a power gain at 1 m (A-1) |
| Noise | "-100 dB" | `noise_w: 1.0e-10` | 10^(-100/10) = 1e-10 relative to 1 W (dBW reading). If dBm were meant: 1e-13 W. No dB conversion exists in the code; the YAML holds linear values, so the dB reading is fixed in the YAML (A-2) |
| Sensor power | "1 W" | `p_max_w: 1.0` | correct |
| Rate requirement | "10 Mbps" | `rate_req_bps: 10.0e+6` | 10 Mbit/s = 1e7 bit/s: correct |
| Task size | "0.1-1.5 Mbits" | `task_min_bits: 100.0e+3`, `task_max_bits: 1.5e+6`; unit `task_unit_bits: 50.0e+3` (A-5) | 1 Mbit = 1e6 bit: correct |
| Deadline | "1-50 ms" | `deadline_min_s: 0.001`, `deadline_max_s: 0.050` | ms to s: correct (A-33) |
| Slot | "1 s" | `slot_s: 1.0` | correct |
| UAV CPU | "1.4 GHz" | `f_uav_hz: 1.4e+9` | correct; unused |
| UAV height, area | "L, W, H 100 m" | `uav_height_m: 100.0`, `area_m: 100.0` | correct |
| Output metrics | ms, mW | `metrics.py:35-36`: latency x 1000 (s to ms); energy x 1000 (J per 1 s slot to mW) | correct, given the 1 s slot (A-27, A-34) |
| Rate | bit/s | `bw * np.log2(1 + sinr)`, bw in Hz | bit/s: correct |
| Offload latency | s | `bits / rate`, bits and bit/s | s: correct |

No dB-to-linear or dBm-to-W conversion code exists; the only dB/dBm question is the reading of "-100 dB", which is the choice in item 2.2.

## 5. Items not resolvable from the text, and open leads

- The cited parameter sources [7] (Nguyen et al. 2023), [18] (Muhammad and Hossain 2021), [19] (Lu et al. 2022) are named by Lin for "parameters related to equipment and links". They are not in the repo and not in `reference.md`. They may state the channel gain, noise and path-loss conventions Lin copied. Reading them would need the HUMAN to supply the PDFs (or approve adding them to `reference.md`). Until then A-1 and A-2 cannot be settled further from text.
- The amount of Table 4 evidence: 8 sensors, all offloaders listed on UAV1 while the text and Fig. 6 give UAV0 to S3, S5, S6, S7 (`docs/lin2025_targets.md` A-31); MU4's power and bandwidth are swapped in the text (p. 12). T-2 could not separate A-1 x A-3 (14/16 for all candidates, TE-008). A new reading (N-1, A-8) cannot be tested on Table 4 because Table 4 gives the fractions explicitly.
- Learned-method numbers (Table 2, Figs. 4 and 5) were not used for any judgement in this audit (Rule 4). Using them to choose a reading would be calibration on a learning method.

## 6. Ranked list for S-3 (readings worth testing because the current one is less literal or tied)

All tests: non-learning policies (Local_Computing, NearestUAVFullPower, random offloading if available), validation seeds 2000-2099 only, one factor changed at a time from the frozen v1.0 environment, N = 8 and N in {2, 12}. The proposed values are measurement points, not choices; none may be selected by closeness to Lin's learned numbers.

1. N-1 shared carrier bandwidth (current is LESS literal). Proposed alternatives: (a) sequential depletion by sensor index, B_n = bw_frac_n x remaining bandwidth of (k, m); (b) proportional rescaling when the fractions on a carrier sum to more than 1; (c) as a bound, equal split of the carrier among its sensors. Needs a new A-ID, since the allocation rule is not in the text. Effect hypothesis: harder, possibly strongly.
2. A-1 constant gain (tie). Proposed value: 1e-5 for every link, against the current distance law; optionally the distance law with beta0 referenced to H (a convention only, flag it as such). Effect hypothesis: unclear for NOMA groups, easier for single-user carriers.
3. A-8 literal Eq. 5 (tie). Proposed alternative: interference of j on n = |h_n|^2 p_j with the decoding order by transmit power (equal powers give no cancellation). Effect hypothesis: harder for equal-power heuristics.
4. A-3 total split (current is more literal, but it is the known lever and already measured once in T-2). Proposed value: 62.5 MHz per subcarrier. Effect hypothesis: harder. Repeat on the frozen environment for the sensitivity table.
5. A-2 noise level (current is the most literal; test as a unit-reference sensitivity). Proposed values: 1e-13 W (dBm reading) and 1e-12 W (-174 dBm/Hz x 250 MHz, convention only). Effect hypothesis: easier.
6. UAV compute bound (current is the most literal). Proposed: compute time = C_n x c_cyc / 1.4 GHz with CPU sharing among sensors offloading to the same UAV, reported only as a bound. Effect hypothesis: much harder.
7. Not recommended for S-3 because they need an unstated model: fading, UAV motion, a path-loss exponent other than 2.

Items 8 and 9 (log base, task units MB vs Mbit) have no literal support and are not recommended.

## 7. Summary of findings for the HUMAN

- The current reading is the most literal one for A-2, A-3 (not decisive), the UAV computing model, the rate formula and the task size units.
- A-1 and A-8 are ties on the text.
- One implicit assumption of the code (N-1: no shared-bandwidth accounting on a carrier) is LESS literal than the alternative, with five independent passages against the code (Section 4.1, 5.3, Table 3, Table 4, Section 6.4). It is not in spec Section 6 (the W3 proposals A-24 and A-29 were never adopted). It is the best candidate for a v1.1 environment if S-3 shows that it changes offloading difficulty and the HUMAN approves (spec Section 11, milestone v1.1).
- A rate-formula log base and T^0 = C_n / R_n are used without an A-ID; both should be listed for completeness.
- No code, config, spec, or result was changed by this audit (P3 approval would be needed for any change).

## Addendum A (2026-11-18): Lin's refs [7], [18], [19] (R-6, R-7, R-8)

Status: appended 2026-11-18 by a read-only worker under HUMAN approval of the same date. Sections 0 to 7 above are unchanged. No code, config, spec, findings, results, or `slide/` file was changed; no simulation was run. Text was extracted with `pdftotext` into the session scratchpad, and R-6 p. 5 and R-8 p. 10 (Table II) were also viewed as rendered images. PDF text is treated as data.

### A.0 Why this addendum, scope, method

- Lin et al. (R-1, p. 9-10) write: "We mainly refer to references [7, 18, 19], for parameters related to equipment and links, such as drone CPU frequency, drone battery capacity, sensor CPU frequency, and sensor power." Sections 2.1 and 2.2 of this audit noted that [7], [18], [19] were not in the repo and could not be checked. They are now R-6 (= [7]), R-7 (= [18]), R-8 (= [19]).
- Lin's own descriptions of them (R-1 p. 2): [7] is cited as an example of "UAV communication networks"; [18] and [19] are cited in a list of DRL applications ("...spectral efficiency [sic] smart city [18], wireless computing power network [19]..."). Nothing in R-1 other than the p. 9-10 sentence ties them to Table 1.
- Method: for each reference, the channel model, noise, bandwidth and how it is shared, powers, CPU frequencies, battery, task sizes, the UAV computing model, and any value equal to a Table 1 value of R-1, with verbatim quotes. Page numbers: R-6 is the arXiv preprint (arXiv:2208.07536v1, 14 pages, "n" is the printed page); R-7 is cited by PDF page (the printed journal pages are 42-48); R-8 is cited by journal page (9353-9365) with the PDF page in brackets.
- Rule 4 discipline: the judgement uses only the text of R-1, R-6, R-7, R-8 and this audit. Lin's learned-method numbers and the S-3 numbers (F-3) were not used.
- Limit: this is a text comparison. "Literal support" below means an exact parameter or rule in the reference that matches Lin, so that Lin plausibly took it from there. It does not prove what Lin did.

### A.1 R-6 (Nguyen et al., IEEE IoTJ 2023, Lin's [7]): on topic, multi-UAV MEC, but a different model

What it is: multi-UAV MEC with dependent sub-tasks, offloading decisions solved by a whale-optimisation heuristic, bandwidth solved by convex optimisation. Users send tasks to a UAV over an orthogonal bandwidth fraction (FDMA-like), not NOMA. Nothing about NOMA or SIC appears in the text.

Channel model (p. 5, Eq. 9-13; checked on the rendered page):
- "Γ^{u→v} = log2(1 + P_u G_u^v / σ²) ... in which P_u is the power of user u, G_u^v is the channel gain between user u and UAV v, while σ² is white the Gaussian noise power." (Eq. 9)
- "G_u^v = 10^{-δ_u^v/10}" (Eq. 10), with δ the path loss, a LoS/NLoS mixture (Eq. 11-16): "δ_u^{v,LoS} = 2n log2(4π d_u^v B_c^lte / c) + L_LoS ... where n ≥ 2 is the path loss exponent, B_c^lte denotes the carrier frequency (i.e., 2 GHz) ... L_LoS and L_NLoS are the average added losses". The LoS probability uses "z_v is the hovering altitude of UAV v" and unspecified environment constants C and D (Eq. 14).
- No "-50 dB", no reference distance, no gain at 1 m. The gain depends on distance, carrier frequency and LoS probability.

Noise (Table II, p. 10, and Eq. 9): "σ² -174dBm". It is used as σ² directly in Eq. 9 and Eq. 28, with no bandwidth factor (the table prints dBm, not dBm/Hz). Not -100 dB.

Bandwidth and how it is shared (the main finding of this reference):
- Table I (p. 5): "B^v: Total accessible bandwidth that UAV v can allocate to its users"; "β_u^v: The percentage of bandwidth will be allocated to user u".
- Eq. 17 (p. 6): "R^{u→v} = β_u^v B^v Γ^{u→v}", "where β_u^v is the percentage of total available bandwidth of UAV v (denoted as B^v) that will be allocated to user u."
- Constraints (45e) and (45f) (p. 8): "Σ_{u∈U_v} β_u^{v→w} ≤ 1 ... β ∈ [0, 1]", and the text: "constraint (45e) and (45f) guarantee that the total fraction bandwidth allocated to users should be less than the maximum available bandwidth at a UAV." The same constraint appears in (54b) (p. 9): "Σ β^{v→u} ≤ 1, ∀v ∈ V".
- How the fractions are chosen: the BS solves a convex problem ("the problem of allocating communication resources is convex ... SCS in CVXPY", p. 2 and p. 9). It is an optimised variable with a sum constraint, not an enforcement rule for an agent's independent actions.
- Two named baseline schemes (p. 11): "In the equal resource allocation scheme, all communication resources (i.e., bandwidth) available at the UAV are equally allocated among its associated users within the area, whether they need it or not in a fair manner. While in the proportion resource scheme, the MEC-enabled UAV allocates communication resources to the users based on the network information: the active users and the size of offloading tasks ..."
- p. 12: "This happens because of the limitation of bandwidth, more users have to share a fixed amount of bandwidth."
- Value: B^v = 3 MHz per UAV (Table II) and B^{v,w} = 8 MHz for the UAV-UAV link. Not 250 MHz, no carriers or subcarriers.

Transmit power (Table II): "P_u 23 dBm" (about 0.2 W); UAV "P_v 30 dBm". Lin: 1 W.

Sensor CPU, battery, task size: no sensor CPU frequency. UAV battery: "E_v^max = N_u × 2,000 ... = 20 kJ" (p. 11). Task sub-task size "Gaussian distribution with a mean value of 6 MB ... variance 1 MB" (p. 10). Lin: sensor CPU 1 GHz, UAV battery 100-500 kJ, tasks 0.1-1.5 Mbit.

UAV computing time (modelled here, unlike in Lin):
- Eq. 22 (p. 6): "t_{u,j}^{exe} = C_u H_{u,j} / f_{u,j}^v".
- Eq. 23-24 (p. 6): the UAV CPU is shared in proportion to input size: "f_{u,j}^v = H_{u,j} / Σ_{q∈U_v} H_{u,q} · F_v^max", with "Σ θ f ≤ F_v^max". Eq. 25: energy "k_v (f)² C_u Z", "k_v = 5 × 10^{-27}".
- UAV CPU frequency: Table II "F_v^max [800, 1000] MHz". Lin Table 1: 1.4 GHz. Mismatch.
- Altitude: "UAV at an altitude of around 50 m" (p. 10). Lin: H = 100 m.

Coincidences with R-1 Table 1: none (searched for 250 MHz, 4 carriers, -50 dB, -100 dB, 1.4 GHz, 1 GHz, 1 W, 100-500 kJ, 10 Mbit/s, 0.1-1.5 Mbit).

### A.2 R-7 (Muhammad and Hossain, IEEE Wireless Commun. 2021, Lin's [18]): OFF-TOPIC

- It is a smart-grid energy-scheduling article: "This article proposes an efficient DRL-based energy scheduling approach that can effectively distribute the energy devices based on consumption and users' demand" (abstract, PDF p. 1). The experiments use the RELOAD household-demand dataset (PDF p. 6): "In the experiments, we use the RELOAD dataset [15]" with metrics delay, cost in US dollars and waiting time.
- The words UAV, drone, aerial, unmanned occur zero times. The text has no channel model, no noise, no bandwidth value, no transmit power, no CPU frequency, no battery capacity, no task size. The only communication-related text is generic ("edge-cloud ... bandwidth of the edge server is sufficient", PDF p. 5).
- Nothing in it can support any of A-1, A-2, A-3, N1 or the UAV computing model, and it does not contain drone CPU, drone battery, sensor CPU or sensor power values. Its inclusion in Lin's sentence is therefore not traceable to anything in the PDF (Lin cites it in a DRL-applications list; see A.0).

### A.3 R-8 (Lu et al., IEEE IoTJ 2023, Lin's [19]): on topic for MEC and DRL, no UAV

What it is: wireless computing power networks (end devices and MEC servers on base stations; no UAV), multi-agent DRL, OFDMA uplink.

Channel model (p. 9357 [PDF p. 5], Eq. 7-9): "r_ij(t) = B_ij(t) log(1 + p_{u_i}(t) h_ij(t) d_ij^s(t)^{-α} / (σ² + I))", "where B_ij(t) is the bandwidth that BS s_j allocates to user u_i at time slot t, p_{u_i}(t) is the transmission power of user u_i, h_ij(t) is the current channel gain between user u_i and BS s_j", and "where −α is the path loss exponent between end user u_i and server s_j. σ² is the noise power." A channel gain h times a distance law d^{-α}. The value of α, the value of h and any gain at a reference distance are not given. No -50 dB. The log base is not stated. Users are within "a radius of 1500 m" (p. 9361). No altitude.

Noise (p. 9361, text): "The noise power N0 is set to −174 dBm/Hz." Table II (p. 9362 [PDF p. 10]) "Noise power function N_0: -174 dBm/Hz". This is a density; the equations write σ² and do not state the multiplication by bandwidth. Not -100 dB.

Bandwidth and how it is shared (Eq. 7 and text): "In our wireless transmission, we adopt the orthogonal frequency division multiple access (OFDMA) model [34] to mitigate the effect of interference." (p. 9357). Per-user bandwidth is B_ij(t), "that BS s_j allocates to user u_i", and the action includes "the channel allocation strategy b^m(t) ... channel resources that assigned to each end user" (Eq. 21, p. 9359). There is no explicit constraint that the B_ij of one BS sum to at most its total bandwidth (constraints (18a)-(18e), p. 9358-9359, only cover association, the sum of sub-task sizes and computation resource). One "average" benchmark "evenly assign[s] the resources to the computation task" (p. 9361). Value: "We set the transmission channel bandwidth B to 20 MHz" (p. 9361; Table II: "Transmission channel bandwidth B | 20 MHz"). Not 250 MHz, no carriers.

Transmit power: "set the transmission power of the user device to 0.2 W [36]" (p. 9361; Table II "User device transmit power p_i | 0.2 W"). Lin: 1 W. Mismatch.

CPU: Table II "CPU frequency of edge server f_j, f_i | [8,24] GHz" (it covers both servers and end devices). Lin: sensor 1 GHz, UAV 1.4 GHz. Mismatch. No battery capacity. Task sizes are not tabulated.

Computing time:
- Local (Eq. 6, p. 9357 [PDF p. 5]; definition on the previous page): "The number of CPU cycles required to execute a one-bit computing task is denoted as ζ. Thus, the computation latency of executing task D_{u_i}(t) by local computing can be written as T_i^cmp = ζ·D_{u_i}(t) / f_i^u(t)." This is the same structure as our A-4 (T^l = C_n c_cyc / F_n, cycles per bit times bits over CPU frequency). The value of ζ is not given.
- Energy (Eq. 13): "E_i^k(t) = D_k φ_i ζ_i f_i²" (a switched-capacitance model).
- Edge server execution time is modelled (Eq. 12, execution time with the assigned computation resource f_k of server k), so computing time is not neglected in R-8 either.

Coincidences with R-1 Table 1: none (same search as for R-6).

### A.4 Mismatch table (values in the references against R-1 Table 1)

| Quantity | R-1 Table 1 | R-6 | R-7 | R-8 |
|---|---|---|---|---|
| Channel gain | -50 dB (bare) | distance, LoS/NLoS, 2 GHz, n >= 2, no -50 dB | none | h x d^-alpha, alpha and h not given |
| Noise | -100 dB | -174 dBm (used as sigma^2) | none | -174 dBm/Hz |
| Bandwidth | 4 carriers x 250 MHz | B^v = 3 MHz per UAV, fractions with sum <= 1 | none | B = 20 MHz, OFDMA, B_ij per user |
| Sensor power | 1 W | 23 dBm (about 0.2 W) | none | 0.2 W |
| UAV CPU | 1.4 GHz | 800-1000 MHz | none | 8-24 GHz (servers and devices) |
| Sensor CPU | 1 GHz | not given | none | 8-24 GHz |
| UAV battery | 100-500 kJ | 20 kJ | none | none |
| Task size | 0.1-1.5 Mbit | 6 MB sub-tasks | none | not tabulated |
| UAV computing time | negligible | modelled (Eq. 22-23, shared CPU) | none | modelled (edge server) |
| Altitude | 100 m | about 50 m | none | none |

Finding: none of the "equipment and links" values that Lin lists (drone CPU, drone battery, sensor CPU, sensor power) can be traced to a value in R-6, R-7 or R-8, and none of Lin's Table 1 values for the channel gain, noise, carrier number or bandwidth appear. The values that exist in the references differ from Lin's. So the p. 9-10 sentence is a loose citation of general sources, not a parameter source that can be checked.

### A.5 Judgement per item (from text only)

Scale: LITERAL (exact parameter match plus the same model), WEAK/INDIRECT (same family of model, no matching value), NOTHING.

**A-1 channel gain (-50 dB).** NOTHING literal. R-6 and R-8 both use distance-dependent gains (R-6 a LoS/NLoS carrier-frequency model, R-8 a d^-alpha law with unspecified constants); neither contains -50 dB, a 1 m reference gain, or a constant gain. R-7: off-topic. WEAK/INDIRECT only for the structure "gain depends on distance", which neither confirms nor excludes the constant reading of Table 1 but is the only form used by the two on-topic references. The tie verdict of Section 2.1 stands; R-6 and R-8 lean slightly toward a distance law. No change to S-2.

**A-2 noise (-100 dB).** NOTHING literal. The two references that give a noise value use -174 dBm (R-6, as a plain sigma^2) and -174 dBm/Hz (R-8). Neither gives -100 of any unit. Weak structural point: in R-6 Eq. 9, sigma^2 is a stand-alone power added in the SINR with no bandwidth factor, the same form as Lin's Eq. 6, so the "power per link, no bandwidth factor" reading is consistent with a reference in Lin's list. The unit (dB, dBm, dBm/Hz) is not settled by any of them. S-2 verdict (current reading most literal, unit by convention only) unchanged.

**A-3 bandwidth per subcarrier (250 MHz per carrier vs B/M).** NOTHING literal: no reference has carriers, subcarriers, 250 MHz or B/M. R-6 states the total bandwidth per UAV (3 MHz) is divided among users, which is the same kind of statement as Lin's Section 3 "B into M subcarriers ... B/M" and does not discriminate between per-carrier and total-split readings. S-2 verdict (per carrier, most literal, not decisive) unchanged.

**N1 (shared-bandwidth allocation rule).** Direction: R-6 is the most relevant text and it supports the qualitative N-1 finding of Section 2.4. In R-6 (and the model family Lin took parameters from) the bandwidth of a UAV is a shared resource: R = beta B Gamma with 0 <= beta <= 1 and "total fraction bandwidth allocated to users should be less than the maximum available bandwidth at a UAV" (Eq. 17, 45e, 54b). The v1.0 code has no such budget (Section 2.4 (2)). R-8 allocates per-user bandwidth B_ij "that BS s_j allocates" under OFDMA but states no sum constraint. Limits, all from the text:
- (a) R-6 gives a constraint (sum of fractions <= 1), solved by convex optimisation at the BS. It gives no rule for enforcing the constraint on independent agent actions, so it does not choose among N1-seq, N1-prop and N1-eq.
- (b) The two named R-6 schemes are "equal" allocation among the associated users of a UAV, and "proportional" to active users and offloading task sizes. R-6's proportional rule is by task size, not by the actor's requested fraction, so N1-prop (f_n B / max(1, S)) is not R-6's proportional scheme. N1-eq resembles R-6's equal scheme but R-6's scheme is per UAV (all associated users), not per (UAV, subcarrier).
- (c) R-6 and R-8 are orthogonal-access models (FDMA, OFDMA); neither has NOMA. In Lin's NOMA model, sensors on one subcarrier are separated by SIC and Eq. 7 writes B_m for the subcarrier. So moving an orthogonal split rule into Lin's NOMA groups is an assumption of ours, not something the references state.
- (d) N1-seq (sequential depletion) has no support in any of the three.
So: WEAK/INDIRECT support that a per-UAV bandwidth budget exists in Lin's reference family (this strengthens the S-2 finding that "no budget" is the less literal reading); NOTHING for any specific allocation rule. It also does not meet the F-4 condition ("an original source or appendix describing the allocation rule"): R-6 and R-8 are not Lin's description of his own rule.

**UAV computing model.** R-6 DOES model UAV computing time and proportional CPU sharing (Eq. 22-25) and R-8 models edge-server execution time; neither uses 1.4 GHz (R-6: 800-1000 MHz; R-8: 8-24 GHz). Lin states three times that the UAV computing time is negligible and does not use the 1.4 GHz value. So the reference family contains the "UAV compute plus sharing" model that the S-3 bound UAV-cb mimics, which makes the 1.4 GHz in Table 1 a parameter inherited from a model Lin did not use. This is WEAK/INDIRECT support for a CPU-sharing model and NOTHING that overrides Lin's explicit "negligible computation time" (R-1 p. 4, p. 10-11). S-2 verdict (current reading most literal; UAV-cb only a bound) unchanged.

**Side finding (A-4).** R-8 Eq. 6 (T = zeta D / f, zeta in cycles per bit) is the same form as our A-4 (T^l = C_n c_cyc / F_n). It is a text support for A-4 from Lin's own reference list; zeta's value is not given, so it does not touch A-33 or the calibrated c_cyc.

### A.6 Verdicts for the HUMAN

- **A-1**: tie stays. R-6 and R-8 use distance-dependent gains and none uses a constant gain or -50 dB; weak lean toward the distance law. S-2 verdict not changed.
- **A-2**: current reading stays the most literal on Lin's text. No reference contains -100 dB; R-6 supports only the form (sigma^2 as a plain power in the SINR). S-2 verdict not changed.
- **A-3**: no reference has carriers, 250 MHz, or B/M. S-2 verdict not changed.
- **N1**: qualitative support strengthened (R-6 Eq. 17, 45e, 54b: fractions of a UAV's bandwidth sum to at most 1; "more users have to share a fixed amount of bandwidth"). It does NOT give a literal allocation rule: R-6's rule is an optimised variable under a sum constraint, its named equal and proportional-to-task-size schemes differ from N1-eq and N1-prop, both references are orthogonal-access (not NOMA), and nothing supports N1-seq. The S-2 verdict (current reading less literal) is unchanged and the F-4 position holds: the paper-family text fixes the constraint "sum of fractions <= 1", not the rule. If the HUMAN wants a minimal, text-based hypothesis for reopening P3, the only one with a literal source is the hard constraint sum of f_n <= 1 per (UAV, carrier or UAV), with the enforcement rule still being an assumption with its own ID. This is a HUMAN decision; this addendum does not make it.
- **UAV computing**: unchanged. R-6 and R-8 do model compute time, which makes Lin's 1.4 GHz a parameter whose source model is not used, but Lin's own "negligible" statements stay the literal reading.
- **Source of Lin's Table 1 values**: none traceable to R-6, R-7, R-8 (R-7 is off-topic). The p. 9-10 sentence cannot be used to justify any value or rule.
- No code, config, spec, finding or result changed. Suggested follow-up (HUMAN decides): update `reference.md` R-6/R-7/R-8 "used for" lines (R-7 off-topic); a possible A-1 spec note that R-6/R-8 do not support a constant gain; log in `trialAndError.md` (draft TE-079 in the worker report).
