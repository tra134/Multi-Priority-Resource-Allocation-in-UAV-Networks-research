# Lin et al. 2025: reproduction targets and spec audit

Source: Lin, Xu, Li, Wang, Li, IET Communications 19:e70063 (2025), 14 pages (PDF in `slide/Dai_driver/`). All page numbers are the journal's "n of 14". Figure values are in `docs/data/lin2025_digitized.csv`. Everything marked "visual" is an approximate read of a rendered page, not data from the authors. Text in the PDF is treated as data.

## (a) Numbers stated in the text and tables

### Table 1 (p. 10), simulation parameters
| Parameter | Value (exact) |
|---|---|
| Number of UAVs | 2 |
| Number of carriers | 4 |
| Carrier bandwidth | 250 MHZ |
| UAV CPU frequency | 1.4 GHz |
| UAV battery capacity | 100-500 KJ |
| Number of sensors | 8 |
| Sensor CPU frequency | 1 GHz |
| Sensor battery capacity | 100-400 KJ |
| Sensor power | 1 W |
| Task size per time slot | 0.1-1.5 Mbits |
| Maximum task latency requirement | 1-50 ms |
| Transmission rate requirement | 10 Mbps |
| Batch size | 320 |
| Actor / Critic learning rate | 0.0003 / 0.00005 |
| Buffer size | 2^17 (checked in the rendered page; text extraction shows "217") |
| Discount factor | 0.9 |
| Time duration of a time slot | 1 s |
| Number of time slots included in a cycle | 10 s |
| L, W, H | 100 m |
| V_max | 50 m/s |
| Noise | -100 dB |
| Channel gain | -50 dB |

### Statements in the text
| Page | Quote | Content |
|---|---|---|
| 3 | "each UAV allocates the frequency resource with bandwidth B into M subcarriers on average, and the transmission bandwidth on each subcarrier is B/M" | bandwidth model |
| 3 | "𝐶𝑛(𝑡) = 𝐶𝑑𝑛(𝑡) is the task size in bits, where 𝐶 is the bit size of each data cell and 𝑑𝑛(𝑡) is the number of data cells in the task, following a Poisson distribution with parameter 𝜆" | Poisson task; C and lambda not given |
| 3 | "𝐷𝑛(𝑡) is the latency requirement of the task, which is proportional to 𝐶𝑛(𝑡)" | D_n rule; no constant |
| 3-4 | Eq. 1 "T^l_n = D_n / F_n"; Eq. 2 "e^l_n = (1-U_n) min{z_n D_n, E_n}"; "z_n = kappa (F_n)^2" | local model; D_n appears in BOTH eq. 1 and 2; no cycles-per-bit anywhere |
| 4 | Eq. 7 "R = B_m log(1+gamma) >= R_0"; "Here B_m is the bandwidth" | rate; log base not stated |
| 4 | Eq. 8 "e^0_n(t) = tau p_n(t) U_n(t)" | tau also named as the latency limit ("completed within delay time tau") and as the soft-update factor (p. 8) |
| 4 | "T^l_n(t) < tau" ; (14d) "T^l <= D_n, T^0 <= D_n, T^l <= tau" | latency constraints |
| 4 | "It is assumed that the UAVs have enough computation resources and negligible computation time" | UAV compute time = 0 |
| 4-5 | Eq. 11 flight constraint, "flight cycle consists of T time slots", "fixed altitude H", safe distance d_s (no value) | trajectory constraint only |
| 5 | "four parameters need to be adjusted for each time slot, that is, the flight trajectory, the offloading access, the offloading power of sensors, and the bandwidth allocation" | trajectory named as a decision |
| 5 | S_n = {x_n, y_n, E_n, C_n, D_n}; S_k = {x_k, y_k, H, B_k, E_k}, B_k = available bandwidth of the UAV's m carriers | state |
| 5 | "U_n = {0,1,...,K} ... U_n = 0 means local computation" | action |
| 6 | Eq. 15: -1 if T^l >= D_n, e^l >= E_n, T^0 >= D_n, sum E >= E_max, R <= R_0; "3, others" | reward |
| 9 | "three (k+1, m, 2)-dimensional hidden layers after a 128-dimensional hidden layer and the Relu6 activation" | actor |
| 9 | "critic ... 64-dimensional hidden layer and a 32-dimensional hidden layer" | critic |
| 9 | "learning rates for the actor network and the critic network to 0.0003 and 0.00005" | |
| 9 | "Gumbel-Softmax" (no temperature stated anywhere) | |
| 10 | "We mainly refer to references [7, 18, 19], for parameters related to equipment and links, such as drone CPU frequency, drone battery capacity, sensor CPU frequency, and sensor power" | where parameters come from |
| 10 | "a system with 2 UAVs and 8 sensors, each equipped with 4 subcarriers of 250 MHz bandwidth" | bandwidth |
| 10 | "POS_MADDPG_NOMA algorithm achieves a success rate of 85.63%, which is 39.8%, 40.6%, and 20% higher than Local_Computing, FOS_MADDPG_NOMA, and POS_MAPPO_NOMA" | see audit A-15 |
| 10 | "lowest average latency of 7.48 ms ... Local_Computing 18.76 ms ... FOS 8.11 ms ... POS_MAPPO 12.5 ms" | |
| 10 | "Local_Computing and POS_MAPPO_NOMA consume the most energy, 18.76 mW and 11.41 mW ... POS_MADDPG 5.68 mW, FOS 4.195 mW" | |
| 10 | "Figure 4 depicts ... rewards per time slot for each sensor agent" (axis values reach 25; see audit) | |
| 11 | Fig. 4 caption "B_max = 250, K = 2, and N = 8" | |
| 11 | "reduce the task success rate by 30.83% and 30.14%, respectively, compared with FOS_MADDPG_NOMA and POS_MADDPG_NOMA" | relative %, checked: (60.88-42.11)/60.88 = 30.83%; (85.63-59.82)/85.63 = 30.14% |
| 11 | "reduces the task success rate by 43.48% and increases the latency by 8.96%, compared with POS_MAPPO_NOMA" | (61.25-34.62)/61.25 = 43.48% relative; latency 12.50 -> 11.38 is a DEcrease of 8.96%, so "increases" is wrong in the paper |
| 11 | "Under the same resources and the fixed total bandwidth, as N increases, the success rate ... decrease gradually, while the average latency and energy consumption increase gradually" | no energy panel exists in Fig. 5 |
| 11 | "FOS_MADDPG_NOMA algorithm has the most successful tasks when the number of sensors is small (N = 2 and 4)" | |
| 12 | "When N > 4, our proposed algorithm POS_MADDPG_NOMA surpasses the other three baseline algorithms" | |
| 12 | "when N is 10, the success rate of POS_MADDPG_NOMA algorithm reaches more than 78%, which is 27.3%, 19.5%, and 12.9% higher than Local_Computing, POS_MADDPG_NOMA [sic, = MAPPO], and FOS_MADDPG_NOMA" | checks against Fig. 5a: 78/61.4 = 1.27, 78/65.3 = 1.19, 78/69.2 = 1.13 (visual) |
| 12 | "uav0 (77, 87), uav1 (30, 53), carriers 100%, available power 70%, 72%" | Table 3 |

Table 2 (p. 11), exactly: FOS_MADDPG_NOMA 60.88% / 8.11 ms; FOS_MADDPG_noSIC 42.11% / 13.50 ms; POS_MAPPO_NOMA 61.25% / 12.50 ms; POS_MAPPO_noSIC 34.62% / 11.38 ms; POS_MADDPG_NOMA 85.63% / 7.48 ms; POS_MADDPG_noSIC 59.82% / 13.08 ms.

Table 4 (p. 12), 8 sensors: (id; x,y; power%; size Mb; latency req s; action; ok). MU0 35,14; 80; 1.1; 0.042; UAV1 c0 bw 99.9 p 99.9; yes. MU1 2,21; 55; 0.825; 0.033; local; yes. MU2 55,35; 67.5; 0.9; 0.038; local; yes. MU3 82,21; 77.5; 0.9; 0.025; UAV1 c2 bw 47.8 p 36.2; yes. MU4 62,35; 32.5; 0.9; 0.035; UAV1 c3 bw 61.1 p 31.1; yes. MU5 1,59; 92.5; 1.31; 0.033; UAV1 c2 bw 45.3 p 35.7; yes. MU6 56,40; 97.5; 1.11; 0.021; UAV1 c1 bw 30.8 p 51; NO. MU7 46,34; 87.5; 0.51; 0.014; UAV1 c1 bw 42.2 p 49; yes.

Facts computed from Table 4 (not stated in the paper): D_n / C_n ranges 0.0189 to 0.0422 s per Mb (8 rows, so "proportional" is only approximate; matches spec A-6). Local tasks that succeeded: MU1 0.825 Mb in 33 ms, MU2 0.9 Mb in 38 ms, so cycles per bit c_cyc <= about 40 to 42 (at F = 1 GHz, 1 Mb = 1e6 bit).

### Where the paper is silent (searched in the full text)
No path-loss model or carrier frequency; no noise bandwidth; no cycles-per-bit; no C or lambda; no D_n/C_n constant; no episode length in slots (only "cycle T = 10"); no soft-update tau value; no optimizer; no exploration noise; no Gumbel temperature; no sampling-per-step or update-frequency; no d_s; no UAV energy parameters; no seeds or number of runs; no confidence intervals.

## (b) Digitized figure values (visual, from 200 dpi renders; see CSV)

Fig. 4, N = 8, K = 2, x axis = episode 0 to ~9800 (about 100 points per curve, so about 10,000 training episodes; panels b-d start at ~900). Converged values (visual mean of the last ~2000 episodes), with the paper's stated number in brackets:

| Algorithm | Reward (a) | Success % (b) | Latency ms (c) | Energy mW (d) |
|---|---|---|---|---|
| Local_Computing | ~15.1 | ~62 (+/-2) | ~18.5 [18.76] | ~18.5 [18.76] |
| FOS_MADDPG_NOMA | ~11.9 | ~60 (+/-2) [60.88] | ~8.2 [8.11] | ~4.2 [4.195] |
| POS_MAPPO_NOMA | ~19.7 | ~70 (+/-2) [Table 2 says 61.25] | ~12.0 [12.5] | ~11.0 [11.41] |
| POS_MADDPG_NOMA | ~23.7 | ~85.3 [85.63] | ~7.0 [7.48] | ~5.4 [5.68] |

Fig. 4 observations (visual):
1. POS_MADDPG success is flat at ~70-73% until ~4000 episodes, then jumps to ~82-85% at ~4400-5000 and stays. POS_MAPPO is noisy (+/-5 points) and ends ~70%. FOS is flat ~55-62%, with a dip to ~47% at the start.
2. Reward axis reaches ~24 for POS_MADDPG. With the Eq. 15 reward (+3 / -1) the mean reward per slot is 4*SR - 1 = 2.43 at SR = 85.63%. 10 slots x 2.43 = 24.3, matching the plot. So the "reward" in Fig. 4a is a SUM over a 10-slot episode (not per slot, as the text says) and one episode = 10 slots. Same check: Local ~15.1 -> SR = (1.51+1)/4 = 63%, matches Fig. 4b (~62%). FOS 11.9 -> 55%, MAPPO 19.7 -> 74%: roughly consistent with Fig. 4b given the noise.
3. The Local_Computing curve in Fig. 4c and Fig. 4d is the same series (identical shape, different axis label); in energy = kappa F^3 * latency form this means energy (mW) is numerically equal to latency (ms), i.e. kappa F^3 = 1 W, e.g. kappa = 1e-27 with F = 1 GHz.
4. In Fig. 4b Local_Computing is ~62%, in the same band as FOS. POS_MAPPO is ~70%, not 61.25%.

Fig. 5 (success rate and latency only, no energy panel; N = 2..12; Local, FOS, POS_MAPPO, POS_MADDPG; 6 points per curve). Success %, then latency ms (visual, +/-1 pt, +/-0.3 ms):

| N | Local | FOS | MAPPO | POS_MADDPG | | Local | FOS | MAPPO | POS_MADDPG |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 60.6 | 100 | 95.9 | 99.6 | | 17.2 | 0.55 | 3.9 | 1.1 |
| 4 | 62.5 | 95.7 | 78.7 | 93.3 | | 18.1 | 2.8 | 7.7 | 4.5 |
| 6 | 61.0 | 87.2 | 77.1 | 92.7 | | 18.2 | 6.2 | 8.2 | 5.05 |
| 8 | 61.3 | 80.9 | 71.2 | 85.5 | | 18.8 | 8.1 | 12.5 | 7.4 |
| 10 | 61.4 | 69.2 | 65.3 | 78.0 | | 19.0 | 9.9 | 15.2 | 9.1 |
| 12 | 61.4 | 64.1 | 62.7 | 71.2 | | 18.8 | 10.9 | 16.9 | 10.25 |

Fig. 5 observations: at N = 2 FOS and POS_MADDPG are both ~100% (FOS marginally best in latency); at N = 4 FOS (95.7) beats POS_MADDPG (93.3), at N = 6 POS_MADDPG wins; matches the text. Success decreases and latency increases with N for all learned methods. At N = 8, POS_MADDPG = 85.5% and latency 7.4 ms agree with Fig. 4. INCONSISTENCY: FOS at N = 8 is ~81% in Fig. 5a but ~60% (60.88%) in Fig. 4b / Table 2, and POS_MAPPO at N = 8 is ~71% in Fig. 5a, which agrees with Fig. 4b (~70) and NOT with Table 2 (61.25). Latency values at N = 8 agree between Fig. 4c and Fig. 5b for all four.

Fig. 6 (p. 13): a 3D picture of the allocation of Tables 3 and 4 (sensors S0..S7 with [power, size, delay requirement], UAV0 at 70% and UAV1 at 72%). In the figure UAV1 serves S0 and S4 and UAV0 serves S3, S5, S6, S7 (matches the Section 6.5 text) but Table 4 lists all offloaders as "UAV1". Table 4 and the text disagree.

## (c) Audit of `docs/lin2025_spec.md`

Legend: OK, WRONG, MISSING-NUANCE.

### Section 1 (scenario)
| Row | Verdict | Evidence |
|---|---|---|
| K = 2, N = 8, M = 4 | OK | Table 1; p. 10 "2 UAVs and 8 sensors" |
| B_m = 250 MHz per subcarrier | MISSING-NUANCE (see A-3) | p. 3 says total B split into M subcarriers of B/M; p. 10 says "each equipped with 4 subcarriers of 250 MHz"; Fig. 4 caption "B_max = 250"; p. 11 "fixed total bandwidth". Three of four statements point at 250 MHz as B (total), not per subcarrier |
| Area, H = 100 m, v_max 50 m/s, xi 1 s, p_max 1 W, F_n 1 GHz, UAV 1.4 GHz | OK | Table 1 |
| Slots per cycle T = 10, "episode" | OK, with nuance | Table 1 "10 s". Algorithm 1 has separate "episode ep = 1 to c" and "step t = 1 to L". The paper never says L = T. Indirect support: Fig. 4a reward ~24 = 10 x 2.43 (see (b)), so 10 slots per episode is consistent. Still an inference |
| Batteries U(100,400) / U(100,500) kJ | MISSING-NUANCE | Table 1 only gives a range and the distribution is not stated. Table 3/4 show initial "available power" as percent (sensors 32.5-97.5%, UAVs 70/72%), so batteries start at a random FRACTION of capacity, not full, in the paper's example |
| Task size 0.1-1.5 Mbit | OK | Table 1; Table 4 sizes 0.51-1.31 Mb |
| D_n 1-50 ms proportional to C_n | OK | Table 1 "Maximum task latency requirement 1-50 ms" (note: the word "Maximum"); p. 3 |
| R_0 = 10 Mbps | OK | Table 1 |
| Noise -100 dB = 1e-10 W | OK as literal | Table 1 says "dB", not dBm/dBW, and gives no bandwidth |
| Channel gain -50 dB | MISSING-NUANCE | Listed as a single number. The paper has no path-loss formula. It may be a fixed constant gain, or a reference gain. See A-1 |

### Section 2 (equations)
| Item | Verdict | Evidence |
|---|---|---|
| Task model C_n = C d_n, Poisson | OK | p. 3 |
| Eq. 1, A-4 | OK, nuance | Eq. 1 and Eq. 2 BOTH use D_n. Spec only fixes Eq. 1; Eq. 2 energy (z_n * D_n) needs the same treatment (spec writes z_n*C_n*c_cyc, which is consistent) |
| z_n = kappa F^2 | OK | p. 4. Predicted kappa: Fig. 4c and 4d have identical Local curves, so energy[mJ] = latency[ms] means kappa F^3 = 1 W, i.e. kappa = 1e-27 at F = 1 GHz |
| Eq. 3, 4 | OK | |
| Channel, A-1 | MISSING-NUANCE | Not in the paper; spec marks it as an assumption, but the stated justification "standard in UAV-MEC papers that Lin cites for parameters" is not supported: p. 10 says [7, 18, 19] are used only for "equipment and links ... CPU frequency, battery capacity, ... sensor power". Titles: [7] Nguyen et al., "Dependency Tasks Offloading and Communication Resource Allocation in Collaborative UAVs Networks: A Meta-Heuristic Approach"; [18] Muhammad & Hossain, "Deep-Reinforcement-Learning-Based Sustainable Energy Distribution for Wireless Communication"; [19] Lu et al., "Energy-Efficient Task Transfer in Wireless Computing Power Networks". None is a UAV path-loss reference we can verify offline |
| Eq. 5/6 SIC | MISSING-NUANCE | The paper's iota_jn is defined with the same h_nk for both users: "|h_nk|^2 p_j > g_n" (p. 4). Spec A-8 reads Eq. 5 as |h_jk|^2; the paper's own iota definition also uses h_nk. Both are typos of the same kind, so the spec's fix is reasonable, but the "strength of j" in iota should also use h_jk |
| Eq. 7 rate | MISSING-NUANCE | Paper writes B_m (not B_n), so the bandwidth allocation B_nm of the action does not appear in Eq. 7 at all. Spec's B_n = bw_frac * B_m is an assumption beyond A-3. Log base not stated (spec uses log2) |
| Eq. 8 e^0 | MISSING-NUANCE | "tau" here is the paper's delay-limit symbol (p. 3, 4) and ALSO the soft-update factor (p. 8). Spec A-11 does not mention that tau is the latency limit; a literal "tau * p_n" would be latency-limit x power |
| Eq. 9, 10 | OK | cumulative over T |
| Eq. 11, 12, A-12, A-13 | WRONG (A-12), OK (A-13 with nuance) | [31] is X. Diao, W. Yang, Y. L. Diao, "UAV-Relaying-Assisted Multi-Access Edge Computing With Multi-Antenna Base Station: Offloading and Scheduling Optimization", IEEE TVT 70(9), 2021. It is NOT Zeng et al. 2019 rotary-wing. Contents cannot be checked offline; the spec's claim "ref [31] -> Zeng rotary-wing" is unsupported. Metric only, low impact |
| Success conditions (Eq. 13, 14d, 15) | MISSING-NUANCE | p. 6 lists latency as "T^l + T^0 <= tau" (a sum) while Eq. 15 and (14d) use T^l and T^0 separately and also compare to tau. Eq. 15 uses ">= / <=" (fail at equality) while Eq. 7 uses ">= R_0" (success at equality) |
| Reward -1 / +3 | OK | Eq. 15. Fig. 4a is consistent with +3/-1 summed over 10 slots (see (b)) |

### Section 3 (agents)
| Item | Verdict | Evidence |
|---|---|---|
| CTDE, local actor, critic global | OK, nuance | Algorithm 1 gives each sensor its OWN critic Q_n(S, A) and own replay buffer. Spec does not say whether critics are per-agent or shared; the paper says per-agent |
| Observations | OK | p. 5 |
| Actor input = S_n + all S_k | OK | p. 7-8 |
| Actions (K+1, M, 2) | OK | p. 9 |
| POS vs FOS (A-14) | OK, nuance | Paper text says "partial offloading" and "each sensor transfers part of its task" (p. 8) yet the action has a binary target U_n in {0..K}, as the spec reads. FOS is "completely offloads". Conclusion about heads is supported |
| Hyperparameters | MISSING-NUANCE | batch 320, lr 3e-4 / 5e-5, buffer 2^17, gamma 0.9 are OK. "Hidden layer 128, ReLU6" is the actor only; critic is 64 then 32 hidden units (p. 9); actor has a shared 128 layer followed by three heads (K+1, M, 2) |

### Section 4 (targets)
| Item | Verdict | Evidence |
|---|---|---|
| POS_MADDPG_NOMA 85.63 / 7.48 / 5.68 | OK | p. 10, Table 2 |
| FOS_MADDPG_NOMA 60.88 / 8.11 / 4.195 | OK | |
| POS_MAPPO_NOMA 61.25% / 12.50 / 11.41 | WRONG (success rate) | Fig. 4b (~70%), Fig. 5a at N = 8 (~71%) and the "20% higher" text (85.63/1.20 = 71.4%) all say ~71%. Table 2's 61.25% is probably Local_Computing's value (see A-15). Latency 12.50 and energy 11.41 are consistent across Fig. 4 and 5 |
| Local_Computing | WRONG / resolved by evidence | Spec says Local is "not stated consistently" and suggests 71.4% or 45.8%. Fig. 4b shows Local ~62%, Fig. 5a shows ~61-62% for every N, and the text in ORDER (Local 39.8%, FOS 40.6%, MAPPO 20%) gives 85.63/1.398 = 61.25 (Local), 85.63/1.406 = 60.9 (FOS), 85.63/1.20 = 71.4 (MAPPO). The labels are NOT swapped; Table 2's MAPPO row is what looks wrong. At N = 10: 78/1.273 = 61.3 = Local. Percentages are RELATIVE in this paper (checked with 30.83%, 30.14%, 43.48%), so the 45.8% absolute-point reading is ruled out |
| noSIC rows | OK | Table 2. Latency of POS_MAPPO_noSIC 11.38 is lower than the NOMA value 12.50 (the paper's "increases the latency by 8.96%" is wrong). Note the same mislabelling doubt for POS_MAPPO_NOMA applies to its 61.25% base, so "43.48%" is computed from the Table 2 number |
| Local 18.76 ms and 18.76 mW | OK, nuance | Same number twice, and the curves are identical. Not an error; implies energy[mJ] = latency[ms] (kappa F^3 = 1 W). This lets C-2 be a closed form and a check on C-1 |
| Fig. 5 "success, latency, energy vs N" | WRONG | Fig. 5 has only two panels, (a) success rate and (b) average delay. There is no energy panel, though p. 11 says energy increases with N. Also Fig. 5 has no noSIC curves |
| Fig. 5 trends (FOS best at N = 2, 4; POS_MADDPG best for N > 4; > 78% at N = 10) | OK | Fig. 5a visual: FOS 95.7 vs POS 93.3 at N = 4; POS 78.0 at N = 10. At N = 2 both ~100% |
| Tables 3-4 | OK | Table 4 lists all offloaders as UAV1 but the text and Fig. 6 give UAV0 to S3, S5, S6, S7 |

### Section 5 (calibration)
| Item | Verdict | Evidence |
|---|---|---|
| C-1 c_cyc from Local latency 18.76 ms | OK, with constraint | Table 4 successes imply c_cyc <= ~40-42 cycles/bit (for 1 Mb = 1e6 bit, F = 1 GHz). Mean task 0.8 Mbit and 18.76 ms would give ~23 cycles/bit, inside that bound. Note the mean latency is over SUCCESSFUL tasks only (conditioned on T <= D) |
| C-2 kappa | OK | kappa F^3 = 1 W predicts kappa = 1e-27 exactly; useful as an independent check |

### Section 6 (assumptions A-1 .. A-16)
| ID | Verdict | Evidence / note |
|---|---|---|
| A-1 | MISSING-NUANCE | No channel model in the paper. Alternative reading: constant gain -50 dB for all links. Spec justification about cited references not supported (see above) |
| A-2 | MISSING-NUANCE | "Noise -100 dB" with no bandwidth; also says "dB". With p = 1 W, gain 1e-5 and noise 1e-10 W, SNR = 1e5 (50 dB); rate at 62.5 MHz or 250 MHz is >1 Gbps, so noise is irrelevant unless interference dominates. A noise-density reading would matter |
| A-3 | MISSING-NUANCE, possibly the wrong default | Evidence for B total: p. 3 ("B ... into M subcarriers ... B/M"), Fig. 4 caption "B_max = 250", p. 11 "fixed total bandwidth". Evidence for per carrier: Table 1 "Carrier bandwidth 250 MHZ", p. 10 "4 subcarriers of 250 MHz". Both readings give rates far above R_0 = 10 Mbps when SINR is high; the readings differ under interference. Keep the config switch; consider running the calibration/validation under both |
| A-4 | OK | Eq. 1 and Eq. 2 both use D_n (see above) |
| A-5 | OK (unsupported number) | Paper gives no C, lambda. The 0.1-1.5 Mbit range and Table 4 sizes (0.51-1.31, mean 0.96) are consistent. Mean of Table 4 sizes is 0.96, above the spec's 0.8 (n = 8, weak evidence) |
| A-6 | OK | Table 4 ratios 0.0189-0.0422 s/Mb verified |
| A-7 | OK | |
| A-8 | MISSING-NUANCE | See Eq. 5/6 above: iota_jn also uses h_nk |
| A-9 | OK, nuance | Supported indirectly by Fig. 4a (reward ~24 = 10 slots x 2.43) |
| A-10 | MISSING-NUANCE | Table 3/4 show start-of-episode power at 32.5-97.5% (sensors) and 70-72% (UAVs): initial charge is a fraction, not full. Spec says "recharged to E_max every cycle" and "sampled per episode"; Fig. 6 shows 55%-97.5% remaining |
| A-11 | MISSING-NUANCE | tau symbol clash (see Eq. 8) |
| A-12 | WRONG | [31] is Diao, Yang, Diao (2021), not Zeng et al. 2019 |
| A-13 | MISSING-NUANCE | Paper states that trajectory is jointly optimized (abstract, p. 5 "four parameters ... flight trajectory", Alg. 1 initializes UAV position), but no trajectory action exists. Table 3 shows UAV positions at integer metres (77,87), (30,53), supporting random initial positions. Hover default is fine; "optimized trajectory" cannot be reproduced and should be called out in the report as a limitation |
| A-14 | OK | |
| A-15 | WRONG | Resolved: Local ~61.25% (see Section 4 audit). The spec's "labels swapped" hypothesis and Local = 71.4% / 45.8% are both refuted by Fig. 4b / 5a. Table 2's POS_MAPPO_NOMA 61.25% is the suspect entry; the figures suggest ~71% |
| A-16 | OK | d_s not stated |

## Reference list entries relevant to the spec
- [7] L. Nguyen, Y. Tun, T. Dang, Y. Park, Z. Han, C. Hong, "Dependency Tasks Offloading and Communication Resource Allocation in Collaborative UAVs Networks: A Meta-Heuristic Approach," IEEE IoT-J 10 (2023): 9062-9076.
- [18] G. Muhammad, M. Hossain, "Deep-Reinforcement-Learning-Based Sustainable Energy Distribution for Wireless Communication," IEEE Wireless Commun. 28(6) (2021): 42-48.
- [19] Y. Lu, B. Ai, Z. Zhong, Y. Zhang, "Energy-Efficient Task Transfer in Wireless Computing Power Networks," IEEE IoT-J 10 (2022): 9353-9365.
- [31] X. Diao, W. Yang, Y. L. Diao, "UAV-Relaying-Assisted Multi-Access Edge Computing With Multi-Antenna Base Station: Offloading and Scheduling Optimization," IEEE TVT 70(9) (2021): 9495-9509.
- [36] Xie et al., "Resource Allocation in Uplink NOMA Systems: A Hybrid-Decision-Based Multi-Agent Deep Reinforcement Learning Approach," IEEE TVT 72(12) (2023): 16760-16765 (cited for the hybrid discrete/continuous action idea).

## (d) New ambiguities (proposed A-23 onward)

> Renumbered by the main agent on 2026-11-13: these were proposed as A-17..A-25, but A-17..A-22 were already taken by W1's items, so they are A-23..A-31 here. Status: proposed, not yet in the spec or code. A-27 adopted as A-34 (T-3), A-28 as A-35, A-30 as A-36 on 2026-11-13; A-23 deferred to R3; A-24/A-29 not adopted (HUMAN did not approve test T-5).


| ID | Ambiguity | Evidence | Suggested assumption |
|---|---|---|---|
| A-23 | Training schedule not stated: number of training episodes, steps per episode L, gradient updates per step, optimizer, soft-update tau, exploration noise, Gumbel-Softmax temperature | Algorithm 1 has "episode ep = 1 to c", "step t = 1 to L"; none given. Fig. 4 x axis ends at ~9800-10000 | 10,000 episodes (read from Fig. 4 axis), L = T = 10 slots, one update per step, Adam, tau_soft = 0.01, Gaussian noise sigma 0.1 decaying (or none for the discrete head via Gumbel sampling), Gumbel temperature 1.0 with straight-through hard sampling. All declared as untuned defaults, chosen before looking at validation results |
| A-24 | Share of a carrier's bandwidth: Table 4 shows sensors on the same carrier with bandwidth fractions that sum below 100% (MU3 47.8% + MU5 45.3% on carrier 2; MU6 30.8% + MU7 42.2% on carrier 1) and the UAV state has "available bandwidth B_k per carrier" | Table 3, 4; Section 4.1 | Each sensor's B_n = bw_frac * B_m (spec A-3), no depletion of the shared carrier; the sensor observes B_k as 1.0 (full) |
| A-25 | Reward value shown in Fig. 4a is a per-episode sum over 10 slots, while the text says "per time slot" | (b) point 2 | Report reward as the sum over the episode when comparing to Fig. 4a (mean per slot = this / 10) |
| A-26 | Symbol tau means latency limit, soft-update factor, and appears in e^0 = tau p_n U_n | pp. 3, 4, 8 | Keep spec A-11 and state that the delay limit is D_n |
| A-27 | Meaning of the metric "average energy (mW)": per task or per second | Fig. 4d Local curve identical to latency curve | Energy per task in mJ with one task per 1 s slot (so mJ per slot = mW), kappa = 1e-27 as a check on C-2 |
| A-28 | Table 2 POS_MAPPO_NOMA success rate (61.25%) disagrees with Figs. 4b / 5a (~70-71%) and Fig. 5a FOS at N = 8 (~81%) disagrees with Table 2 / Fig. 4b (60.88%) | (b) | Use Fig. 4 / Table 2 as the primary N = 8 targets for POS_MADDPG, FOS and Local; treat MAPPO as ~71% (figure + text) and flag as inconsistent in the paper; ranking-based gate unchanged. Success-rate FOS vs MAPPO ordering at N = 8 is paper-inconsistent and must not be a gate item |
| A-29 | Same-carrier sensors under SIC: whether UAV-to-sensor assignment counts the UAV's available bandwidth B_k as a state that changes during the episode | Table 3 ("100%") | Constant 1.0 |
| A-30 | Whether Local_Computing (and FOS) success statistic in Fig. 4b / 5a is measured under the same task and deadline stream as the learners | not stated | Same seeded stream for all methods (paired evaluation) |
| A-31 | Fig. 6 / Table 4 give inconsistent UAV assignment (text and figure: UAV0 and UAV1; Table 4: all UAV1) | p. 12-13 | Ignore the example for assignment logic; use only the sensor sizes, deadlines and positions as realistic ranges |
