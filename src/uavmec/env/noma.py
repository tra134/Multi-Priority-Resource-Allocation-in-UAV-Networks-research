"""Uplink NOMA with SIC (Eq. 5-7, assumptions A-3, A-7, A-8)."""

from __future__ import annotations

import numpy as np


def noma_group(
    rx_power: np.ndarray,
    bandwidth_hz: np.ndarray,
    noise_w: float,
    rate_req_bps: float,
    sic: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SINR, rate and decode flag for users sharing ONE (UAV, subcarrier).

    rx_power: g_n = |h_nk|^2 * p_n received at the UAV (W), shape (n,).
    bandwidth_hz: B_n of each user, shape (n,).
    Returns (sinr, rate_bps, decoded), each shape (n,).

    SIC (Eq. 5): users are decoded strongest first. I_n = sum_{j != n} (1 - iota_jn * d_j) g_j,
    where iota_jn = 1 if j is decoded before n (g_j > g_n; ties broken by index) and d_j = 1
    iff j was decoded successfully, i.e. R_j >= R_0 (A-7, A-8). So weaker users always stay
    as interference, and a stronger user that failed to decode also stays as interference.
    noSIC: I_n = sum_{j != n} g_j (nothing is cancelled); decoded is still R_n >= R_0.
    Eq. 6: gamma_n = g_n / (I_n + sigma^2). Eq. 7: R_n = B_n * log2(1 + gamma_n).
    """
    g = np.asarray(rx_power, dtype=float)
    bw = np.asarray(bandwidth_hz, dtype=float)
    n = g.size
    sinr = np.zeros(n)
    rate = np.zeros(n)
    decoded = np.zeros(n, dtype=bool)
    if n == 0:
        return sinr, rate, decoded
    if not sic:
        interference = g.sum() - g
        sinr = g / (interference + noise_w)
        rate = bw * np.log2(1.0 + sinr)
        return sinr, rate, rate >= rate_req_bps
    order = np.argsort(-g, kind="stable")  # strongest first
    for pos, i in enumerate(order):
        # users decoded before i that failed still interfere; all later users interfere
        failed_before = [j for j in order[:pos] if not decoded[j]]
        interference = g[order[pos + 1 :]].sum() + g[failed_before].sum()
        sinr[i] = g[i] / (interference + noise_w)
        rate[i] = bw[i] * np.log2(1.0 + sinr[i])
        decoded[i] = rate[i] >= rate_req_bps
    return sinr, rate, decoded


def noma_links(
    uav_idx: np.ndarray,
    sub_idx: np.ndarray,
    rx_power: np.ndarray,
    bandwidth_hz: np.ndarray,
    noise_w: float,
    rate_req_bps: float,
    sic: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply noma_group to every (UAV, subcarrier) group among the offloading users.

    All arrays have shape (n_offloaders,); rx_power[i] is the strength of user i at its own UAV.
    """
    uav_idx = np.asarray(uav_idx)
    sub_idx = np.asarray(sub_idx)
    n = uav_idx.size
    sinr = np.zeros(n)
    rate = np.zeros(n)
    decoded = np.zeros(n, dtype=bool)
    for k in np.unique(uav_idx):
        for m in np.unique(sub_idx[uav_idx == k]):
            idx = np.where((uav_idx == k) & (sub_idx == m))[0]
            s, r, d = noma_group(rx_power[idx], bandwidth_hz[idx], noise_w, rate_req_bps, sic)
            sinr[idx], rate[idx], decoded[idx] = s, r, d
    return sinr, rate, decoded
