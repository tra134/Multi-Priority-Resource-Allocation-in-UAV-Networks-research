"""S-3 sensitivity variants (docs/lin2025_spec.md Section 11, "S-3 detailed registration").

Each variant changes ONE reading of Lin et al. 2025 and is selected only through
LinMECEnv(..., s3_variant=...). The default (None) is the v1.0 environment, untouched.
These readings are measured, never adopted as the paper's model. Every variant line below is
marked with its ID (N1-seq, N1-prop, N1-eq, A1-const, A8-lit, A3-total, A2-dBm, A2-psd, UAV-cb).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import yaml

from uavmec.config import LinConfig
from uavmec.env.noma import noma_links

_REPO_ROOT = Path(__file__).resolve().parents[3]
BW_RULES = ("v0", "seq", "prop", "eq")
BW_SPLITS = ("table", "total_split")


@dataclass(frozen=True)
class S3Variant:
    """One named variant. All fields at their default give V0 (= v1.0 behaviour).

    bw_rule: "v0" B_n = f_n B | "seq" N1-seq | "prop" N1-prop | "eq" N1-eq.
    gain_const: A1-const, a constant |h|^2 for every link (None = distance law, A-1).
    a8_literal: A8-lit, literal Eq. 5 (own-gain interference, order by transmit power).
    bw_split: "table" B = 250 MHz (A-3) | "total_split" A3-total, B = 250 MHz / M.
    noise_w: A2-dBm / A2-psd, noise power override (None = cfg.noise_w).
    uav_compute_bound: UAV-cb, add C_n c_cyc / (f_uav / n_k) to offloaded latency.
    """

    name: str
    bw_rule: str = "v0"
    gain_const: float | None = None
    a8_literal: bool = False
    bw_split: str = "table"
    noise_w: float | None = None
    uav_compute_bound: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if self.bw_rule not in BW_RULES:
            raise ValueError(f"bw_rule must be one of {BW_RULES}, got {self.bw_rule!r}")
        if self.bw_split not in BW_SPLITS:
            raise ValueError(f"bw_split must be one of {BW_SPLITS}, got {self.bw_split!r}")
        if self.gain_const is not None and not self.gain_const > 0:
            raise ValueError("gain_const must be > 0")
        if self.noise_w is not None and not self.noise_w > 0:
            raise ValueError("noise_w must be > 0")

    @classmethod
    def from_dict(cls, name: str, raw: dict | None) -> S3Variant:
        raw = dict(raw or {})
        known = {f.name for f in fields(cls)} - {"name"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown variant keys for {name!r}: {sorted(unknown)}")
        return cls(name=name, **raw)

    def subcarrier_bw(self, cfg: LinConfig) -> float:
        """B in Hz. A3-total: 250 MHz / M = 62.5 MHz; otherwise the v1.0 value (A-3)."""
        if self.bw_split == "total_split":  # A3-total
            return cfg.bandwidth_hz / cfg.n_subcarriers
        return cfg.subcarrier_bw_hz

    def noise(self, cfg: LinConfig) -> float:
        """Noise power in W. A2-dBm 1e-13, A2-psd 1e-12; otherwise cfg.noise_w."""
        return cfg.noise_w if self.noise_w is None else self.noise_w  # A2-dBm, A2-psd

    def channel_gain(self, gain: np.ndarray) -> np.ndarray:
        """A1-const: replace every link gain by the constant; otherwise return gain unchanged."""
        if self.gain_const is None:
            return gain
        return np.full_like(gain, self.gain_const)  # A1-const


def as_variant(v: S3Variant | dict | None) -> S3Variant | None:
    if v is None or isinstance(v, S3Variant):
        return v
    return S3Variant.from_dict(str(v.get("name", "custom")), {k: x for k, x in v.items() if k != "name"})


def load_s3_variants(path: str | Path = "configs/s3_variants.yaml") -> dict[str, S3Variant]:
    """Load the named variant blocks (in file order) from the S-3 config."""
    p = Path(path)
    if not p.exists() and not p.is_absolute() and (_REPO_ROOT / p).exists():
        p = _REPO_ROOT / p
    with open(p) as f:
        raw = yaml.safe_load(f) or {}
    return {name: S3Variant.from_dict(name, block) for name, block in raw["variants"].items()}


# ------------------------------------------------------------------ N-1
def allocate_bandwidth(
    rule: str,
    sensor_idx: np.ndarray,
    uav_idx: np.ndarray,
    sub_idx: np.ndarray,
    bw_frac: np.ndarray,
    bandwidth_hz: float,
) -> np.ndarray:
    """Bandwidth B_n (Hz) of every offloading sensor under a shared-carrier rule.

    A group (k, m) = sensors offloading to UAV k on subcarrier m; f_n = bw_frac, B = bandwidth_hz.
    "v0":   B_n = f_n B (v1.0).
    "seq":  N1-seq, sensors in increasing sensor index, remaining R starts at B,
            B_n = f_n R, then R <- R - B_n.
    "prop": N1-prop, B_n = f_n B / max(1, sum of f in the group).
    "eq":   N1-eq, B_n = B / |group| (ignores f_n).
    """
    sensor_idx = np.asarray(sensor_idx)
    uav_idx = np.asarray(uav_idx)
    sub_idx = np.asarray(sub_idx)
    f = np.asarray(bw_frac, dtype=float)
    if rule == "v0":
        return f * bandwidth_hz
    out = np.zeros(f.size)
    for k in np.unique(uav_idx):
        for m in np.unique(sub_idx[uav_idx == k]):
            idx = np.where((uav_idx == k) & (sub_idx == m))[0]
            if rule == "seq":  # N1-seq
                remaining = bandwidth_hz
                for i in idx[np.argsort(sensor_idx[idx], kind="stable")]:
                    out[i] = f[i] * remaining
                    remaining -= out[i]
            elif rule == "prop":  # N1-prop
                out[idx] = f[idx] * bandwidth_hz / max(1.0, f[idx].sum())
            elif rule == "eq":  # N1-eq
                out[idx] = bandwidth_hz / idx.size
            else:
                raise ValueError(f"unknown bandwidth rule {rule!r}")
    return out


# ------------------------------------------------------------------ A-8 literal
def noma_group_a8lit(
    gain_own: np.ndarray,
    power_w: np.ndarray,
    bandwidth_hz: np.ndarray,
    noise_w: float,
    rate_req_bps: float,
    sic: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A8-lit: literal Eq. 5 for users sharing ONE (UAV, subcarrier). Returns (sinr, rate, decoded).

    Interference of j on n is h_n p_j (gain of the decoded sensor n times the power of j);
    chi_jn = 1 iff p_j > p_n (equal powers: never cancelled); d_j = 1 iff R_j >= R_0 (A-7).
    Sensors are processed in decreasing p: I_n = sum_{j != n} (1 - chi_jn d_j) h_n p_j,
    gamma_n = h_n p_n / (I_n + sigma^2), R_n = B_n log2(1 + gamma_n).
    sic=False: I_n = sum_{j != n} h_n p_j (nothing cancelled).
    """
    h = np.asarray(gain_own, dtype=float)
    p = np.asarray(power_w, dtype=float)
    bw = np.asarray(bandwidth_hz, dtype=float)
    n = h.size
    sinr = np.zeros(n)
    rate = np.zeros(n)
    decoded = np.zeros(n, dtype=bool)
    order = np.argsort(-p, kind="stable")  # decreasing transmit power
    for i in order:
        others = np.arange(n) != i
        keep = np.ones(n)  # (1 - chi_jn d_j)
        if sic:
            chi = p > p[i]  # strictly larger power; these are processed before i
            keep = 1.0 - (chi & decoded).astype(float)
        interference = h[i] * float((keep * p)[others].sum())
        sinr[i] = h[i] * p[i] / (interference + noise_w)
        rate[i] = bw[i] * np.log2(1.0 + sinr[i])
        decoded[i] = rate[i] >= rate_req_bps
    return sinr, rate, decoded


# ------------------------------------------------------------------ link evaluation
def s3_offload_links(
    v: S3Variant,
    cfg: LinConfig,
    sic: bool,
    sensor_idx: np.ndarray,
    uav_idx: np.ndarray,
    sub_idx: np.ndarray,
    gain_own: np.ndarray,
    power_w: np.ndarray,
    bw_frac: np.ndarray,
    bits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SINR, rate, decoded flag and extra UAV compute latency (s) of the offloading sensors.

    With every field of v at its default this is exactly the v1.0 computation (V0).
    """
    bw = allocate_bandwidth(  # N1-seq / N1-prop / N1-eq (V0: f_n B); A3-total via subcarrier_bw
        v.bw_rule, sensor_idx, uav_idx, sub_idx, bw_frac, v.subcarrier_bw(cfg)
    )
    noise = v.noise(cfg)  # A2-dBm / A2-psd
    if not v.a8_literal:
        rx = gain_own * power_w
        sinr, rate, decoded = noma_links(uav_idx, sub_idx, rx, bw, noise, cfg.rate_req_bps, sic)
    else:  # A8-lit
        n = uav_idx.size
        sinr, rate, decoded = np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool)
        for k in np.unique(uav_idx):
            for m in np.unique(sub_idx[uav_idx == k]):
                idx = np.where((uav_idx == k) & (sub_idx == m))[0]
                s, r, d = noma_group_a8lit(
                    gain_own[idx], power_w[idx], bw[idx], noise, cfg.rate_req_bps, sic
                )
                sinr[idx], rate[idx], decoded[idx] = s, r, d
    extra = np.zeros(uav_idx.size)
    if v.uav_compute_bound:  # UAV-cb: C_n c_cyc / (f_uav / n_k), equal CPU sharing
        n_k = np.bincount(uav_idx)[uav_idx]
        extra = np.asarray(bits, dtype=float) * cfg.c_cyc / (cfg.f_uav_hz / n_k)
    return sinr, rate, decoded, extra
