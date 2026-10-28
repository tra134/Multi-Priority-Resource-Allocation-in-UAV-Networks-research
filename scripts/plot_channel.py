"""Phase 1 plots: channel and traffic model sanity checks.

Saves figures to results/phase1/:

  f1_p_los_vs_elevation.png   - Al-Hourani P_LoS vs elevation angle
  f2_path_loss_vs_distance.png - mean path loss vs ground radius (LoS/NLoS/mean)
  f3_snr_sample.png            - 2 s SNR sample with a +15 dB shadow window
  f4_queue_overload.png        - queue growth under constant overload
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uavrl.env.channel import Channel, noise_power_w, p_los, rate
from uavrl.env.traffic import Traffic

OUT = "results/phase1"
SLOT_MS = 5.0
DT = SLOT_MS / 1000.0


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def plot_p_los():
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    theta = np.linspace(0, 90, 400)
    ax.plot(theta, p_los(theta), lw=2, color="#1f77b4")
    ax.set_xlabel("Elevation angle [deg]")
    ax.set_ylabel("P_LoS")
    ax.set_title("Al-Hourani P_LoS vs elevation (urban, a=9.61, b=0.16)")
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    _save(fig, "f1_p_los_vs_elevation.png")


def plot_path_loss():
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    radius = np.linspace(10, 600, 400)
    ch = Channel(n_classes=1, H=100.0, fc=2e9, rng=np.random.default_rng(0))
    pl_mean = ch.path_loss(radius, shadow_db=0.0)
    # LoS-only and NLoS-only references at the same radii
    p = p_los(ch.elevation_angle_deg(radius))
    fspl = 20.0 * np.log10(4.0 * np.pi * ch.distance_m(radius) / 2.99792458e8)
    ax.plot(radius, pl_mean, lw=2, label="Mean PL (P-weighted)", color="#1f77b4")
    ax.plot(radius, fspl + 1.0, "--", lw=1, label="LoS only", color="#2ca02c", alpha=0.7)
    ax.plot(radius, fspl + 20.0, "--", lw=1, label="NLoS only", color="#d62728", alpha=0.7)
    ax.set_xlabel("Ground radius [m]")
    ax.set_ylabel("Path loss [dB]")
    ax.set_title("Mean A2G path loss vs distance (H = 100 m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, "f2_path_loss_vs_distance.png")


def plot_snr_sample():
    """2 s (400 slots) of received SNR per class with a +15 dB shadow window."""
    rng = np.random.default_rng(7)
    ch = Channel(n_classes=3, H=100.0, fc=2e9, rng=rng)
    radii = np.array([100.0, 250.0, 400.0])  # Emergency, Important, Normal
    n_slots = int(2.0 / DT)  # 400 slots
    shadow = np.zeros(n_slots)
    shadow[150:350] = 15.0  # +15 dB shadow window

    snr_db = np.zeros((n_slots, 3))
    n0 = noise_power_w(1.0, -174.0, 7.0)
    for t in range(n_slots):
        gain = ch.update(radii, shadow_db=shadow[t])
        # received power for 1 W over 1 MHz reference
        pr = 1.0 * gain
        snr_db[t] = 10.0 * np.log10(np.maximum(pr / (n0 * 1e6), 1e-12))

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for k, label in enumerate(["Emergency", "Important", "Normal"]):
        ax.plot(np.arange(n_slots) * DT, snr_db[:, k], label=label)
    ax.axvspan(150 * DT, 350 * DT, color="orange", alpha=0.2, label="+15 dB shadow")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("SNR [dB] (1 W, 1 MHz ref)")
    ax.set_title("Received SNR over 2 s with a +15 dB shadow window")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    _save(fig, "f3_snr_sample.png")


def plot_queue_overload():
    """Queue growth under constant overload (arrivals exceed service)."""
    rng = np.random.default_rng(11)
    tr = Traffic([0.5, 8.0, 6.0], packet_bytes=200, rng=rng)
    # Fixed service rate below the arrival rate -> queue grows.
    service = np.array([0.2, 4.0, 3.0])  # Mbps
    n_slots = 400
    Q = np.zeros(3)
    q_hist = np.zeros((n_slots, 3))
    for t in range(n_slots):
        arrivals = tr.arrivals_bits(DT)  # bits
        served = service * 1e6 * DT  # bits
        Q = np.maximum(0.0, Q - served) + arrivals
        q_hist[t] = Q / 1e6  # Mbit

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for k, label in enumerate(["Emergency", "Important", "Normal"]):
        ax.plot(np.arange(n_slots) * DT, q_hist[:, k], label=label)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Queue length [Mbit]")
    ax.set_title("Queue growth under constant overload (service < arrivals)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, "f4_queue_overload.png")


def main():
    plot_p_los()
    plot_path_loss()
    plot_snr_sample()
    plot_queue_overload()
    print("Phase 1 plots complete in", OUT)


if __name__ == "__main__":
    main()
