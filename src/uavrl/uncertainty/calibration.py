"""Uncertainty calibration for the ensemble-critic OOD trigger.

U_t(s,a) = standard deviation of Q(s,a) across the N critics, evaluated at the
actor's action.  Under in-distribution (nominal) conditions the critics agree
(low U); under OOD they disagree (high U).  We calibrate tau_unc as a
percentile of U_t measured on in-distribution validation episodes.

Ground-truth OOD labels (env.info["ood_flag"]) are used for scoring ONLY --
never fed to the agent or the LLM prompt.
"""

from __future__ import annotations

import numpy as np


def uncertainty_at(agent, obs, action):
    """Ensemble std (and mean) of Q(s,a)."""
    return agent.uncertainty(obs, action)


def collect_trajectory(env, agent, deterministic=True):
    """Run one episode; return lists of (U_t, meanQ, ood_flag, delay_emg, slot)."""
    obs, _ = env.reset()
    rows = []
    done = False
    while not done:
        action = agent.select_action(obs, deterministic=deterministic)
        u, mean_q = agent.uncertainty(obs, action)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(
            {
                "slot": int(info["slot"]),
                "U": float(u),
                "meanQ": float(mean_q),
                "ood_flag": bool(info["ood_flag"]),
                "delay_emg_s": float(info["delay_s"][0]),
                "queue_emg_bits": float(info["queue_bits"][0]),
                "reward": float(reward),
            }
        )
        done = terminated or truncated
    return rows


def normalize_U(rows, c=1.0):
    """Normalized uncertainty U / (|mean Q| + c). Returns list aligned with rows."""
    return [
        r["U"] / (abs(r["meanQ"]) + c) for r in rows
    ]


def percentile_U(rows, q=99.0, normalized=False):
    """q-th percentile of U over a collection of trajectories (list of row-lists)."""
    vals = []
    for traj in rows:
        for r in traj:
            vals.append(r["U_norm"] if normalized else r["U"])
    return float(np.percentile(vals, q))


def ood_metrics(rows, tau, D_th_s=15.0 / 1000.0, normalized=False):
    """Precision/recall/detection-lead-time for one trajectory vs threshold tau.

    Returns dict with per-slot arrays and scalar summary metrics.
    Lead time = first trigger slot - first Emergency-violation slot (negative
    = warned before damage).
    """
    us = np.array([r["U_norm"] if normalized else r["U"] for r in rows])
    ood = np.array([r["ood_flag"] for r in rows], dtype=bool)
    viol = np.array([r["delay_emg_s"] > D_th_s for r in rows])
    trigger = us > tau
    first_ood = int(np.argmax(ood)) if ood.any() else None
    first_trigger = int(np.argmax(trigger)) if trigger.any() else None
    first_viol = int(np.argmax(viol)) if viol.any() else None
    lead_time = None
    if first_trigger is not None and first_viol is not None:
        lead_time = first_trigger - first_viol
    return {
        "U": us,
        "ood": ood,
        "violation": viol,
        "trigger": trigger,
        "first_ood": first_ood,
        "first_trigger": first_trigger,
        "first_violation": first_viol,
        "lead_time": lead_time,
        "precision": float(np.mean(ood[trigger])) if trigger.any() else 0.0,
        "recall": float(np.mean(trigger[ood])) if ood.any() else 0.0,
    }


def auroc(scores, labels):
    """Rank-based AUROC (scores higher => more likely positive)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    auroc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auroc)

