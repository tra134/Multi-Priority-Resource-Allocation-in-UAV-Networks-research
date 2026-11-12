"""Statistics for the final comparison (plan.md Stage 8, D14).

The training seed is the statistical unit: episodes inside one seed share an
actor and are not independent.  Every interval therefore comes from a
hierarchical bootstrap that resamples training seeds with replacement and then
episodes with replacement inside each chosen seed.  Column names follow
``scripts/run_grid.py`` (``method``, ``scenario``, ``seed``, ``episode``,
``eval_seed``, ``client``).
"""

import numpy as np
import pandas as pd

MAIN_METHOD = "m12_proposed"
MAIN_BASELINES = ("m3_pure_sac", "m4_redq_sac", "m7_rule_shield", "m13_fixed_reserve")
MAIN_SCENARIOS = ("S1", "S2", "S4")
PAIR_KEYS = ("seed", "scenario", "episode", "eval_seed")
MOCK_NOTE = "mock - pipeline check, not evidence for the LLM"


def _seed_groups(df, cols):
    """List of (episodes x len(cols)) arrays, one per training seed."""
    return [g[list(cols)].to_numpy(dtype=float)
            for _, g in df.groupby("seed", sort=True)]


def _boot_means(groups, n_boot, rng):
    """Bootstrap seed-mean vectors, shape (n_boot, k)."""
    n_seeds = len(groups)
    k = groups[0].shape[1]
    out = np.empty((n_boot, k))
    for b in range(n_boot):
        picks = rng.integers(0, n_seeds, size=n_seeds)
        means = np.empty((n_seeds, k))
        for j, s in enumerate(picks):
            g = groups[s]
            idx = rng.integers(0, len(g), size=len(g))
            means[j] = g[idx].mean(axis=0)
        out[b] = means.mean(axis=0)
    return out


def _point(groups):
    return np.mean([g.mean(axis=0) for g in groups], axis=0)


def hierarchical_bootstrap_ci(df, metric, n_boot=2000, seed=0, alpha=0.05):
    """Mean of ``metric`` (seed means averaged) with a percentile CI."""
    groups = _seed_groups(df, [metric])
    rng = np.random.default_rng(seed)
    boots = _boot_means(groups, n_boot, rng)[:, 0]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(_point(groups)[0]), "ci_low": float(lo),
            "ci_high": float(hi), "n_seeds": len(groups), "n_boot": n_boot}


def paired_comparison(df_a, df_b, metric, n_boot=2000, seed=0, alpha=0.05,
                      keys=PAIR_KEYS):
    """Compare method A with method B, matched on training seed, scenario and
    episode seed.  Returns diff = V_a - V_b with CI, a two-sided bootstrap
    p-value, and rel_improvement = (V_b - V_a) / V_b with CI (RQ4)."""
    keys = [k for k in keys if k in df_a.columns and k in df_b.columns]
    merged = df_a[keys + [metric]].merge(
        df_b[keys + [metric]], on=keys, suffixes=("_a", "_b"))
    n_pairs = len(merged)
    if n_pairs == 0:
        raise ValueError("no matched (seed, scenario, episode) pairs")
    merged = merged.rename(columns={metric + "_a": "va", metric + "_b": "vb"})
    groups = _seed_groups(merged, ["va", "vb"])
    rng = np.random.default_rng(seed)
    boots = _boot_means(groups, n_boot, rng)
    d_boot = boots[:, 0] - boots[:, 1]
    va, vb = _point(groups)
    diff = va - vb
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_boot = (boots[:, 1] - boots[:, 0]) / boots[:, 1]
    rel_boot = rel_boot[np.isfinite(rel_boot)]
    rel = float((vb - va) / vb) if vb != 0 else float("nan")
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    d_lo, d_hi = np.percentile(d_boot, q)
    r_lo, r_hi = (np.percentile(rel_boot, q) if len(rel_boot)
                  else (float("nan"), float("nan")))
    p = 2 * min(np.mean(d_boot <= 0), np.mean(d_boot >= 0))
    return {"mean_a": float(va), "mean_b": float(vb), "diff": float(diff),
            "diff_ci_low": float(d_lo), "diff_ci_high": float(d_hi),
            "p_value": float(min(1.0, p)), "rel_improvement": rel,
            "rel_ci_low": float(r_lo), "rel_ci_high": float(r_hi),
            "n_pairs": n_pairs, "n_seeds": len(groups), "n_boot": n_boot}


def holm_bonferroni(p_values, alpha=0.05):
    """Holm-Bonferroni step-down.  Returns (adjusted p, reject flags)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj, adj <= alpha


def comparison_table(df, metric="sos_violation_ood", main=MAIN_METHOD,
                     baselines=MAIN_BASELINES, scenarios=MAIN_SCENARIOS,
                     n_boot=2000, seed=0, alpha=0.05):
    """Pre-declared comparisons (main vs each baseline, per scenario) with Holm
    correction over all of them.  Rows where either side was produced with a
    mock client are kept but labelled in ``note``."""
    rows = []
    for sc in scenarios:
        for base in baselines:
            a = df[(df["method"] == main) & (df["scenario"] == sc)]
            b = df[(df["method"] == base) & (df["scenario"] == sc)]
            if a.empty or b.empty:
                continue
            res = paired_comparison(a, b, metric, n_boot=n_boot, seed=seed,
                                    alpha=alpha)
            mock = "client" in df.columns and (
                (a["client"] == "mock").any() or (b["client"] == "mock").any())
            rows.append({"scenario": sc, "main": main, "baseline": base,
                         "metric": metric, **res,
                         "note": MOCK_NOTE if mock else ""})
    tab = pd.DataFrame(rows)
    if tab.empty:
        return tab
    adj, rej = holm_bonferroni(tab["p_value"].to_numpy(), alpha)
    tab["p_holm"] = adj
    tab["reject_holm"] = rej
    return tab
