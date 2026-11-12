"""Regenerate every figure and table from stored per-episode CSVs (Stage 8).

Reads one or more run_grid.py outputs (a metrics.csv file or a directory that
holds one) and writes PNG + PDF figures and CSV + Markdown tables to --out.
It never writes to results/ unless you point --out there yourself.

Outputs: violation_bars.{png,pdf}, throughput_vs_violation.{png,pdf},
holm_table.{csv,md}.  Rows with client=mock are labelled "mock - pipeline
check" in every table and figure; they are never evidence that an LLM helps.

Example:
    venv/bin/python scripts/make_figures.py --in results/final --out /tmp/figs
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uavrl.eval.stats import (  # noqa: E402
    MAIN_BASELINES,
    MAIN_METHOD,
    MAIN_SCENARIOS,
    MOCK_NOTE,
    comparison_table,
    hierarchical_bootstrap_ci,
)

VIOLATION = "sos_violation_ood"


def load(inputs):
    frames = []
    for p in inputs:
        p = Path(p)
        f = p / "metrics.csv" if p.is_dir() else p
        if not f.exists():
            raise SystemExit(f"input not found: {f}")
        frames.append(pd.read_csv(f))
    return pd.concat(frames, ignore_index=True)


def save_fig(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def label_for(df, method):
    sub = df[df["method"] == method]
    name = sub["method_label"].iloc[0] if "method_label" in sub else method
    if "client" in sub and (sub["client"] == "mock").any():
        name += " [mock - pipeline check]"
    return name


def violation_bars(df, out, n_boot, scenarios):
    methods = list(dict.fromkeys(df["method"]))
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / max(len(methods), 1)
    rows = []
    for i, m in enumerate(methods):
        xs, ys, lo, hi = [], [], [], []
        for j, sc in enumerate(scenarios):
            sub = df[(df["method"] == m) & (df["scenario"] == sc)]
            if sub.empty:
                continue
            r = hierarchical_bootstrap_ci(sub, VIOLATION, n_boot=n_boot)
            xs.append(j + i * width)
            ys.append(100 * r["mean"])
            lo.append(100 * (r["mean"] - r["ci_low"]))
            hi.append(100 * (r["ci_high"] - r["mean"]))
            rows.append({"method": m, "scenario": sc, **r})
        ax.bar(xs, ys, width, yerr=[lo, hi], capsize=2, label=label_for(df, m))
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(scenarios))])
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("In-window Emergency violation (%)")
    ax.set_title("Emergency delay violation, 95% hierarchical bootstrap CI")
    ax.legend(fontsize=6)
    save_fig(fig, out, "violation_bars")
    pd.DataFrame(rows).to_csv(out / "violation_bars.csv", index=False)


def throughput_scatter(df, out):
    need = {"delivered_imp_ood_mbps", "delivered_norm_ood_mbps"}
    if not need <= set(df.columns):
        return
    d = df.assign(thr=df["delivered_imp_ood_mbps"] + df["delivered_norm_ood_mbps"])
    g = d.groupby(["method", "scenario"]).agg(
        viol=(VIOLATION, "mean"), thr=("thr", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m, s in g.groupby("method"):
        ax.scatter(100 * s["viol"], s["thr"], label=label_for(df, m))
    ax.set_xlabel("In-window Emergency violation (%)")
    ax.set_ylabel("Important + Normal throughput in window (Mbps)")
    ax.legend(fontsize=6)
    save_fig(fig, out, "throughput_vs_violation")


def to_markdown(tab):
    cols = list(tab.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in tab.iterrows():
        lines.append("| " + " | ".join(
            f"{v:.4g}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inputs", action="append", required=True,
                    help="metrics.csv or a directory holding it; repeatable")
    ap.add_argument("--out", required=True,
                    help="output directory (required; never defaults to results/)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    ap.add_argument("--main", default=MAIN_METHOD)
    ap.add_argument("--baselines", default=",".join(MAIN_BASELINES))
    ap.add_argument("--scenarios", default=",".join(MAIN_SCENARIOS))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(args.inputs)
    scenarios = tuple(args.scenarios.split(","))
    violation_bars(df, out, args.n_boot, scenarios)
    throughput_scatter(df, out)
    tab = comparison_table(df, VIOLATION, main=args.main,
                           baselines=tuple(args.baselines.split(",")),
                           scenarios=scenarios, n_boot=args.n_boot, seed=args.seed)
    tab.to_csv(out / "holm_table.csv", index=False)
    md = "# Pre-declared comparisons (Holm-corrected)\n\n"
    if (tab["note"] != "").any() if len(tab) else False:
        md += f"> {MOCK_NOTE}\n\n"
    md += to_markdown(tab)
    (out / "holm_table.md").write_text(md, encoding="utf-8")
    print(f"wrote figures and tables to {out}")


if __name__ == "__main__":
    main()
