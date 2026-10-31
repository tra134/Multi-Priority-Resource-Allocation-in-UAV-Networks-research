"""Phase 8 smoke test for the real LLM arbiter (Groq / OpenAI-compatible).

Sends triggered situations from a surge scenario and a deep-fading scenario to
the configured model, validates the JSON against the project schema, scores the
diagnosis against the ground-truth scenario (for reporting only), and measures
wall-clock latency + token usage.

Usage
-----
    python scripts/llm_smoke_test.py --n 20
    python scripts/llm_smoke_test.py --n 20 --mock      # no network, CI check
    python scripts/llm_smoke_test.py --n 50 --write-latency-config

Outputs
-------
    results/phase8/smoke_test.json
    results/phase8/smoke_test.csv
    configs/llm_latency.yaml   (only with --write-latency-config)

Exit code is 0 when at least 95% of responses are schema-valid (plan.md Phase 8
acceptance criterion), 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uavrl.config import find_project_root, llm_settings, load_env
from uavrl.llm.cache import LLMCache
from uavrl.llm.client import MockLLMClient, RealLLMClient
from uavrl.llm.prompt import build_prompt

EXPECTED = {
    "surge": {"SOS_SURGE"},
    "shadow": {"DEEP_FADING"},
}


def make_features(scenario: str, i: int, rng: np.random.Generator) -> dict:
    """Create a plausible triggered situation (no ground-truth labels inside)."""
    slot = 600 + i * 37
    if scenario == "surge":
        q_emg = float(rng.uniform(8e5, 3.0e6))
        dq = float(rng.uniform(0.2e5, 2.0e5))
        csi = float(rng.uniform(0.3, 0.8))
        shadow = 0.0
    elif scenario == "shadow":
        q_emg = float(rng.uniform(2e5, 1.2e6))
        dq = float(rng.uniform(-1.0e5, 0.5e5))
        csi = float(rng.uniform(0.01, 0.08))
        shadow = float(rng.uniform(15.0, 22.0))
    else:  # nominal (not scored)
        q_emg = float(rng.uniform(1e3, 5e4))
        dq = float(rng.uniform(-2e4, 2e4))
        csi = float(rng.uniform(0.8, 1.4))
        shadow = 0.0

    return {
        "slot": slot,
        "trigger_reason": "queue" if scenario == "surge" else "uncertainty",
        "U_t": float(rng.uniform(0.05, 0.5)) if scenario == "surge" else float(rng.uniform(0.5, 2.0)),
        "csi_emergency_mean": csi,
        "csi_emergency_var": float(rng.uniform(1e-4, 1e-2)),
        "csi_emergency_slope": float(rng.uniform(-0.02, 0.02)),
        "csi_important_mean": float(rng.uniform(0.4, 1.2)),
        "csi_important_var": float(rng.uniform(1e-4, 1e-2)),
        "csi_important_slope": float(rng.uniform(-0.02, 0.02)),
        "csi_normal_mean": float(rng.uniform(0.5, 1.5)),
        "csi_normal_var": float(rng.uniform(1e-4, 1e-2)),
        "csi_normal_slope": float(rng.uniform(-0.02, 0.02)),
        "queue_emg_bits": q_emg,
        "queue_imp_bits": float(rng.uniform(1e5, 2e6)),
        "queue_norm_bits": float(rng.uniform(1e5, 2e6)),
        "dQ_emg": dq,
        "shadow_db": shadow,  # not in the prompt; used by MockLLM diagnosis only
    }


def fit_lognormal(latencies_ms):
    vals = [x for x in latencies_ms if x > 0.0]
    if len(vals) < 2:
        return None
    logs = np.log(vals)
    mu, sigma = float(logs.mean()), float(logs.std(ddof=1))
    return {
        "n": len(vals),
        "mu_log": mu,
        "sigma_log": sigma,
        "median_ms": float(math.exp(mu)),
        "mean_ms": float(np.mean(vals)),
        "p5_ms": float(np.percentile(vals, 5)),
        "p95_ms": float(np.percentile(vals, 95)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 8 real-LLM smoke test (JSON validity, diagnosis accuracy, latency)"
    )
    parser.add_argument("--config", default="configs/llm.yaml")
    parser.add_argument("--n", type=int, default=20, help="number of triggered situations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/phase8")
    parser.add_argument("--mock", action="store_true", help="use MockLLM (no network)")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--write-latency-config", action="store_true")
    parser.add_argument("--min-valid", type=float, default=0.95)
    args = parser.parse_args()

    load_env()
    cfg = llm_settings(args.config)
    root = find_project_root()

    cache_cfg = cfg.get("cache", {})
    cache = None
    if not args.no_cache and cache_cfg.get("enabled", True):
        cache = LLMCache(enabled=True, dir=cache_cfg.get("dir", "results/llm_cache"))

    if args.mock:
        rng = np.random.default_rng(args.seed)
        client = MockLLMClient(latency_median_ms=350.0, rng=rng, cache=cache)
        model_name = "mock"
    else:
        if not cfg.get("api_key"):
            print("ERROR: no LLM_API_KEY. Put it in .env or export it.")
            print("Copy .env.example to .env, then fill in your Groq key.")
            return 1
        client = RealLLMClient(
            model=cfg.get("model"),
            base_url=cfg.get("base_url"),
            api_key=cfg.get("api_key"),
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 200),
            retry_max=cfg.get("retry_max", 1),
            timeout_s=cfg.get("timeout_s", 30.0),
            json_mode=cfg.get("response_format_json", True),
            cache=cache,
        )
        model_name = cfg.get("model")
        print(f"provider={cfg.get('provider', 'openai-compatible')} model={model_name}")

    rng = np.random.default_rng(args.seed)
    scenarios = ["surge", "shadow"]
    rows = []
    latencies = []
    t_start = time.time()

    for i in range(args.n):
        scenario = scenarios[i % len(scenarios)]
        features = make_features(scenario, i, rng)
        prompt = build_prompt(features)
        t0 = time.perf_counter()
        response, latency_ms, ok, reason = client.call(prompt, features)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        if latency_ms <= 0.0:
            latency_ms = wall_ms

        mode = response.get("operating_mode") if isinstance(response, dict) else None
        correct = bool(mode in EXPECTED[scenario]) if mode else False
        rows.append({
            "i": i,
            "scenario": scenario,
            "ok": bool(ok),
            "reason": reason,
            "mode": mode,
            "diagnosis_correct": correct,
            "latency_ms": float(latency_ms),
            "prompt_tokens": int(getattr(client, "last_usage", {}).get("prompt_tokens", 0)),
            "completion_tokens": int(getattr(client, "last_usage", {}).get("completion_tokens", 0)),
        })
        if ok:
            latencies.append(latency_ms)
        print(
            f"[{i+1:3d}/{args.n}] {scenario:6s} ok={ok!s:5s} mode={str(mode):10s} "
            f"correct={correct!s:5s} latency={latency_ms:7.1f}ms reason={reason}"
        )

    n = len(rows)
    n_ok = sum(r["ok"] for r in rows)
    n_correct = sum(r["diagnosis_correct"] for r in rows)
    valid_rate = n_ok / max(n, 1)
    diag_acc = n_correct / max(n, 1)
    lat = fit_lognormal(latencies)

    stats = getattr(client, "stats", lambda: {})()
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "provider": cfg.get("provider"),
        "base_url": cfg.get("base_url"),
        "model": model_name,
        "n": n,
        "valid_rate": valid_rate,
        "diagnosis_accuracy": diag_acc,
        "latency": lat,
        "wall_seconds": time.time() - t_start,
        "client_stats": stats,
        "cache_enabled": cache is not None,
        "mock": bool(args.mock),
    }

    if rows:
        out_dir = os.path.join(str(root), args.out)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "smoke_test.json"), "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "rows": rows}, f, indent=2)
        with open(os.path.join(out_dir, "smoke_test.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print("\n=== smoke test summary ===")
    print(f"model                 : {model_name}")
    print(f"valid JSON/schema rate: {valid_rate:.1%}  (target >= {args.min_valid:.0%})")
    print(f"diagnosis accuracy    : {diag_acc:.1%}  (strict label match)")
    if lat:
        print(
            f"latency               : median {lat['median_ms']:.0f} ms, "
            f"p5 {lat['p5_ms']:.0f} ms, p95 {lat['p95_ms']:.0f} ms (n={lat['n']})"
        )
    if stats:
        print(f"tokens                : {stats.get('prompt_tokens', 0)} in / "
              f"{stats.get('completion_tokens', 0)} out")

    if args.write_latency_config and lat:
        import yaml
        path = os.path.join(str(root), "configs", "llm_latency.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"llm_latency": lat}, f, sort_keys=False)
        print(f"wrote {path}")

    if n_ok == 0:
        print("\nNo successful calls. Check the key, endpoint and network/region access.")
        print(f"last error: {stats.get('last_error') if stats else 'see messages above'}")
        return 1
    return 0 if valid_rate >= args.min_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
