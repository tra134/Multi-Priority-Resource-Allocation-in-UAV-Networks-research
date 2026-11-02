"""Phase 9 grid runner: compare all methods on S0-S5 with a resumable CSV.

Methods (plan.md Section 6, Phase 9)
------------------------------------
M1  strict priority              static heuristic
M2  greedy sum-rate              static heuristic
M3  Pure SAC                     N=2 checkpoint, no LLM / no mask
M4  REDQ-SAC                     N=10 checkpoint, no LLM / no mask
M5  DR-SAC                       N=2 checkpoint trained with --dr-wide
M7  Rule-Shield                  queue-triggered mask (0 ms and 350 ms)
M8  Random-trigger LLM           random timing at --random-rate
M9  Periodic LLM-DRL             every K slots (K and 2x)
M10 Reactive LLM-DRL             trigger after the first violation
M11 Always-LLM                   mock LLM, one call per slot
M12 Proposed                     hybrid trigger + LLM + safety layer
M12b Proposed + interim fallback fallback while the response is in flight

M6 (Lagrangian SAC) is optional in plan.md and is not implemented.

Scenarios
---------
S0 nominal, S1 surge, S2 shadow, S3 surge + degraded API, S4 compound +
degraded API, S5 unseen magnitude.

Usage
-----
    # full grid, mock LLM (fast, reproducible; 5 training seeds)
    python scripts/run_grid.py --seeds 0,1,2,3,4 --episodes 20

    # one Colab-sized chunk (resume on the next session)
    python scripts/run_grid.py --seeds 0 --methods m1_strict,m2_greedy,m3_pure_sac

    # real Groq arbiter (costs tokens; no S3/S4 latency inflation for real calls)
    python scripts/run_grid.py --seeds 0 --llm real

Output: ``results/final/metrics.csv`` (one row per method/scenario/seed/episode).
Re-running with the same arguments skips completed rows.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
    ),
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uavrl.agents.sac import SACAgent
from uavrl.baselines.rule_shield import RuleShield
from uavrl.baselines.static_policies import (
    fair_share,
    greedy_sum_rate,
    strict_priority,
)
from uavrl.config import (
    checkpoint_run_name,
    find_project_root,
    llm_settings,
    load_env,
)
from uavrl.control.hybrid_controller import HybridController
from uavrl.env.scenarios import Scenario
from uavrl.env.uav_env import UAVEnv
from uavrl.eval.runner import run_episode
from uavrl.llm.cache import LLMCache
from uavrl.llm.client import MockLLMClient, RealLLMClient

# --------------------------------------------------------------------------- registry
SCENARIOS = {
    "S0": {
        "name": "nominal",
        "disturbance": {"type": "none"},
        "ood": False,
        "mode": "NORMAL",
        "latency": "nominal",
    },
    "S1": {
        "name": "surge",
        "disturbance": {
            "type": "surge", "surge_multiplier": 4.0,
            "duration_slots": 400, "start_slot": [500, 1200],
        },
        "ood": True,
        "mode": "SOS_SURGE",
        "latency": "nominal",
    },
    "S2": {
        "name": "shadow",
        "disturbance": {
            "type": "shadow", "extra_db": 15.0,
            "duration_slots": 400, "start_slot": [500, 1200],
        },
        "ood": True,
        "mode": "DEEP_FADING",
        "latency": "nominal",
    },
    "S3": {
        "name": "surge_stress",
        "disturbance": {
            "type": "surge", "surge_multiplier": 4.0,
            "duration_slots": 400, "start_slot": [500, 1200],
            "latency_median_ms": 4000.0,
        },
        "ood": True,
        "mode": "SOS_SURGE",
        "latency": "stress",
    },
    "S4": {
        "name": "compound_stress",
        "disturbance": {
            "type": "compound", "surge_multiplier": 4.0, "shadow_db": 15.0,
            "duration_slots": 400, "start_slot": [500, 1200],
            "latency_median_ms": 4000.0,
        },
        "ood": True,
        "mode": "COMPOUND",
        "latency": "stress",
    },
    "S5": {
        "name": "unseen",
        "disturbance": {
            "type": "compound", "surge_multiplier": 8.0, "shadow_db": 25.0,
            "duration_slots": 400, "start_slot": [500, 1200],
        },
        "ood": True,
        "mode": "COMPOUND",
        "latency": "nominal",
    },
}

METHODS = {
    "m1_strict": {"label": "M1 Strict Priority", "kind": "static",
                  "policy": strict_priority},
    "m2_greedy": {"label": "M2 Greedy Sum-Rate", "kind": "static",
                  "policy": greedy_sum_rate},
    "m3_pure_sac": {"label": "M3 Pure SAC (N=2)", "kind": "agent",
                    "n_critics": 2},
    "m4_redq_sac": {"label": "M4 REDQ-SAC (N=10)", "kind": "agent",
                    "n_critics": 10},
    "m5_dr_sac": {"label": "M5 DR-SAC (wide DR, N=2)", "kind": "agent",
                  "n_critics": 2, "suffix": "_drwide"},
    "m7_rule_shield": {"label": "M7 Rule-Shield (0 ms)", "kind": "rule",
                       "latency_ms": 0},
    "m7_rule_shield_350ms": {"label": "M7 Rule-Shield (350 ms)",
                             "kind": "rule", "latency_ms": 350},
    "m8_random_llm": {"label": "M8 Random-trigger LLM", "kind": "llm",
                      "trigger": "random"},
    "m9_periodic_llm": {"label": "M9 Periodic LLM (K)", "kind": "llm",
                        "trigger": "periodic"},
    "m9_periodic_llm_2x": {"label": "M9 Periodic LLM (2x)", "kind": "llm",
                           "trigger": "periodic", "k_div": 2},
    "m10_reactive_llm": {"label": "M10 Reactive LLM", "kind": "llm",
                         "trigger": "reactive"},
    "m11_always_llm": {"label": "M11 Always-LLM (mock)", "kind": "llm",
                       "trigger": "always", "force_mock": True},
    "m12_proposed": {"label": "M12 Proposed", "kind": "llm",
                     "trigger": "hybrid"},
    "m12b_interim": {"label": "M12b Proposed + interim fallback", "kind": "llm",
                     "trigger": "hybrid", "interim": True},
}

FIELDS = [
    "method", "method_label", "scenario", "scenario_name", "seed", "episode",
    "eval_seed", "slots", "n_critics", "trigger_mode",
    "sos_violation_rate", "sos_p99_delay_s", "mean_delay_emg_s", "mean_reward",
    "mean_rate_emg_bps", "mean_rate_imp_bps", "mean_rate_norm_bps",
    "imp_throughput_ratio", "norm_throughput_ratio",
    "total_drops_emg_bits", "total_drops_imp_bits", "total_drops_norm_bits",
    "llm_calls", "llm_call_rate", "llm_valid", "llm_invalid", "llm_timeouts",
    "mask_active_fraction", "fallback_active_fraction", "trigger_fraction",
    "json_valid_rate", "timeout_rate",
    "trigger_precision", "trigger_recall", "lead_time_slots",
    "diagnosis_accuracy", "budget_violation_count",
    "prompt_tokens", "completion_tokens", "llm_latency_ms_median",
    "inference_ms_per_slot", "wall_ms_per_slot", "notes",
]


# --------------------------------------------------------------------------- helpers
class LatencyInflationClient:
    """Add synthetic delay to the reported latency (S3/S4 with a real API).

    The controller schedules response arrival from the reported latency, so
    adding milliseconds simulates a degraded provider without sleeping.
    """

    def __init__(self, inner, extra_ms):
        self.inner = inner
        self.extra_ms = float(extra_ms)

    def call(self, prompt, features=None):
        resp, latency_ms, ok, reason = self.inner.call(prompt, features)
        return resp, float(latency_ms) + self.extra_ms, ok, reason

    def stats(self):
        return self.inner.stats() if hasattr(self.inner, "stats") else {}


def make_scenario(scenario_id, eval_seed):
    spec = SCENARIOS[scenario_id]
    return Scenario(
        spec["name"],
        dict(spec["disturbance"]),
        ood=spec["ood"],
        rng=np.random.default_rng(eval_seed),
    )


def trigger_metrics(steps):
    """Precision / recall / lead time of the controller trigger vs OOD flag."""
    if not steps:
        return math.nan, math.nan, math.nan
    ood = np.array([bool(s.get("ood_flag", False)) for s in steps])
    trig = np.array([bool(s.get("trigger", False)) for s in steps])
    if not trig.any():
        return math.nan, math.nan, math.nan

    tp = int(np.sum(ood & trig))
    precision = tp / int(trig.sum()) if trig.any() else math.nan
    recall = tp / int(ood.sum()) if ood.any() else math.nan

    # OOD segments -> first trigger within 100 slots before the segment.
    padded = np.r_[0, ood.astype(int), 0]
    edges = np.flatnonzero(np.diff(padded))
    leads = []
    for k in range(0, len(edges), 2):
        start, end = int(edges[k]), int(edges[k + 1])
        lo = max(0, start - 100)
        window = trig[lo:end]
        if window.any():
            trigger_slot = lo + int(np.argmax(window))
            leads.append(start - trigger_slot)
    lead = float(np.mean(leads)) if leads else math.nan
    return float(precision), float(recall), lead


def diagnosis_accuracy(call_log, scenario_mode):
    calls = [c for c in call_log if c.get("ok") and c.get("mode")]
    if not calls:
        return math.nan
    correct = sum(
        c["mode"] == (scenario_mode if c.get("ood") else "NORMAL")
        for c in calls
    )
    return correct / len(calls)


def benchmark_agent(agent, n=100):
    obs = np.zeros(agent.obs_dim, dtype=np.float32)
    for _ in range(5):
        a = agent.select_action(obs, deterministic=True)
        agent.uncertainty(obs, a)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        a = agent.select_action(obs, deterministic=True)
        agent.uncertainty(obs, a)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def benchmark_policy(policy, n=100):
    obs = np.zeros(17, dtype=np.float32)
    info = {"gain": [1.0, 1.0, 1.0], "queue_bits": [0.0, 0.0, 0.0]}
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        policy(obs, info)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def clean(value):
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def parse_list(value):
    return [v.strip() for v in value.split(",") if v.strip()]


def checkpoint_path(ckpt_dir, seed, n_critics, suffix=""):
    """Locate a checkpoint in either supported Drive layout.

    Flat:      <ckpt_dir>/sac_seed0_N10_latest.pt
    Per-run:   <ckpt_dir>/sac_seed0_N10/sac_seed0_N10_latest.pt
    """
    name = checkpoint_run_name("sac", seed, n_critics,
                               dr_wide=(suffix == "_drwide"))
    flat = os.path.join(ckpt_dir, f"{name}_latest.pt")
    if os.path.exists(flat):
        return flat
    nested = os.path.join(ckpt_dir, name, f"{name}_latest.pt")
    if os.path.exists(nested):
        return nested
    return flat  # report the canonical flat path when nothing exists


def preflight_checkpoints(methods, seeds, ckpt_dir):
    missing = []
    for seed in seeds:
        for name in methods:
            spec = METHODS[name]
            if spec["kind"] != "agent":
                continue
            path = checkpoint_path(
                ckpt_dir, seed, spec["n_critics"], spec.get("suffix", "")
            )
            if not os.path.exists(path):
                missing.append(path)
    if missing:
        raise SystemExit(
            "Missing checkpoints (train them first):\n  "
            + "\n  ".join(missing)
        )


def nominal_reference(cfg, eval_seed):
    """Nominal fair-share mean rates, used for throughput ratios."""
    env = UAVEnv(
        config=cfg,
        scenario=Scenario("nominal", {"type": "none"}, ood=False),
        seed=eval_seed,
    )
    _, summary = run_episode(env, fair_share, seed=eval_seed)
    return float(summary["mean_rate_imp_bps"]), float(summary["mean_rate_norm_bps"])


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description="Phase 9 grid: M1-M12 on S0-S5 (resumable CSV)"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--llm-config", default="configs/llm.yaml")
    parser.add_argument("--out", default="results/final/metrics.csv")
    parser.add_argument("--ckpt-dir", default="results/ckpt")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--seeds", default="0,1,2,3,4",
                        help="training seeds (also the paired blocks)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--slots", type=int, default=None,
                        help="episode length; default from config")
    parser.add_argument("--eval-seed0", type=int, default=10000)
    parser.add_argument("--llm", default="mock", choices=["mock", "real"])
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--tau-unc", type=float, default=None)
    parser.add_argument("--theta-queue", type=float, default=None)
    parser.add_argument("--cooldown", type=int, default=None)
    parser.add_argument("--ttl-slots", type=int, default=None)
    parser.add_argument("--random-rate", type=float, default=0.05)
    parser.add_argument("--periodic-k", type=int, default=250)
    parser.add_argument("--rule-theta", type=float, default=20000.0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    load_env()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    timing = cfg.get("timing", {})
    slots = int(args.slots or timing.get("episode_slots", 2000))
    # The env reads its horizon from the config; --slots must shorten the
    # episode itself, not just the reported row.
    cfg.setdefault("timing", {})["episode_slots"] = slots
    nominal_latency = float(timing.get("llm_latency_median_ms", 350.0))
    stress_latency = float(timing.get("stress_latency_median_ms", 2000.0))
    latency_sigma = float(timing.get("llm_latency_sigma", 0.6))
    timeout_ms = int(timing.get("llm_timeout_ms", 1000))

    trigger_cfg = cfg.get("trigger", {})
    tau_unc = args.tau_unc if args.tau_unc is not None else trigger_cfg.get("tau_unc")
    theta_queue = (args.theta_queue if args.theta_queue is not None
                   else trigger_cfg.get("theta_queue"))
    if theta_queue is None:
        theta_queue = 20000.0
    cooldown = (args.cooldown if args.cooldown is not None
                else int(trigger_cfg.get("cooldown", 50)))
    ttl_slots = (args.ttl_slots if args.ttl_slots is not None
                 else int(trigger_cfg.get("ttl_slots", 600)))

    methods = parse_list(args.methods)
    scenarios = parse_list(args.scenarios)
    seeds = [int(s) for s in parse_list(args.seeds)]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}")
    unknown = [s for s in scenarios if s not in SCENARIOS]
    if unknown:
        raise SystemExit(f"Unknown scenarios: {unknown}")

    preflight_checkpoints(methods, seeds, args.ckpt_dir)

    # ---- LLM clients
    llm_cfg = llm_settings(args.llm_config)
    if args.llm == "real" and not llm_cfg.get("api_key"):
        raise SystemExit("LLM_API_KEY is not set; use --llm mock or configure .env")
    cache = None
    if not args.no_cache and llm_cfg.get("cache", {}).get("enabled", True):
        cache = LLMCache(enabled=True, dir=llm_cfg["cache"].get("dir", "results/llm_cache"))

    real_clients = {}

    def get_real_client(method_name):
        if method_name not in real_clients:
            real_clients[method_name] = RealLLMClient(
                model=llm_cfg.get("model"),
                base_url=llm_cfg.get("base_url"),
                api_key=llm_cfg.get("api_key"),
                temperature=llm_cfg.get("temperature", 0.0),
                max_tokens=llm_cfg.get("max_tokens", 512),
                retry_max=llm_cfg.get("retry_max", 1),
                timeout_s=llm_cfg.get("timeout_s", 30.0),
                json_mode=llm_cfg.get("response_format_json", True),
                reasoning_effort=llm_cfg.get("reasoning_effort"),
                rate_limit_retries=llm_cfg.get("rate_limit_retries", 3),
                rate_limit_max_sleep_s=llm_cfg.get("rate_limit_max_sleep_s", 30.0),
                cache=cache,
            )
        return real_clients[method_name]

    def make_client(method_name, spec, scenario_id, eval_seed):
        if spec.get("force_mock"):
            use_mock = True
        else:
            use_mock = args.llm == "mock"
        if use_mock:
            median = (stress_latency if SCENARIOS[scenario_id]["latency"] == "stress"
                      else nominal_latency)
            return MockLLMClient(
                latency_median_ms=median,
                latency_sigma=latency_sigma,
                rng=np.random.default_rng(eval_seed),
            ), "mock"
        client = get_real_client(method_name)
        if SCENARIOS[scenario_id]["latency"] == "stress":
            client = LatencyInflationClient(
                client, extra_ms=max(0.0, stress_latency - nominal_latency)
            )
        return client, "real"

    # ---- output + resume
    out_path = os.path.join(str(find_project_root()), args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != FIELDS:
                raise SystemExit(
                    "Existing metrics.csv has a different schema; "
                    "move it aside or run with --no-resume."
                )
            for row in reader:
                done.add((row["method"], row["scenario"],
                          str(row["seed"]), str(row["episode"])))
    write_header = not os.path.exists(out_path) or not args.resume
    csv_file = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
    if write_header:
        writer.writeheader()

    # ---- caches
    agents = {}
    refs = {}
    infer_ms = {}

    def get_agent(spec, seed):
        key = (spec["n_critics"], spec.get("suffix", ""), seed)
        if key not in agents:
            agent = SACAgent(
                obs_dim=17, act_dim=6,
                device="cuda" if _cuda() else "cpu",
                seed=seed, N_critics=spec["n_critics"],
                rng=np.random.default_rng(seed),
            )
            path = checkpoint_path(
                args.ckpt_dir, seed, spec["n_critics"], spec.get("suffix", "")
            )
            agent.load(path)
            agents[key] = agent
        return agents[key]

    total = len(methods) * len(scenarios) * len(seeds) * args.episodes
    n_done = 0
    t_start = time.time()
    print(f"grid: {len(methods)} methods x {len(scenarios)} scenarios x "
          f"{len(seeds)} seeds x {args.episodes} episodes = {total} rows")

    for method_name in methods:
        spec = METHODS[method_name]
        for scenario_id in scenarios:
            scenario_spec = SCENARIOS[scenario_id]
            for seed in seeds:
                eval_base = args.eval_seed0 + seed * 1000
                if seed not in refs:
                    refs[seed] = nominal_reference(cfg, eval_base)
                ref_imp, ref_norm = refs[seed]
                for episode in range(args.episodes):
                    key = (method_name, scenario_id, str(seed), str(episode))
                    if key in done:
                        continue
                    eval_seed = eval_base + episode
                    env = UAVEnv(
                        config=cfg,
                        scenario=make_scenario(scenario_id, eval_seed),
                        seed=eval_seed,
                    )

                    row = {field: "" for field in FIELDS}
                    row.update({
                        "method": method_name,
                        "method_label": spec["label"],
                        "scenario": scenario_id,
                        "scenario_name": scenario_spec["name"],
                        "seed": seed,
                        "episode": episode,
                        "eval_seed": eval_seed,
                        "slots": slots,
                        "budget_violation_count": 0,
                    })

                    t0 = time.perf_counter()
                    if spec["kind"] == "static":
                        policy = spec["policy"]
                        steps, summary = run_episode(env, policy, seed=eval_seed)
                        if (method_name, seed) not in infer_ms:
                            infer_ms[(method_name, seed)] = benchmark_policy(policy)
                    elif spec["kind"] == "agent":
                        agent = get_agent(spec, seed)

                        def policy(obs, info, _agent=agent):
                            return _agent.select_action(obs, deterministic=True)

                        steps, summary = run_episode(env, policy, seed=eval_seed)
                        if (method_name, seed) not in infer_ms:
                            infer_ms[(method_name, seed)] = benchmark_agent(agent)
                        row["n_critics"] = spec["n_critics"]
                    elif spec["kind"] == "rule":
                        shield = RuleShield(
                            theta_queue_bits=args.rule_theta,
                            m=3, ttl_slots=ttl_slots, cooldown=cooldown,
                            latency_slots=int(round(spec["latency_ms"] / cfg.get(
                                "timing", {}).get("slot_ms", 5.0))),
                        )
                        steps, summary = run_episode(env, shield, seed=eval_seed)
                        row["mask_active_fraction"] = (
                            shield.mask_active_slots / max(len(steps), 1)
                        )
                        row["fallback_active_fraction"] = 0.0
                        row["trigger_fraction"] = (
                            shield.trigger_count / max(len(steps), 1)
                        )
                        if (method_name, seed) not in infer_ms:
                            infer_ms[(method_name, seed)] = benchmark_policy(shield)
                    else:  # LLM controller
                        agent = get_agent(
                            {"n_critics": 10, "suffix": ""}, seed
                        )
                        client, client_kind = make_client(
                            method_name, spec, scenario_id, eval_seed
                        )
                        trigger = spec["trigger"]
                        ctrl = HybridController(
                            agent, env, client, cfg,
                            trigger_mode=trigger,
                            tau_unc=tau_unc,
                            theta_queue=theta_queue,
                            debounce_m=int(trigger_cfg.get("debounce_m", 3)),
                            cooldown=0 if trigger in ("periodic", "always", "random")
                            else cooldown,
                            ttl_slots=ttl_slots,
                            timeout_ms=timeout_ms,
                            fallback_slots=int(cfg.get("fallback", {}).get(
                                "fallback_slots", 200)),
                            periodic_K=(args.periodic_k // spec.get("k_div", 1)),
                            random_rate=args.random_rate,
                            interim_fallback=bool(spec.get("interim", False)),
                            rng=np.random.default_rng(eval_seed),
                        )
                        stats_before = (
                            client.stats() if client_kind == "real" else {}
                        )
                        steps, summary = ctrl.run_episode(seed=eval_seed)
                        stats_after = (
                            client.stats() if client_kind == "real" else {}
                        )
                        row["n_critics"] = 10
                        row["trigger_mode"] = trigger
                        for k in ["mask_active_fraction",
                                  "fallback_active_fraction",
                                  "trigger_fraction"]:
                            row[k] = summary.get(k, "")
                        precision, recall, lead = trigger_metrics(steps)
                        row["trigger_precision"] = precision
                        row["trigger_recall"] = recall
                        row["lead_time_slots"] = lead
                        row["diagnosis_accuracy"] = diagnosis_accuracy(
                            ctrl.call_log, scenario_spec["mode"]
                        )
                        calls = ctrl.call_log
                        valid_calls = sum(1 for c in calls if c["ok"])
                        timeout_calls = sum(1 for c in calls if c["timeout"])
                        row["llm_calls"] = len(calls)
                        row["llm_call_rate"] = len(calls) / max(len(steps), 1)
                        row["llm_valid"] = valid_calls
                        row["llm_invalid"] = len(calls) - valid_calls
                        row["llm_timeouts"] = timeout_calls
                        if calls:
                            row["json_valid_rate"] = valid_calls / len(calls)
                            row["timeout_rate"] = timeout_calls / len(calls)
                            lat = [c["latency_ms"] for c in calls
                                   if c["latency_ms"] > 0]
                            if lat:
                                row["llm_latency_ms_median"] = float(
                                    np.median(lat)
                                )
                        if client_kind == "real":
                            row["prompt_tokens"] = (
                                stats_after.get("prompt_tokens", 0)
                                - stats_before.get("prompt_tokens", 0)
                            )
                            row["completion_tokens"] = (
                                stats_after.get("completion_tokens", 0)
                                - stats_before.get("completion_tokens", 0)
                            )
                            row["notes"] = (
                                "real LLM"
                                + (" + latency inflation" if scenario_spec[
                                    "latency"] == "stress" else "")
                            )
                        else:
                            row["notes"] = "mock LLM"
                        if (method_name, seed) not in infer_ms:
                            infer_ms[(method_name, seed)] = benchmark_agent(agent)

                    wall_s = time.perf_counter() - t0
                    row["wall_ms_per_slot"] = wall_s * 1000.0 / max(len(steps), 1)
                    row["inference_ms_per_slot"] = infer_ms.get(
                        (method_name, seed), ""
                    )
                    for k in [
                        "sos_violation_rate", "sos_p99_delay_s",
                        "mean_delay_emg_s", "mean_reward",
                        "mean_rate_emg_bps", "mean_rate_imp_bps",
                        "mean_rate_norm_bps", "total_drops_emg_bits",
                        "total_drops_imp_bits", "total_drops_norm_bits",
                    ]:
                        row[k] = summary.get(k, "")
                    row["imp_throughput_ratio"] = (
                        summary["mean_rate_imp_bps"] / ref_imp
                        if ref_imp > 0 else math.nan
                    )
                    row["norm_throughput_ratio"] = (
                        summary["mean_rate_norm_bps"] / ref_norm
                        if ref_norm > 0 else math.nan
                    )
                    writer.writerow({k: clean(v) for k, v in row.items()})
                    csv_file.flush()
                    n_done += 1
                    if n_done % 10 == 0 or n_done == 1:
                        elapsed = time.time() - t_start
                        print(
                            f"[{n_done}/{total - len(done)}] "
                            f"{method_name} {scenario_id} seed={seed} "
                            f"ep={episode}  ({elapsed:.0f}s)",
                            flush=True,
                        )

    csv_file.close()
    print(f"done: {n_done} new rows -> {out_path}")


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
