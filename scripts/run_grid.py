"""Experiment grid runner: compare all methods on S0-S5 with a resumable CSV.

Methods (plan.md Section 6, Phase 9; changed by Section 11, Stage 1)
--------------------------------------------------------------------
M1   queue-aware strict priority          static heuristic (F1)
M1b  emergency-only bound                 NOT a fair baseline; upper bound
M2   greedy sum-rate                      static heuristic
M3   Pure SAC                             N=2 checkpoint, no LLM / no mask
M4   REDQ-SAC                             plain N=10 checkpoint, no LLM / no mask
M5   DR-SAC                               N=2 checkpoint trained with --dr-wide
M7   Rule-Shield                          mask-aware N=10 SAC + queue-triggered rule mask
                                          (zero latency, and fitted latency)
M8   Random-trigger LLM                   random timing at --random-rate
M9   Periodic LLM-DRL                     every K slots (K and 2x)
M10  Reactive LLM-DRL                     trigger after the first violation
M11  Always-LLM                           mock LLM, one call per slot
M12  Proposed                             hybrid trigger + LLM + safety layer
M12b Proposed + interim fallback          fallback while the response is in flight
M13  Fixed Reservation                    N=10 SAC + permanent Emergency floor

All learned methods use the deterministic actor (F4).  LLM timing always comes
from the fitted latency model, for mock and real clients alike (F13).
M6 (Lagrangian SAC) is optional in plan.md and is not implemented.

Scenarios live in configs/scenarios.yaml (F20): S0 nominal, S1 surge,
S2 shadow, S3 surge + degraded API, S4 compound + degraded API,
S5 unseen magnitude.

Usage
-----
    # full grid, mock LLM (pipeline check; 5 training seeds)
    python scripts/run_grid.py --seeds 0,1,2,3,4 --episodes 20

    # static methods only (no checkpoints needed)
    python scripts/run_grid.py --seeds 0 --methods m1_strict,m1b_emergency_only,m2_greedy

    # real Groq arbiter (costs tokens; only these rows count for Claim B)
    python scripts/run_grid.py --seeds 0 --llm real

Output: results/final/metrics.csv (one row per method/scenario/seed/episode)
and results/final/manifest.jsonl (config hash, commit, seeds, client, time).
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from uavrl.agents.sac import SACAgent
from uavrl.baselines.rule_shield import FixedReservation, RuleShield
from uavrl.baselines.static_policies import (
    QueueAwarePriority,
    emergency_only,
    greedy_sum_rate,
)
from uavrl.config import (
    checkpoint_run_name,
    find_project_root,
    llm_settings,
    load_env,
)
from uavrl.control.hybrid_controller import HybridController
from uavrl.env.scenarios import load_scenario_specs
from uavrl.env.scenarios import make_scenario as _registry_make_scenario
from uavrl.env.uav_env import UAVEnv
from uavrl.eval.manifest import append_manifest, collect_manifest
from uavrl.eval.metrics import episode_metrics, finite_or_blank
from uavrl.eval.runner import run_episode
from uavrl.llm.cache import LLMCache
from uavrl.llm.client import MockLLMClient, RealLLMClient
from uavrl.llm.latency import LatencyModel
from uavrl.seeds import MAX_EPISODES, MAX_TRAIN_SEEDS, test_seed

SCENARIO_FILE = os.path.join(ROOT, "configs", "scenarios.yaml")
SCENARIOS, _ALIASES = load_scenario_specs(SCENARIO_FILE)


def make_scenario(scenario_id, eval_seed):
    """Scenario from configs/scenarios.yaml; ``eval_seed`` draws the start slot."""
    return _registry_make_scenario(scenario_id, eval_seed, path=SCENARIO_FILE)


# --------------------------------------------------------------------------- registry
# Methods that use masks (Rule-Shield, Fixed Reservation, all LLM variants) share
# one mask-aware N=10 base policy; M4 evaluates the plain N=10 policy (D5, F11).
PLAIN_N10 = {"n_critics": 10, "suffix": ""}
MASK_N10 = {"n_critics": 10, "suffix": "_mask"}

METHODS = {
    "m1_strict": {"label": "M1 Strict Priority (queue-aware)", "kind": "static",
                  "factory": "queue_priority"},
    "m1b_emergency_only": {"label": "M1b Emergency-only (bound)", "kind": "static",
                           "policy": emergency_only},
    "m2_greedy": {"label": "M2 Greedy Sum-Rate", "kind": "static",
                  "policy": greedy_sum_rate},
    "m3_pure_sac": {"label": "M3 Pure SAC (N=2)", "kind": "agent", "n_critics": 2},
    "m4_redq_sac": {"label": "M4 REDQ-SAC (N=10)", "kind": "agent", "n_critics": 10},
    "m5_dr_sac": {"label": "M5 DR-SAC (wide DR, N=2)", "kind": "agent",
                  "n_critics": 2, "suffix": "_drwide"},
    "m7_rule_shield": {"label": "M7 Rule-Shield (0 ms)", "kind": "rule",
                       "shield_latency": "zero"},
    "m7_rule_shield_fitted": {"label": "M7 Rule-Shield (fitted latency)",
                              "kind": "rule", "shield_latency": "fitted"},
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
    "m12_proposed": {"label": "M12 Proposed", "kind": "llm", "trigger": "hybrid"},
    "m12b_interim": {"label": "M12b Proposed + interim fallback", "kind": "llm",
                     "trigger": "hybrid", "interim": True},
    "m13_fixed_reserve": {"label": "M13 Fixed Reservation", "kind": "reserve"},
}

FIELDS = [
    "method", "method_label", "scenario", "scenario_name", "seed", "episode",
    "eval_seed", "slots", "client", "n_critics", "trigger_mode",
    "sos_violation_ood", "sos_violation_recovery", "sos_violation_episode",
    "sos_p99_delay_s", "sos_p99_delay_ood_s", "mean_delay_emg_s", "mean_reward",
    "delivered_emg_mbps", "delivered_imp_mbps", "delivered_norm_mbps",
    "delivered_imp_ood_mbps", "delivered_norm_ood_mbps",
    "loss_ratio_emg", "total_drops_emg_bits", "n_events",
    "mean_b_share_emg", "mean_p_share_emg",
    "mean_r_emg", "mean_r_imp", "mean_r_norm", "mean_penalty",
    "llm_calls", "llm_call_rate", "llm_valid", "llm_invalid", "llm_timeouts",
    "mask_active_fraction", "fallback_active_fraction", "trigger_fraction",
    "json_valid_rate", "timeout_rate",
    "trigger_precision", "trigger_recall", "detection_delay_slots",
    "warning_margin_slots", "false_triggers_per_episode",
    "diagnosis_accuracy", "budget_violation_count",
    "prompt_tokens", "completion_tokens", "llm_latency_ms_median",
    "measured_latency_ms_median",
    "inference_ms_per_slot", "wall_ms_per_slot", "notes",
]


# --------------------------------------------------------------------------- helpers
def diagnosis_accuracy(call_log, scenario_mode):
    """Share of answered calls whose mode matches the ground truth (scoring only)."""
    calls = [c for c in call_log if c.get("ok") and c.get("mode")]
    if not calls:
        return math.nan
    correct = sum(
        c["mode"] == (scenario_mode if c.get("ood") else "NORMAL") for c in calls
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


def parse_list(value):
    return [v.strip() for v in value.split(",") if v.strip()]


def checkpoint_path(ckpt_dir, seed, n_critics, suffix=""):
    """Locate a checkpoint in either supported Drive layout.

    Flat:      <ckpt_dir>/sac_seed0_N10_latest.pt
    Per-run:   <ckpt_dir>/sac_seed0_N10/sac_seed0_N10_latest.pt
    """
    name = checkpoint_run_name("sac", seed, n_critics,
                               dr_wide=(suffix == "_drwide"),
                               mask_aware=(suffix == "_mask"))
    flat = os.path.join(ckpt_dir, f"{name}_latest.pt")
    if os.path.exists(flat):
        return flat
    nested = os.path.join(ckpt_dir, name, f"{name}_latest.pt")
    if os.path.exists(nested):
        return nested
    return flat  # report the canonical flat path when nothing exists


def agent_spec(spec):
    """Which checkpoint a method needs, or None for static methods."""
    if spec["kind"] == "agent":
        return {"n_critics": spec["n_critics"], "suffix": spec.get("suffix", "")}
    if spec["kind"] in ("rule", "reserve", "llm"):
        return MASK_N10
    return None


def preflight_checkpoints(methods, seeds, ckpt_dir):
    missing = []
    for seed in seeds:
        for name in methods:
            need = agent_spec(METHODS[name])
            if need is None:
                continue
            path = checkpoint_path(ckpt_dir, seed, need["n_critics"], need["suffix"])
            if not os.path.exists(path) and path not in missing:
                missing.append(path)
    if missing:
        raise SystemExit(
            "Missing checkpoints (train them first):\n  " + "\n  ".join(missing)
        )


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description="Experiment grid: M1-M13 on S0-S5 (resumable CSV)"
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
    parser.add_argument("--llm", default="mock", choices=["mock", "real"])
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--tau-unc", type=float, default=None)
    parser.add_argument("--theta-queue", type=float, default=None)
    parser.add_argument("--cooldown", type=int, default=None)
    parser.add_argument("--ttl-slots", type=int, default=None)
    parser.add_argument("--random-rate", type=float, default=0.05,
                        help="M8 call rate; Stage 5 sets it from M12's validation rate")
    parser.add_argument("--periodic-k", type=int, default=250,
                        help="M9 period; Stage 5 sets it from M12's validation rate")
    parser.add_argument("--rule-theta", type=float, default=None,
                        help="Rule-Shield threshold; default = trigger.theta_queue")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    if args.episodes > MAX_EPISODES:
        raise SystemExit(f"--episodes must be <= {MAX_EPISODES}")

    load_env()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    timing = cfg.get("timing", {})
    slots = int(args.slots or timing.get("episode_slots", 2000))
    # The env reads its horizon from the config; --slots must shorten the
    # episode itself, not just the reported row.
    cfg.setdefault("timing", {})["episode_slots"] = slots
    slot_ms = float(timing.get("slot_ms", 5.0))
    nominal_latency = float(timing.get("llm_latency_median_ms", 1159.0))
    stress_latency = float(timing.get("stress_latency_median_ms", 4000.0))
    latency_sigma = float(timing.get("llm_latency_sigma", 0.705))
    timeout_ms = int(timing.get("llm_timeout_ms", 5000))

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
    rule_theta = args.rule_theta if args.rule_theta is not None else theta_queue

    methods = parse_list(args.methods)
    scenarios = parse_list(args.scenarios)
    seeds = [int(s) for s in parse_list(args.seeds)]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}")
    unknown = [s for s in scenarios if s not in SCENARIOS]
    if unknown:
        raise SystemExit(f"Unknown scenarios: {unknown}")
    if any(not (0 <= s < MAX_TRAIN_SEEDS) for s in seeds):
        raise SystemExit(f"--seeds must be in [0, {MAX_TRAIN_SEEDS})")

    preflight_checkpoints(methods, seeds, args.ckpt_dir)

    # ---- LLM clients
    llm_cfg = llm_settings(args.llm_config)
    uses_llm = any(METHODS[m]["kind"] == "llm" for m in methods)
    if uses_llm and args.llm == "real" and not llm_cfg.get("api_key"):
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

    def client_kind_for(spec):
        if spec["kind"] != "llm":
            return "none"
        return "mock" if (spec.get("force_mock") or args.llm == "mock") else "real"

    def scenario_latency(scenario_id):
        stress = SCENARIOS[scenario_id]["latency"] == "stress"
        return stress_latency if stress else nominal_latency

    def make_client(method_name, spec, eval_seed):
        if client_kind_for(spec) == "mock":
            return MockLLMClient(
                latency_median_ms=nominal_latency, latency_sigma=latency_sigma,
                rng=np.random.default_rng(eval_seed),
            )
        return get_real_client(method_name)

    kinds = {client_kind_for(METHODS[m]) for m in methods} - {"none"}
    run_client = "+".join(sorted(kinds)) if kinds else "none"

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
    append_manifest(
        os.path.join(os.path.dirname(out_path), "manifest.jsonl"),
        collect_manifest(
            find_project_root(), cfg, run_client,
            args={"methods": methods, "scenarios": scenarios, "seeds": seeds,
                  "episodes": args.episodes, "slots": slots, "tau_unc": tau_unc,
                  "theta_queue": theta_queue, "cooldown": cooldown,
                  "ttl_slots": ttl_slots, "random_rate": args.random_rate,
                  "periodic_k": args.periodic_k, "rule_theta": rule_theta,
                  "llm_model": llm_cfg.get("model") if "real" in kinds else None},
            files=["configs/default.yaml", "configs/scenarios.yaml", "configs/llm.yaml"],
        ),
    )
    csv_file = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
    if write_header:
        writer.writeheader()

    # ---- caches
    agents = {}
    infer_ms = {}

    def get_agent(need, seed):
        key = (need["n_critics"], need["suffix"], seed)
        if key not in agents:
            agent = SACAgent(
                obs_dim=17, act_dim=6,
                device="cuda" if _cuda() else "cpu",
                seed=seed, N_critics=need["n_critics"],
                rng=np.random.default_rng(seed),
            )
            agent.load(checkpoint_path(args.ckpt_dir, seed, need["n_critics"], need["suffix"]))
            agents[key] = agent
        return agents[key]

    def actor_policy(agent):
        return lambda obs, info, _a=agent: _a.select_action(obs, deterministic=True)

    total = len(methods) * len(scenarios) * len(seeds) * args.episodes
    n_done = 0
    t_start = time.time()
    print(f"grid: {len(methods)} methods x {len(scenarios)} scenarios x "
          f"{len(seeds)} seeds x {args.episodes} episodes = {total} rows "
          f"(client: {run_client})")

    for method_name in methods:
        spec = METHODS[method_name]
        for scenario_id in scenarios:
            scenario_spec = SCENARIOS[scenario_id]
            for seed in seeds:
                for episode in range(args.episodes):
                    key = (method_name, scenario_id, str(seed), str(episode))
                    if key in done:
                        continue
                    eval_seed = test_seed(seed, episode)
                    env = UAVEnv(
                        config=cfg,
                        scenario=make_scenario(scenario_id, eval_seed),
                        seed=eval_seed,
                    )

                    row = {field: "" for field in FIELDS}
                    row.update({
                        "method": method_name, "method_label": spec["label"],
                        "scenario": scenario_id, "scenario_name": scenario_spec["name"],
                        "seed": seed, "episode": episode, "eval_seed": eval_seed,
                        "slots": slots, "client": client_kind_for(spec),
                    })

                    t0 = time.perf_counter()
                    ctrl = None
                    shield = None
                    bench = None
                    stats_before = stats_after = {}
                    if spec["kind"] == "static":
                        policy = (QueueAwarePriority.from_config(cfg)
                                  if spec.get("factory") == "queue_priority"
                                  else spec["policy"])
                        steps, summary = run_episode(env, policy, seed=eval_seed)
                        bench = ("policy", policy)
                    elif spec["kind"] == "agent":
                        agent = get_agent(agent_spec(spec), seed)
                        steps, summary = run_episode(env, actor_policy(agent), seed=eval_seed)
                        bench = ("agent", agent)
                        row["n_critics"] = spec["n_critics"]
                    elif spec["kind"] in ("rule", "reserve"):
                        agent = get_agent(MASK_N10, seed)
                        base = actor_policy(agent)
                        if spec["kind"] == "rule":
                            fitted = spec["shield_latency"] == "fitted"
                            shield = RuleShield(
                                theta_queue_bits=rule_theta, m=3,
                                ttl_slots=ttl_slots, cooldown=cooldown,
                                base_policy=base,
                                latency_model=(LatencyModel(nominal_latency, latency_sigma)
                                               if fitted else None),
                                slot_ms=slot_ms,
                                rng=np.random.default_rng(eval_seed),
                            )
                        else:
                            shield = FixedReservation(base)
                        steps, summary = run_episode(env, shield, seed=eval_seed)
                        n = max(len(steps), 1)
                        row["mask_active_fraction"] = shield.mask_active_slots / n
                        row["fallback_active_fraction"] = 0.0
                        row["trigger_fraction"] = shield.trigger_count / n
                        row["n_critics"] = 10
                        bench = ("agent", agent)
                    else:  # LLM controller
                        agent = get_agent(MASK_N10, seed)
                        client = make_client(method_name, spec, eval_seed)
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
                            latency_model=LatencyModel(
                                scenario_latency(scenario_id), latency_sigma),
                            use_fitted_latency=True,
                        )
                        is_real = row["client"] == "real"
                        stats_before = client.stats() if is_real else {}
                        steps, summary = ctrl.run_episode(seed=eval_seed)
                        stats_after = client.stats() if is_real else {}
                        row["n_critics"] = 10
                        row["trigger_mode"] = trigger
                        for k in ["mask_active_fraction", "fallback_active_fraction",
                                  "trigger_fraction"]:
                            row[k] = summary.get(k, "")
                        row["diagnosis_accuracy"] = diagnosis_accuracy(
                            ctrl.call_log, scenario_spec["mode"])
                        calls = ctrl.call_log
                        valid_calls = sum(1 for c in calls if c["ok"])
                        timeout_calls = sum(1 for c in calls if c["timeout"])
                        row["llm_calls"] = len(calls)
                        row["llm_call_rate"] = len(calls) / max(len(steps), 1)
                        row["llm_valid"] = valid_calls
                        row["llm_timeouts"] = timeout_calls
                        row["llm_invalid"] = len(calls) - valid_calls - timeout_calls
                        if calls:
                            row["json_valid_rate"] = valid_calls / len(calls)
                            row["timeout_rate"] = timeout_calls / len(calls)
                            row["llm_latency_ms_median"] = float(
                                np.median([c["latency_ms"] for c in calls]))
                            measured = [c["measured_latency_ms"] for c in calls
                                        if math.isfinite(c["measured_latency_ms"])]
                            if measured:
                                row["measured_latency_ms_median"] = float(np.median(measured))
                        if is_real:
                            row["prompt_tokens"] = (stats_after.get("prompt_tokens", 0)
                                                    - stats_before.get("prompt_tokens", 0))
                            row["completion_tokens"] = (
                                stats_after.get("completion_tokens", 0)
                                - stats_before.get("completion_tokens", 0))
                        row["notes"] = f"{row['client']} LLM"
                        bench = ("agent", agent)

                    row["wall_ms_per_slot"] = (time.perf_counter() - t0) * 1000.0 / max(len(steps), 1)
                    if (method_name, seed) not in infer_ms and bench is not None:
                        infer_ms[(method_name, seed)] = (
                            benchmark_agent(bench[1]) if bench[0] == "agent"
                            else benchmark_policy(bench[1]))
                    row["inference_ms_per_slot"] = infer_ms.get((method_name, seed), "")

                    row.update({k: v for k, v in episode_metrics(steps, slot_ms).items()
                                if k in FIELDS})
                    for k in ["mean_reward", "mean_delay_emg_s"]:
                        row[k] = summary.get(k, "")
                    writer.writerow({k: finite_or_blank(v) for k, v in row.items()})
                    csv_file.flush()
                    n_done += 1
                    if n_done % 10 == 0 or n_done == 1:
                        print(f"[{n_done}/{total - len(done)}] {method_name} "
                              f"{scenario_id} seed={seed} ep={episode}  "
                              f"({time.time() - t_start:.0f}s)", flush=True)

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
