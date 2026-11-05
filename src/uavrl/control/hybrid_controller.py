"""Hybrid controller: async LLM arbiter with safety layer (Phase 7).

Per-slot loop (simulated time; the fast loop never waits on the LLM):

    each slot t:
      1. apply LLM responses with arrival_slot <= t:
            validate -> install Mask(expires_at = arrival + ttl)
            invalid / timed-out -> activate Fallback for fallback_slots
      2. a_raw = actor(s); U = ensemble_std(s, a_raw)
      3. a = project(a_raw, active_mask or fallback or None)
      4. s', info = env.step(a)
      5. trigger = debounce(U > tau) OR (Q_emg > theta_queue);
         if trigger and cooldown_ok:
             prompt = build_prompt(features)
             resp, L, ok, reason = llm.call(prompt, features)
             arrival_slot = t + ceil(L / slot_ms)
             if L > timeout or not ok:
                 schedule fallback at t + ceil(timeout / slot_ms)
             else:
                 store pending response keyed by arrival_slot
      6. expire masks past expires_at

Trigger modes: uncertainty, queue, hybrid, periodic(K), reactive, always,
random(rate).  Counts: calls, valid / invalid / timeouts, mask-active
fraction, fallback-active fraction.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..control.fallback import FALLBACK_BW, FALLBACK_POWER
from ..control.mask import Mask, project_bw_power
from ..llm.client import LLMClient
from ..llm.latency import LatencyModel
from ..llm.prompt import build_prompt
from ..llm.schema import LLMResponse

WINDOW = 50  # sliding window for CSI statistics / violation fraction


def _softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _inv_softmax(shares):
    shares = np.clip(np.asarray(shares, dtype=float), 1e-9, 1.0)
    ls = np.log(shares)
    return (ls - ls.mean()) / 3.0


class HybridController:
    def __init__(
        self,
        agent,
        env,
        llm_client: LLMClient,
        cfg: dict,
        trigger_mode: str = "hybrid",
        tau_unc: Optional[float] = None,
        theta_queue: Optional[float] = None,
        debounce_m: int = 3,
        cooldown: int = 50,
        ttl_slots: int = 600,
        timeout_ms: int = 1000,
        fallback_slots: int = 200,
        periodic_K: int = 20,
        random_rate: float = 0.05,
        interim_fallback: bool = False,
        rng=None,
        latency_model: Optional[LatencyModel] = None,
        use_fitted_latency: bool = False,
    ):
        self.agent = agent
        self.env = env
        self.llm_client = llm_client
        self.cfg = cfg
        self.trigger_mode = trigger_mode
        self.tau_unc = tau_unc
        self.theta_queue = theta_queue
        self.debounce_m = debounce_m
        self.cooldown = cooldown
        self.ttl_slots = ttl_slots
        self.timeout_ms = timeout_ms
        self.fallback_slots = fallback_slots
        self.periodic_K = periodic_K
        self.random_rate = random_rate
        self.interim_fallback = interim_fallback
        self.rng = np.random.default_rng() if rng is None else rng

        self.slot_ms = float(cfg.get("timing", {}).get("slot_ms", 5.0))
        timing = cfg.get("timing", {})
        # F13: scheduling latency comes from the fitted model whenever the
        # client reports a cache hit (0 ms) or use_fitted_latency is set.
        self.latency_model = latency_model or LatencyModel(
            median_ms=float(timing.get("llm_latency_median_ms", 350.0)),
            sigma=float(timing.get("llm_latency_sigma", 0.6)),
        )
        self.use_fitted_latency = bool(use_fitted_latency)
        self.fallback_bw = FALLBACK_BW
        self.fallback_power = FALLBACK_POWER

        # per-episode state (reset in run_episode)
        self.active_mask = None
        self.fallback_until = -1
        self.pending_responses = []   # {arrival_slot, response}
        self.pending_fallbacks = []   # slots at which a failure becomes known (F5)
        self.csi_ref = np.ones(3)
        self.consec_unc = 0
        self.cooldown_counter = 0
        self.slot = -1
        self.csi_hist = []            # list of gain vectors
        self.prev_q_emg = 0.0
        self.violation_seen = False   # for reactive mode

        # counters
        self.calls = 0
        self.valid = 0
        self.invalid = 0
        self.timeouts = 0
        self.cache_hits = 0
        self.mask_active_slots = 0
        self.fallback_active_slots = 0
        self.call_log = []            # one dict per LLM call (Phase 9 metrics)

    # ------------------------------------------------------------------ reset
    def reset(self):
        self.active_mask = None
        self.fallback_until = -1
        self.pending_responses = []
        self.pending_fallbacks = []
        self.consec_unc = 0
        self.cooldown_counter = 0
        self.slot = -1
        self.csi_hist = []
        self.prev_q_emg = 0.0
        self.violation_seen = False
        self.calls = 0
        self.valid = 0
        self.invalid = 0
        self.timeouts = 0
        self.cache_hits = 0
        self.mask_active_slots = 0
        self.fallback_active_slots = 0
        self.call_log = []

    # ------------------------------------------------------------------ features
    def _build_features(self, obs, info, trigger_reason):
        # csi_hist holds the last WINDOW slots, normalized so nominal is ~1.0
        # (F6: updated every slot; F21: normalized by the reference gain).
        hist = np.array(self.csi_hist)  # (T, 3)
        feats = {"slot": self.slot, "trigger_reason": trigger_reason}
        for j, cls in enumerate(["emergency", "important", "normal"]):
            col = hist[:, j]
            feats[f"csi_{cls}_mean"] = float(col.mean())
            feats[f"csi_{cls}_var"] = float(col.var())
            feats[f"csi_{cls}_slope"] = float(col[-1] - col[0]) if len(col) > 1 else 0.0
        q = np.asarray(info.get("queue_bits", np.zeros(3)), dtype=float)
        feats["queue_emg_bits"] = float(q[0])
        feats["queue_imp_bits"] = float(q[1])
        feats["queue_norm_bits"] = float(q[2])
        feats["dQ_emg"] = float(q[0] - self.prev_q_emg)
        u, _ = self._last_u
        feats["U_t"] = float(u)
        return feats

    # ------------------------------------------------------------------ run one episode
    def run_episode(self, seed=None, deterministic_actor=True):
        obs, info = self.env.reset(seed=seed)
        self.reset()
        self.csi_ref = np.maximum(
            np.asarray(self.env.channel.reference_gain(), dtype=float), 1e-30
        )
        steps = []
        done = False
        self._last_u = (0.0, 0.0)

        while not done:
            self.slot += 1
            t = self.slot

            # ---- 1. apply LLM responses arriving this slot
            still_pending = []
            for resp in self.pending_responses:
                if resp["arrival_slot"] <= t:
                    try:
                        parsed = LLMResponse(**resp["response"])
                        mp = parsed.to_mask()
                        if parsed.operating_mode == "NORMAL":
                            # NORMAL releases the shield instead of installing
                            # the model's floors (F18).
                            self.active_mask = None
                            self.valid += 1
                            continue
                        mask = Mask(
                            bw_floor=np.array(mp["bw_floor"], dtype=float),
                            power_floor_emg=float(mp["power_floor_emg"]),
                            expires_at=t + int(mp["ttl_slots"]),
                        )
                        if mask.valid():
                            self.active_mask = mask
                            self.valid += 1
                        else:
                            raise ValueError("invalid mask params")
                    except Exception:
                        self.active_mask = None
                        self.fallback_until = t + self.fallback_slots
                        self.invalid += 1
                else:
                    still_pending.append(resp)
            self.pending_responses = still_pending

            # ---- failures become known only now (F5): start their fallback
            remaining = []
            for start in self.pending_fallbacks:
                if start <= t:
                    self.fallback_until = max(self.fallback_until, start + self.fallback_slots)
                else:
                    remaining.append(start)
            self.pending_fallbacks = remaining

            # ---- expire active mask
            if self.active_mask is not None and self.active_mask.expired(t):
                self.active_mask = None

            # ---- cooldown countdown
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1

            # ---- 2. actor + uncertainty
            a_raw = self.agent.select_action(obs, deterministic=deterministic_actor)
            u, mean_q = self.agent.uncertainty(obs, a_raw)
            self._last_u = (u, mean_q)

            # ---- fallback vs mask vs plain
            fallback_active = self.fallback_until > t
            mask_active = self.active_mask is not None and not fallback_active

            if fallback_active:
                b_share = self.fallback_bw.copy()
                p_share = self.fallback_power.copy()
            elif mask_active:
                b_raw_s = _softmax(3.0 * a_raw[:3])
                p_raw_s = _softmax(3.0 * a_raw[3:])
                b_share, p_share = project_bw_power(b_raw_s, p_raw_s, self.active_mask)
            else:
                b_share = _softmax(3.0 * a_raw[:3])
                p_share = _softmax(3.0 * a_raw[3:])

            action = np.concatenate([_inv_softmax(b_share), _inv_softmax(p_share)]).astype(np.float32)

            # ---- 4. env step
            obs, reward, terminated, truncated, info = self.env.step(action)

            # ---- CSI history: every slot, normalized by the reference gain
            gain = np.asarray(info.get("gain", np.zeros(3)), dtype=float)
            self.csi_hist.append(gain / self.csi_ref)
            if len(self.csi_hist) > WINDOW:
                self.csi_hist.pop(0)

            # ---- 5. trigger
            q_emg = float(info.get("queue_bits", np.zeros(3))[0])
            trig_unc = False
            if self.tau_unc is not None:
                if u > self.tau_unc:
                    self.consec_unc += 1
                else:
                    self.consec_unc = 0
                trig_unc = self.consec_unc >= self.debounce_m

            trig_queue = self.theta_queue is not None and q_emg > self.theta_queue

            if self.trigger_mode == "uncertainty":
                trigger = trig_unc
            elif self.trigger_mode == "queue":
                trigger = trig_queue
            elif self.trigger_mode == "hybrid":
                trigger = trig_unc or trig_queue
            elif self.trigger_mode == "periodic":
                trigger = (t % self.periodic_K == 0)
            elif self.trigger_mode == "reactive":
                trigger = self.violation_seen
            elif self.trigger_mode == "always":
                trigger = True
            elif self.trigger_mode == "random":
                trigger = self.rng.random() < self.random_rate
            else:
                trigger = False

            llm_called_this_slot = False
            if trigger and self.cooldown_counter == 0:
                self.calls += 1
                llm_called_this_slot = True
                features = self._build_features(obs, info,
                                                "uncertainty" if trig_unc else
                                                "queue" if trig_queue else "mixed")
                cur_mask = None
                if self.active_mask is not None:
                    cur_mask = {"bw_floor": self.active_mask.bw_floor.tolist(),
                                "power_floor_emg": self.active_mask.power_floor_emg}
                prompt = build_prompt(features, current_mask=cur_mask)
                resp, L_meas, ok, reason = self.llm_client.call(prompt, features)
                if self.use_fitted_latency or reason == "cache":
                    L = self.latency_model.sample(self.rng)
                else:
                    L = float(L_meas)
                is_timeout = bool(reason == "timeout" or L > self.timeout_ms)
                mode = resp.get("operating_mode") if ok and isinstance(resp, dict) else None
                self.call_log.append({
                    "slot": t,
                    "ood": bool(info.get("ood_flag", False)),
                    "trigger_reason": features["trigger_reason"],
                    "ok": bool(ok and not is_timeout),
                    "timeout": is_timeout,
                    "reason": reason,
                    "mode": mode,
                    "latency_ms": float(L),
                    "measured_latency_ms": (
                        float("nan") if reason == "cache" else float(L_meas)
                    ),
                })
                if is_timeout:
                    # The caller only learns of the timeout after timeout_ms.
                    known_slot = t + math.ceil(self.timeout_ms / self.slot_ms)
                    self.timeouts += 1
                    self.pending_fallbacks.append(known_slot)
                elif not ok:
                    # An invalid reply is known when it arrives.
                    known_slot = t + math.ceil(L / self.slot_ms)
                    self.invalid += 1
                    self.pending_fallbacks.append(known_slot)
                else:
                    known_slot = t + math.ceil(L / self.slot_ms)
                    self.pending_responses.append(
                        {"arrival_slot": known_slot, "response": resp}
                    )
                if self.interim_fallback:
                    # M12b: safe allocation while the answer is in flight.
                    self.fallback_until = max(self.fallback_until, known_slot)
                self.cooldown_counter = self.cooldown

            # ---- 6. masks naturally expire (handled at top of next loop)
            self.prev_q_emg = q_emg
            if info.get("emergency_violation", False):
                self.violation_seen = True

            steps.append({
                "slot": t,
                "reward": float(reward),
                "emergency_violation": bool(info.get("emergency_violation", False)),
                "ood_flag": bool(info.get("ood_flag", False)),
                "delay_emg_s": float(info["delay_s"][0]),
                "delay_imp_s": float(info["delay_s"][1]),
                "delay_norm_s": float(info["delay_s"][2]),
                "rate_emg_bps": float(info["rate_bps"][0]),
                "rate_imp_bps": float(info["rate_bps"][1]),
                "rate_norm_bps": float(info["rate_bps"][2]),
                "queue_emg_bits": float(info["queue_bits"][0]),
                "queue_imp_bits": float(info["queue_bits"][1]),
                "queue_norm_bits": float(info["queue_bits"][2]),
                "drops_emg_bits": float(info["drops_bits"][0]),
                "drops_imp_bits": float(info["drops_bits"][1]),
                "drops_norm_bits": float(info["drops_bits"][2]),
                "served_emg_bits": float(info["served_bits"][0]),
                "served_imp_bits": float(info["served_bits"][1]),
                "served_norm_bits": float(info["served_bits"][2]),
                "arrival_emg_bits": float(info["arrival_bits"][0]),
                "b_sum": float(info["b_sum"]),
                "p_sum": float(info["p_sum"]),
                "b_share_emg": float(info["b_share"][0]),
                "p_share_emg": float(info["p_share"][0]),
                "r_emg": float(info["reward_terms"]["r_emg"]),
                "r_imp": float(info["reward_terms"]["r_imp"]),
                "r_norm": float(info["reward_terms"]["r_norm"]),
                "penalty": float(info["reward_terms"]["penalty"]),
                "U_t": float(self._last_u[0]),
                "meanQ": float(self._last_u[1]),
                "trigger": bool(trigger),
                "llm_call": bool(llm_called_this_slot),
                "mask_active": bool(mask_active),
                "fallback_active": bool(fallback_active),
                "gain_emg": float(info["gain"][0]),
            })
            done = terminated or truncated

        summary = self._summarize(steps)
        return steps, summary

    def _summarize(self, steps):
        if not steps:
            return {}
        n = len(steps)
        arr = {k: np.array([s[k] for s in steps]) for k in steps[0]}
        sos = arr["emergency_violation"]
        return {
            "slots": int(n),
            "mean_reward": float(arr["reward"].mean()),
            "sos_violation_rate": float(sos.mean()),
            "sos_p99_delay_s": float(np.percentile(arr["delay_emg_s"], 99)),
            "mean_delay_emg_s": float(arr["delay_emg_s"].mean()),
            "mean_rate_emg_bps": float(arr["rate_emg_bps"].mean()),
            "mean_rate_imp_bps": float(arr["rate_imp_bps"].mean()),
            "mean_rate_norm_bps": float(arr["rate_norm_bps"].mean()),
            "total_drops_emg_bits": float(arr["drops_emg_bits"].sum()),
            "llm_calls": int(self.calls),
            "llm_call_rate": float(self.calls / max(n, 1)),
            "llm_valid": int(self.valid),
            "llm_invalid": int(self.invalid),
            "llm_timeouts": int(self.timeouts),
            "mask_active_fraction": float(np.mean(arr["mask_active"])),
            "fallback_active_fraction": float(np.mean(arr["fallback_active"])),
            "trigger_fraction": float(np.mean(arr["trigger"])),
            "final_battery_fraction": float(arr.get(
                "battery_fraction", np.ones(n))[-1]) if "battery_fraction" in arr else 1.0,
        }

