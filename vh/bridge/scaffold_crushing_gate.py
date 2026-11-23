"""Molecular -> cellular scaffold crushing-gate dynamics and rational identifiability bounds.

Research simulation module only; ingests zero clinical patient records and makes
zero clinical treatment claims.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ScaffoldGateState:
    walls: int
    mobile_sw: int
    reserve_mob: int
    active_hold: int
    crossed: bool
    crushed: bool


def occupancy_to_scaffold_counts(
    exon_anchor_fraction: float,
    displacer_cu_nm: float,
    displacer_kd_nm: float,
    total_switches: int = 9,
) -> Tuple[int, int]:
    if not (0.0 <= exon_anchor_fraction <= 1.0):
        raise ValueError("exon_anchor_fraction must lie in [0, 1]")
    if displacer_cu_nm < 0.0 or displacer_kd_nm <= 0.0:
        raise ValueError("displacer_cu_nm >= 0 and displacer_kd_nm > 0 required")
    walls = int(np.clip(np.rint(total_switches * exon_anchor_fraction), 0, total_switches))
    occ = float(displacer_cu_nm / (displacer_cu_nm + displacer_kd_nm))
    pull_aura = int(np.clip(np.rint((total_switches - walls) * occ), 0, total_switches))
    return walls, pull_aura


def step_scaffold_gate(
    state: ScaffoldGateState,
    action: str,
    *,
    switch_req: int = 9,
    pull_aura: int = 3,
    sticky_relaxation: bool = False,
) -> Optional[ScaffoldGateState]:
    if state.crushed or state.crossed:
        return None
    w, ms, mr, h = state.walls, state.mobile_sw, state.reserve_mob, state.active_hold
    if action == "engage_hold":
        return ScaffoldGateState(w, ms, mr, h + 1, False, False) if h < 2 else None
    if action == "lock_on_switch":
        return ScaffoldGateState(w + 1, ms - 1, mr, h, False, False) if (ms > 0 and w < switch_req) else None
    if action == "deploy_and_lock_reserve":
        return ScaffoldGateState(w + 1, ms, mr - 1, h, False, False) if (mr > 0 and w + ms < switch_req) else None
    if action == "cross_gate":
        if (w + ms + h) < switch_req:
            return None
        ms_after = max(0, ms - pull_aura)
        if sticky_relaxation or (w + ms_after) >= switch_req:
            return ScaffoldGateState(w, ms_after, mr, 0, True, False)
        return ScaffoldGateState(w, ms_after, mr, 0, False, True)
    raise ValueError(f"Unknown action: {action}")


def solve_and_audit_scaffold_gate(
    initial_walls: int,
    initial_mobile_sw: int,
    initial_reserve: int,
    *,
    switch_req: int = 9,
    pull_aura: int = 3,
    sticky_relaxation: bool = False,
) -> Dict[str, object]:
    start = ScaffoldGateState(initial_walls, initial_mobile_sw, initial_reserve, 0, False, False)
    dist: Dict[ScaffoldGateState, int] = {start: 0}
    parent: Dict[ScaffoldGateState, Tuple[ScaffoldGateState, str]] = {}
    q: deque[ScaffoldGateState] = deque([start])
    actions = ("engage_hold", "lock_on_switch", "deploy_and_lock_reserve", "cross_gate")
    best_goal: Optional[ScaffoldGateState] = None

    while q:
        cur = q.popleft()
        d = dist[cur]
        if cur.crossed and not cur.crushed and best_goal is None:
            best_goal = cur
        for act in actions:
            nxt = step_scaffold_gate(
                cur, act, switch_req=switch_req, pull_aura=pull_aura, sticky_relaxation=sticky_relaxation
            )
            if nxt is not None and nxt not in dist:
                dist[nxt] = d + 1
                parent[nxt] = (cur, act)
                q.append(nxt)

    bellman_violations = 0
    edges_audited = 0
    for u, du in dist.items():
        for act in actions:
            v = step_scaffold_gate(
                u, act, switch_req=switch_req, pull_aura=pull_aura, sticky_relaxation=sticky_relaxation
            )
            if v is not None:
                edges_audited += 1
                if dist[v] > du + 1:
                    bellman_violations += 1

    trace: List[str] = []
    if best_goal is not None:
        cur = best_goal
        while cur in parent:
            prev, act = parent[cur]
            trace.append(act)
            cur = prev
        trace.reverse()

    survives_true_gate = False
    crushed_at_step: Optional[int] = None
    if best_goal is not None:
        sim = start
        for idx, act in enumerate(trace, start=1):
            nxt = step_scaffold_gate(
                sim, act, switch_req=switch_req, pull_aura=pull_aura, sticky_relaxation=False
            )
            assert nxt is not None
            sim = nxt
            if sim.crushed:
                crushed_at_step = idx
                break
        survives_true_gate = sim.crossed and not sim.crushed

    return {
        "initial_walls": initial_walls,
        "sticky_relaxation": sticky_relaxation,
        "reachable_states": len(dist),
        "edges_audited": edges_audited,
        "bellman_violations": bellman_violations,
        "optimal_steps": dist[best_goal] if best_goal is not None else None,
        "final_walls": best_goal.walls if best_goal is not None else None,
        "optimal_trace": trace,
        "survives_true_crushing_gate": survives_true_gate,
        "crushed_at_step_on_true_gate": crushed_at_step,
    }


@dataclass(frozen=True)
class ClonalGateState:
    v_s: int
    v_r: int
    w_mhc: int
    crushed: bool


def solve_clonal_schedule(
    *,
    sticky_gate: bool,
    allow_wall_transmutation: bool,
    mhc_switch_req: int = 7,
    v_crit: int = 4,
) -> Dict[str, object]:
    start = ClonalGateState(v_s=6, v_r=2, w_mhc=1, crushed=False)
    dist: Dict[ClonalGateState, int] = {start: 0}
    parent: Dict[ClonalGateState, Tuple[ClonalGateState, str]] = {}
    q: deque[ClonalGateState] = deque([start])
    actions = ["cytotoxic_pulse"] + (
        ["lock_mhc_wall", "resensitized_clearance"] if allow_wall_transmutation else []
    )

    def step(u: ClonalGateState, act: str) -> Optional[ClonalGateState]:
        if u.crushed or (u.v_s == 0 and u.v_r == 0):
            return None
        vs, vr, wm = u.v_s, u.v_r, u.w_mhc
        if act == "cytotoxic_pulse":
            vs_n, vr_n, wm_n = max(0, vs - 3), (max(0, vr - 1) if sticky_gate and vs == 0 else vr), wm
        elif act == "lock_mhc_wall":
            if wm >= mhc_switch_req:
                return None
            vs_n, vr_n, wm_n = vs, vr, wm + 2
        elif act == "resensitized_clearance":
            if wm < mhc_switch_req:
                return None
            vs_n, vr_n, wm_n = max(0, vs - 3), max(0, vr - 2), wm
        else:
            return None
        if not sticky_gate and vr_n > 0 and (vs_n + wm_n) < mhc_switch_req:
            return ClonalGateState(vs_n, v_crit, wm_n, True)
        return ClonalGateState(vs_n, vr_n, wm_n, vr_n >= v_crit)

    best_goal: Optional[ClonalGateState] = None
    while q:
        cur = q.popleft()
        d = dist[cur]
        if not cur.crushed and cur.v_s == 0 and cur.v_r == 0 and best_goal is None:
            best_goal = cur
        for act in actions:
            nxt = step(cur, act)
            if nxt is not None and nxt not in dist:
                dist[nxt] = d + 1
                parent[nxt] = (cur, act)
                q.append(nxt)

    bellman_violations = sum(
        1 for u, du in dist.items() for act in actions if (v := step(u, act)) is not None and dist[v] > du + 1
    )
    edges_audited = sum(1 for u in dist for act in actions if step(u, act) is not None)
    trace: List[str] = []
    if best_goal is not None:
        cur = best_goal
        while cur in parent:
            prev, act = parent[cur]
            trace.append(act)
            cur = prev
        trace.reverse()

    return {
        "sticky_gate": sticky_gate,
        "allow_wall_transmutation": allow_wall_transmutation,
        "reachable_states": len(dist),
        "edges_audited": edges_audited,
        "bellman_violations": bellman_violations,
        "optimal_steps": dist[best_goal] if best_goal is not None else None,
        "optimal_trace": trace,
    }


def verify_floating_rs_continuum_identity(p_seed: Fraction = Fraction(1, 20)) -> Dict[str, object]:
    if not (Fraction(1, 20) <= p_seed <= Fraction(1, 1)):
        raise ValueError("p_seed must lie in [1/20, 1]")
    q_r19 = (19 * p_seed) / (18 * p_seed + 1)
    q_c25 = (11 * p_seed + Fraction(1, 4) * (1 - p_seed)) / (10 * p_seed + 1)
    factored_gap = (14 * p_seed - 1) * (1 - p_seed) / (4 * (18 * p_seed + 1) * (10 * p_seed + 1))
    poly_val = 570 * p_seed * p_seed - 169 * p_seed + 17
    sos_lhs = 2280 * poly_val
    sos_rhs = (1140 * p_seed - 169) * (1140 * p_seed - 169) + 10199
    u_c25 = (Fraction(1, 4) * (1 - p_seed)) / (10 * p_seed + 1)

    def anchor_minimax(p_low: Fraction) -> float:
        k1, k0 = float(p_low / (1 - p_low)), 0.25
        y1 = float((11 * p_low + Fraction(1, 4) * (1 - p_low)) / (10 * p_low + 1))
        s_val = y1 + 0.8
        a_c, b_c, c_c = (2.0 - s_val) * k1 * k0, (1.0 - s_val) * (k1 + k0), -s_val
        rs_star = (-b_c + math.sqrt(b_c * b_c - 4.0 * a_c * c_c)) / (2.0 * a_c)
        return (y1 - (k1 * rs_star / (k1 * rs_star + 1.0))) / 2.0

    return {
        "gap_matches_factorization": (q_r19 - q_c25) == factored_gap,
        "half_gap": abs(q_r19 - q_c25) / 2,
        "strictly_below_1_over_60": abs(q_r19 - q_c25) / 2 < Fraction(1, 60),
        "quadratic_val": poly_val,
        "sos_identity_holds": sos_lhs == sos_rhs and sos_rhs > 0,
        "discriminant": (-169) * (-169) - 4 * 570 * 17,
        "p_1_25_anchor_eps": anchor_minimax(Fraction(1, 25)),
        "p_1_26_anchor_eps": anchor_minimax(Fraction(1, 26)),
        "two_channel_farkas_eps": u_c25 / 2,
    }
