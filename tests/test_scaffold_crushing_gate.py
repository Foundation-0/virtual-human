"""Offline tests for vh.bridge.scaffold_crushing_gate and rational identifiability laws."""

from __future__ import annotations

from fractions import Fraction

from vh.bridge.scaffold_crushing_gate import (
    occupancy_to_scaffold_counts,
    solve_and_audit_scaffold_gate,
    solve_clonal_schedule,
    verify_floating_rs_continuum_identity,
)


def test_occupancy_to_scaffold_counts_low_vs_high_exon_anchor() -> None:
    w_low, pull_low = occupancy_to_scaffold_counts(
        0.22, displacer_cu_nm=15.0, displacer_kd_nm=20.0, total_switches=9
    )
    w_high, pull_high = occupancy_to_scaffold_counts(
        0.85, displacer_cu_nm=15.0, displacer_kd_nm=20.0, total_switches=9
    )
    assert w_low == 2 and pull_low == 3
    assert w_high == 8 and pull_high == 0


def test_scaffold_crushing_gate_phase_transition_and_bellman_audit() -> None:
    sticky = solve_and_audit_scaffold_gate(2, 5, 2, sticky_relaxation=True)
    crushing = solve_and_audit_scaffold_gate(2, 5, 2, sticky_relaxation=False)
    high_wall = solve_and_audit_scaffold_gate(8, 1, 0, sticky_relaxation=False)
    assert sticky["bellman_violations"] == 0 and crushing["bellman_violations"] == 0
    assert sticky["optimal_steps"] == 3 and sticky["survives_true_crushing_gate"] is False
    assert crushing["optimal_steps"] == 8 and crushing["final_walls"] == 9
    assert crushing["survives_true_crushing_gate"] is True and high_wall["optimal_steps"] == 2


def test_clonal_schedule_sticky_mirage_vs_wall_transmutation() -> None:
    assert solve_clonal_schedule(sticky_gate=True, allow_wall_transmutation=False)["optimal_steps"] == 4
    assert solve_clonal_schedule(sticky_gate=False, allow_wall_transmutation=False)["optimal_steps"] is None
    assert solve_clonal_schedule(sticky_gate=False, allow_wall_transmutation=True)["optimal_steps"] == 5


def test_floating_rs_19_continuum_identity_and_remedies() -> None:
    for num in (1, 2, 3, 4, 7, 10, 15, 20):
        cert = verify_floating_rs_continuum_identity(Fraction(num, 20))
        assert cert["gap_matches_factorization"] is True
        assert cert["strictly_below_1_over_60"] is True
        assert cert["sos_identity_holds"] is True and cert["discriminant"] == -10199
    c0 = verify_floating_rs_continuum_identity(Fraction(1, 20))
    assert c0["p_1_25_anchor_eps"] < (1.0 / 60.0) < c0["p_1_26_anchor_eps"]
    assert c0["two_channel_farkas_eps"] == Fraction(19, 240)
