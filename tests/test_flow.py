"""Flow-graph tests: the constitutive laws, the graph schema, and the solvers.

Every test compares against an independent authority -- an exact analytic
solution of the same physics (Hagen-Poiseuille, the Windkessel exponential, the
Starling waterfall, the Womersley impedance, or a matrix-exponential reference)
-- rather than against a stored number produced by this code.
"""

import numpy as np
import pytest

from vh.graph import algorithms as A
from vh.graph.flow import (
    G,
    MMHG,
    MU_BLOOD,
    RHO_BLOOD,
    FlowNetwork,
    FlowNodeSpec,
    VesselSpec,
    apply_solution,
    bernoulli_head,
    build_flow_graph,
    extended_bernoulli_residual,
    hydrostatic_head,
    kinetic_pressure_drop,
    lumped_impedance,
    poiseuille_resistance,
    starling_flow,
    tube_compliance,
    tube_inertance,
    womersley_impedance,
    womersley_number,
)


def two_reservoirs(p_up, p_down, **vessel):
    nodes = [
        FlowNodeSpec("up", "reservoir", pressure_pa=p_up),
        FlowNodeSpec("down", "reservoir", pressure_pa=p_down),
    ]
    vessels = [VesselSpec("e", "up", "down", length_m=0.1, radius_m=0.005, **vessel)]
    return build_flow_graph(nodes, vessels)


def solve_steady(nodes, vessels):
    g, schema = build_flow_graph(nodes, vessels)
    assert g.validate(schema) == []
    return FlowNetwork.from_graph(g).steady_state()


# ---------------------------------------------------------------------------
# constitutive laws
# ---------------------------------------------------------------------------
def test_poiseuille_and_inertance_laws():
    # Independent authority: the Hagen-Poiseuille and rho L / A formulas.
    r, l = 4.0e-3, 0.12
    expected_r = 8.0 * MU_BLOOD * l / (np.pi * r ** 4)
    expected_l = RHO_BLOOD * l / (np.pi * r ** 2)
    assert poiseuille_resistance(l, r) == pytest.approx(expected_r, rel=1e-14)
    assert tube_inertance(l, r) == pytest.approx(expected_l, rel=1e-14)

    # A thin-walled cylinder: C = dV/dp = 2 pi r^3 L / (E h).
    c = tube_compliance(l, r, 1.0e-3, 1.0e6)
    assert c == pytest.approx(2.0 * np.pi * r ** 3 * l / (1.0e6 * 1.0e-3), rel=1e-14)


def test_hydrostatic_head_is_78_mmhg_per_metre():
    # 1060 kg/m^3 * 9.80665 m/s^2 * 1 m / 133.322 Pa/mmHg = 78.0 mmHg.
    assert hydrostatic_head(0.0, 1.0) / MMHG == pytest.approx(78.0, abs=0.05)


# ---------------------------------------------------------------------------
# steady flow on a graph
# ---------------------------------------------------------------------------
def test_poiseuille_flow_on_a_graph():
    p_up, resistance = 1000.0, 1.5e6
    sol = solve_steady(
        [FlowNodeSpec("up", "reservoir", pressure_pa=p_up),
         FlowNodeSpec("down", "reservoir", pressure_pa=0.0)],
        [VesselSpec("e", "up", "down", length_m=0.1, radius_m=0.005,
                    resistance_pa_s_m3=resistance, inertance_kg_m4=0.0)],
    )
    assert sol.converged
    assert sol.method == "linear"
    assert sol.flow["e"] == pytest.approx(p_up / resistance, rel=1e-12)
    assert sol.pressure["up"] == pytest.approx(p_up, abs=1e-12)


def test_gravity_drives_flow_down_a_vertical_column():
    # Equal reservoir pressures; only the 1 m elevation difference drives flow.
    r = 1.0e6
    nodes = [
        FlowNodeSpec("top", "reservoir", pressure_pa=0.0, z_m=1.0),
        FlowNodeSpec("bottom", "reservoir", pressure_pa=0.0, z_m=0.0),
    ]
    vessels = [VesselSpec("e", "top", "bottom", length_m=1.0, radius_m=0.005,
                          resistance_pa_s_m3=r, inertance_kg_m4=0.0)]
    sol = solve_steady(nodes, vessels)
    # hydrostatic_head(src, dst) is the pressure *rise* from src to dst, so the
    # driving pressure for src -> dst is -head.
    assert sol.flow["e"] == pytest.approx(-hydrostatic_head(1.0, 0.0) / r, rel=1e-12)
    assert sol.flow["e"] > 0  # downward, from the higher node


def test_bernoulli_stenosis_adds_a_kinetic_pressure_drop():
    # A source injects q into a compartment through a narrowing.  The upstream
    # pressure must supply viscous + Bernoulli kinetic losses.
    q = 2.0e-5
    r1, r2 = 5.0e-3, 2.0e-3
    a1, a2 = np.pi * r1 ** 2, np.pi * r2 ** 2
    resistance = 1.0e6
    sol = solve_steady(
        [FlowNodeSpec("up", "compartment", compliance_m3_pa=1e-9,
                      source_m3_s=q, pressure_pa=0.0),
         FlowNodeSpec("down", "reservoir", pressure_pa=0.0)],
        [VesselSpec("e", "up", "down", length_m=0.1, radius_m=r1,
                    resistance_pa_s_m3=resistance, inertance_kg_m4=0.0,
                    area_src_m2=a1, area_dst_m2=a2)],
    )
    expected = kinetic_pressure_drop(q, a1, a2) + resistance * q
    assert sol.pressure["up"] == pytest.approx(expected, rel=1e-10)
    assert extended_bernoulli_residual(
        sol.pressure["up"], 0.0, q / a1,
        0.0, 0.0, q / a2,
        losses_pa=resistance * q,
    ) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# transients
# ---------------------------------------------------------------------------
def test_windkessel_exact_and_implicit_match_the_analytic_exponential():
    # C dp/dt = source - p/R, p(0)=0  =>  p(t) = q R (1 - exp(-t/RC)).
    c, r, source, t_end = 1.0e-9, 1.0e8, 1.0e-5, 0.5
    nodes = [FlowNodeSpec("a", "compartment", compliance_m3_pa=c, source_m3_s=source,
                          pressure_pa=0.0),
             FlowNodeSpec("v", "reservoir", pressure_pa=0.0)]
    vessels = [VesselSpec("e", "a", "v", length_m=0.1, radius_m=0.005,
                          resistance_pa_s_m3=r, inertance_kg_m4=0.0)]
    net = FlowNetwork.from_specs(nodes, vessels)
    analytic = source * r * (1.0 - np.exp(-t_end / (r * c)))

    exact = net.simulate(t_end, 1.0e-3, initial="zero", exact=True)
    assert exact.pressure[-1, 0] == pytest.approx(analytic, rel=1e-12)

    ie = net.simulate(t_end, 1.0e-3, initial="zero", exact=False)
    assert abs(ie.pressure[-1, 0] - analytic) < 0.25  # < 0.03 % of 993 Pa

    # First-order convergence: halving dt halves the error.
    coarse = net.simulate(t_end, 2.0e-3, initial="zero").pressure[-1, 0]
    fine = net.simulate(t_end, 1.0e-3, initial="zero").pressure[-1, 0]
    assert abs(fine - analytic) < 0.6 * abs(coarse - analytic)


def test_exact_solver_returns_the_steady_flow_when_there_are_no_compartments():
    # A network of reservoirs only has no ODE state; the exact path must still
    # report the instantaneous R flow rather than zeros.
    net = FlowNetwork.from_specs(
        [FlowNodeSpec("up", "reservoir", pressure_pa=1000.0),
         FlowNodeSpec("down", "reservoir", pressure_pa=0.0)],
        [VesselSpec("e", "up", "down", length_m=0.1, radius_m=0.005,
                    resistance_pa_s_m3=1.0e6, inertance_kg_m4=0.0)],
    )
    tr = net.simulate(0.1, 1.0e-3, initial="zero", exact=True)
    assert np.allclose(tr.flow[:, 0], 1.0e-3)


def test_inertance_step_response_matches_the_analytic_R_L_curve():
    # L dq/dt = dp - R q  =>  q(t) = (dp/R)(1 - exp(-R t/L)).
    dp, r, l = 1000.0, 1.0e8, 1.0e6
    tau = l / r
    nodes = [FlowNodeSpec("up", "reservoir", pressure_pa=dp),
             FlowNodeSpec("down", "reservoir", pressure_pa=0.0)]
    vessels = [VesselSpec("e", "up", "down", length_m=0.1, radius_m=0.005,
                          resistance_pa_s_m3=r, inertance_kg_m4=l)]
    net = FlowNetwork.from_specs(nodes, vessels)
    tr = net.simulate(5.0 * tau, 1.0e-5, initial="zero")
    analytic = (dp / r) * (1.0 - np.exp(-5.0))
    assert tr.flow[-1, 0] == pytest.approx(analytic, rel=2e-3)


# ---------------------------------------------------------------------------
# valves and collapsible veins (complementarity)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p_up,p_down,p_ext", [
    (2000.0, 1000.0, 500.0),   # not collapsed: q = (p_up - p_down)/R
    (2000.0, 100.0, 500.0),    # collapsed: q = (p_up - p_ext)/R
    (400.0, 100.0, 500.0),     # inflow below external pressure: closed
    (400.0, 300.0, 500.0),     # downstream below external: still closed
])
def test_starling_waterfall_matches_the_analytic_law(p_up, p_down, p_ext):
    r = 1.0e6
    sol = solve_steady(
        [FlowNodeSpec("up", "reservoir", pressure_pa=p_up),
         FlowNodeSpec("down", "reservoir", pressure_pa=p_down)],
        [VesselSpec("e", "up", "down", length_m=0.1, radius_m=0.005,
                    resistance_pa_s_m3=r, collapsible=True,
                    external_pressure_pa=p_ext)],
    )
    assert sol.converged
    assert sol.method == "active-set"
    assert sol.flow["e"] == pytest.approx(starling_flow(p_up, p_down, p_ext, r), abs=1e-12)


@pytest.mark.parametrize("p_up,p_down", [(2000.0, 1000.0), (1000.0, 2000.0)])
def test_valve_allows_forward_flow_and_blocks_reverse_flow(p_up, p_down):
    sol = solve_steady(
        [FlowNodeSpec("up", "reservoir", pressure_pa=p_up),
         FlowNodeSpec("down", "reservoir", pressure_pa=p_down)],
        [VesselSpec("e", "up", "down", length_m=0.1, radius_m=0.005,
                    resistance_pa_s_m3=1.0e6, valve=True)],
    )
    expected = max(0.0, p_up - p_down) / 1.0e6
    assert sol.flow["e"] == pytest.approx(expected, abs=1e-12)
    assert sol.converged


# ---------------------------------------------------------------------------
# the Womersley reference and the lumped edge law
# ---------------------------------------------------------------------------
def test_womersley_number_scales_with_radius_and_frequency():
    r, omega = 2.0e-3, 2.0 * np.pi
    expected = r * np.sqrt(omega * RHO_BLOOD / MU_BLOOD)
    assert womersley_number(r, omega) == pytest.approx(expected, rel=1e-14)


def test_womersley_impedance_limits():
    l, r = 0.1, 1.0e-3
    # alpha -> 0: the exact impedance tends to the Poiseuille resistance.
    omega_small = 1.0e-4
    z = womersley_impedance(r, l, omega_small)
    assert z.real == pytest.approx(poiseuille_resistance(l, r), rel=1e-10)
    # At this alpha the reactive part is O(alpha^2) smaller than the resistance.
    assert abs(z.imag) / abs(z) < 1e-5

    # alpha -> infinity: inertia dominates, Z -> i omega L.
    omega_large = 2.0 * np.pi * 500.0
    z = womersley_impedance(r, l, omega_large)
    ideal = 1j * omega_large * tube_inertance(l, r)
    assert abs(z - ideal) / abs(ideal) < 0.08
    assert z.imag > 10.0 * z.real


def test_lumped_impedance_is_accurate_at_low_alpha_and_fails_at_high_alpha():
    # Independent authority: the exact Womersley solution of the pulsatile
    # Navier-Stokes equations.  The graph's R + i omega L edge is its low-alpha
    # approximation; this measures where it stops being accurate.
    l, r = 0.1, 1.0e-3
    def error(alpha):
        omega = alpha ** 2 * MU_BLOOD / (r ** 2 * RHO_BLOOD)
        z = womersley_impedance(r, l, omega)
        return abs(z - lumped_impedance(r, l, omega)) / abs(z)
    assert error(0.1) < 1.0e-3
    assert error(1.0) < 0.05
    assert error(5.0) > 0.15  # the lumped law is not a high-alpha model


# ---------------------------------------------------------------------------
# graph algorithms on the flow network
# ---------------------------------------------------------------------------
def test_effective_resistance_reproduces_series_and_parallel_rules():
    g_series, _ = build_flow_graph(
        [FlowNodeSpec("a", "junction"), FlowNodeSpec("b", "junction"),
         FlowNodeSpec("c", "junction")],
        [VesselSpec("e1", "a", "b", length_m=0.1, radius_m=0.005,
                    resistance_pa_s_m3=2.0e6),
         VesselSpec("e2", "b", "c", length_m=0.1, radius_m=0.005,
                    resistance_pa_s_m3=3.0e6)],
    )
    assert A.effective_resistance(g_series, "a", "c") == pytest.approx(5.0e6, rel=1e-9)

    g_par, _ = build_flow_graph(
        [FlowNodeSpec("a", "junction"), FlowNodeSpec("b", "junction")],
        [VesselSpec("e1", "a", "b", length_m=0.1, radius_m=0.005,
                    resistance_pa_s_m3=2.0e6),
         VesselSpec("e2", "a", "b", length_m=0.1, radius_m=0.005,
                    resistance_pa_s_m3=2.0e6)],
    )
    assert A.effective_resistance(g_par, "a", "b") == pytest.approx(1.0e6, rel=1e-9)


def test_effective_resistance_equals_the_solved_network_resistance():
    # Two parallel veins between the same reservoirs.
    nodes = [FlowNodeSpec("a", "reservoir", pressure_pa=1000.0),
             FlowNodeSpec("b", "reservoir", pressure_pa=0.0)]
    vessels = [VesselSpec("e1", "a", "b", length_m=0.1, radius_m=0.004,
                          resistance_pa_s_m3=1.0e6),
               VesselSpec("e2", "a", "b", length_m=0.1, radius_m=0.004,
                          resistance_pa_s_m3=2.0e6)]
    g, schema = build_flow_graph(nodes, vessels)
    assert g.validate(schema) == []
    sol = FlowNetwork.from_graph(g).steady_state()
    r_eff = A.effective_resistance(g, "a", "b")
    assert r_eff == pytest.approx(1000.0 / sum(sol.flow.values()), rel=1e-10)


def test_current_flow_splits_inversely_with_resistance():
    g, _ = build_flow_graph(
        [FlowNodeSpec("a", "junction"), FlowNodeSpec("b", "junction"),
         FlowNodeSpec("c", "junction"), FlowNodeSpec("d", "junction")],
        [VesselSpec("e1", "a", "b", length_m=0.1, radius_m=0.005,
                    resistance_pa_s_m3=1.0e6),
         VesselSpec("e2", "a", "c", length_m=0.1, radius_m=0.005,
                    resistance_pa_s_m3=3.0e6),
         VesselSpec("e3", "b", "d", length_m=0.1, radius_m=0.005,
                    resistance_pa_s_m3=1.0e6),
         VesselSpec("e4", "c", "d", length_m=0.1, radius_m=0.005,
                    resistance_pa_s_m3=3.0e6)],
    )
    current = A.current_flow(g, "a", "d")
    # Unit injection splits between the two source edges.
    assert (current[("a", "b", "conduit")]
            + current[("a", "c", "conduit")]) == pytest.approx(1.0, rel=1e-9)
    # The lower-resistance branch carries 3x the current.
    assert current[("a", "b", "conduit")] == pytest.approx(0.75, rel=1e-9)
    assert current[("a", "c", "conduit")] == pytest.approx(0.25, rel=1e-9)
    # Kirchhoff at the junction b: what enters leaves.
    assert current[("b", "d", "conduit")] == pytest.approx(
        current[("a", "b", "conduit")], rel=1e-9)


# ---------------------------------------------------------------------------
# schema and solution plumbing
# ---------------------------------------------------------------------------
def test_schema_rejects_a_component_without_a_pressure_reference():
    g, schema = build_flow_graph(
        [FlowNodeSpec("a", "compartment", compliance_m3_pa=1e-9),
         FlowNodeSpec("b", "compartment", compliance_m3_pa=1e-9)],
        [VesselSpec("e", "a", "b", length_m=0.1, radius_m=0.005,
                    resistance_pa_s_m3=1.0e6)],
    )
    problems = g.validate(schema)
    assert any("pressure reservoir" in p for p in problems)


def test_schema_checks_gravity_head_consistency_and_positive_resistance():
    nodes = [FlowNodeSpec("a", "reservoir", pressure_pa=0.0, z_m=0.0),
             FlowNodeSpec("b", "reservoir", pressure_pa=0.0, z_m=1.0)]
    vessels = [VesselSpec("e", "a", "b", length_m=1.0, radius_m=0.005,
                          resistance_pa_s_m3=1.0e6)]
    g, schema = build_flow_graph(nodes, vessels)
    assert g.validate(schema) == []
    g.edges[0].attrs["gravity_head_pa"] += 1.0
    assert any("gravity head" in p for p in g.validate(schema))

    g2, schema2 = build_flow_graph(
        [FlowNodeSpec("a", "reservoir", pressure_pa=0.0),
         FlowNodeSpec("b", "reservoir", pressure_pa=0.0)],
        [VesselSpec("e", "a", "b", length_m=0.1, radius_m=0.005,
                    resistance_pa_s_m3=-1.0)],
    )
    assert any("resistance" in p for p in g2.validate(schema2))


def test_apply_solution_writes_flows_that_balance_on_the_graph():
    nodes = [FlowNodeSpec("a", "reservoir", pressure_pa=1000.0),
             FlowNodeSpec("b", "compartment", compliance_m3_pa=1e-9,
                          pressure_pa=0.0),
             FlowNodeSpec("c", "reservoir", pressure_pa=0.0)]
    vessels = [VesselSpec("e1", "a", "b", length_m=0.1, radius_m=0.005,
                          resistance_pa_s_m3=1.0e6),
               VesselSpec("e2", "b", "c", length_m=0.1, radius_m=0.005,
                          resistance_pa_s_m3=2.0e6)]
    g, schema = build_flow_graph(nodes, vessels)
    assert g.validate(schema) == []
    sol = FlowNetwork.from_graph(g).steady_state()
    apply_solution(g, sol)
    assert g.edges[0].attrs["flow"] == pytest.approx(
        float(np.asarray(sol.flow["e1"]) * 60000.0), rel=1e-12)
    bal = A.flow_balance(g, "flow", ("conduit",), sources=("a",), sinks=("c",))
    assert bal["balanced"]


def test_bernoulli_head_is_the_sum_of_its_three_terms():
    p, z, v = 1000.0, 0.5, 0.2
    expected = p + 0.5 * RHO_BLOOD * v ** 2 + RHO_BLOOD * G * z
    assert bernoulli_head(p, z, v) == pytest.approx(expected, rel=1e-14)
