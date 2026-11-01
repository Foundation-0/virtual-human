"""Command line: simulate a protein at atomic resolution.

    python -m vh.molecular.run data/structures/1uao_chignolin.pdb --ps 50 --temp 300

Writes <out>.pdb (multi-model trajectory), <out>_energy.csv and prints a
summary. Open the PDB in PyMOL, ChimeraX or VMD to watch the motion.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description="Atom-level protein simulation (virtual-human molecular engine)")
    p.add_argument("pdb", help="input PDB file")
    p.add_argument("--ps", type=float, default=20.0, help="simulated time in picoseconds (default 20)")
    p.add_argument("--temp", type=float, default=300.0, help="temperature in kelvin (default 300)")
    p.add_argument("--dt", type=float, default=1.0, help="time step in femtoseconds (default 1.0)")
    p.add_argument("--friction", type=float, default=1.0, help="Langevin friction, 1/ps (default 1)")
    p.add_argument("--frame-every", type=float, default=0.5, help="save a frame every N ps (default 0.5)")
    p.add_argument("--precision", choices=["single", "double"], default="single")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="output prefix (default: <pdb name>_traj)")
    a = p.parse_args(argv)

    from . import LangevinBAOAB, build_energy_model, minimize, simulate
    from .io import write_energy_log, write_pdb_trajectory
    from .prep import load_protein

    out = a.out or (Path(a.pdb).stem + "_traj")
    t0 = time.time()
    system, pos, _, _ = load_protein(a.pdb)
    print(system.summary())
    dtype = jnp.float32 if a.precision == "single" else jnp.float64
    m64 = build_energy_model(system, dtype=jnp.float64)
    x, info = minimize(m64, pos)
    print(f"minimized: E = {info['energy']:.1f} kJ/mol, RMS force {info['rms_force']:.2f} kJ/mol/nm ({info['iterations']} iterations)")

    model = m64 if dtype == jnp.float64 else build_energy_model(system, dtype=dtype)
    dt_ps = a.dt * 1e-3
    integ = LangevinBAOAB(model, dt=dt_ps, temperature=a.temp, friction=a.friction)
    state = integ.init_state(x, T=a.temp, seed=a.seed)
    n_steps = int(round(a.ps / dt_ps))
    every = max(1, int(round(a.frame_every / dt_ps)))
    integ.run(state, 1)  # compile once (not timed)
    t_sim = time.time()

    def progress(st, log):
        print(f"  t = {st.time:8.2f} ps   E_pot = {log['e_pot'][-1]:10.1f} kJ/mol   T = {log['T'][-1]:6.1f} K", flush=True)

    state, frames, log = simulate(integ, state, n_steps, every, callback=progress)
    wall = time.time() - t_sim
    write_pdb_trajectory(out + ".pdb", system, np.concatenate([x[None], frames]), np.concatenate([[0.0], log["time_ps"]]))
    write_energy_log(out + "_energy.csv", log)
    print(f"done: {a.ps} ps in {wall:.1f} s  ->  {a.ps / 1000 / (wall / 86400):.1f} ns/day "
          f"({n_steps / wall:.0f} steps/s); <T> = {np.mean(log['T'][len(log['T']) // 5:]):.1f} K")
    print(f"wrote {out}.pdb ({len(frames) + 1} frames) and {out}_energy.csv  [total {time.time() - t0:.1f} s]")


if __name__ == "__main__":
    main()
