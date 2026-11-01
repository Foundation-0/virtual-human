"""Population variability in drug exposure.

Simulates midazolam pharmacokinetics across a virtual adult population and
reports the spread in exposure (AUC) and peak concentration. This illustrates
why the same dose produces different blood levels in different people: body
size, organ blood flow, and enzyme activity all vary.

    python examples/population_exposure.py
"""

import numpy as np

from vh.body import MIDAZOLAM, Dose, PBPKModel, reference_individual, sample_population
from vh.bridge import target_engagement


def exposure(model, dose):
    sim = model.simulate([dose], t_end_h=24, dt_h=0.05)
    auc = float(np.trapezoid(sim["plasma"], sim["time_h"]))     # mg*h/L
    cmax = float(sim["plasma"].max())
    occ = target_engagement(model, sim)["occupancy"].max()      # peak GABA-A occupancy
    return auc, cmax, occ


def main():
    dose = Dose(0.0, 7.5, "oral")
    ref = PBPKModel(MIDAZOLAM, reference_individual("male"))
    a0, c0, o0 = exposure(ref, dose)
    print("Reference man, 7.5 mg oral midazolam:")
    print(f"  AUC {a0:.3f} mg*h/L,  Cmax {c0 * 1000:.1f} ng/mL,  peak receptor occupancy {o0 * 100:.0f}%")
    print(f"  {ref.summary()}\n")

    pop = sample_population(100, seed=0)
    rows = [exposure(PBPKModel(MIDAZOLAM, ind), dose) for ind in pop]
    auc = np.array([r[0] for r in rows])
    print(f"Virtual population (n={len(pop)}), same 7.5 mg oral dose:")
    print(f"  AUC   median {np.median(auc):.3f}, 5th-95th percentile {np.percentile(auc, 5):.3f}-{np.percentile(auc, 95):.3f} mg*h/L")
    print(f"  spread (max/min) {auc.max() / auc.min():.1f}-fold  —  one dose does not give one blood level")


if __name__ == "__main__":
    main()
