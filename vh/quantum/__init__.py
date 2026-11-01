"""Quantum layer: the accuracy ceiling.

Ab initio electronic structure (Hartree-Fock, DFT, MP2, CCSD, CCSD(T)) for the
smallest real pieces of the body -- the molecules it is built from. CCSD(T) at
the complete-basis-set limit is the "gold standard" of quantum chemistry: the
method against which cheaper methods, force fields and experiments are judged.

This layer is where "molecular accuracy" is defined, not approximated. It runs
on single molecules and small clusters (seconds to hours); `vh.quantum.qmmm`
couples a quantum region to a classical environment so a drug can be treated
quantum-mechanically inside its real surroundings.
"""

from .ab_initio import (  # noqa: F401
    HARTREE_TO_KCAL,
    HARTREE_TO_KJ,
    Molecule,
    cbs_helgaker,
    interaction_energy,
    optimize_geometry,
    single_point,
)
