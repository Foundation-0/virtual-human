"""Drug descriptions: physicochemical, binding, metabolism and absorption inputs.

Every number carries a source string so a prediction can always be traced back
to where its inputs came from (measured, literature, in vitro, or assumed).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .partition import Ionization


@dataclass
class EnzymeProcess:
    """In vitro liver-microsome kinetics, scaled to the whole liver (IVIVE)."""
    enzyme: str
    vmax_pmol_min_mg: float       # per mg microsomal protein
    km_um: float                  # Michaelis constant, uM
    fu_inc: float = 1.0           # unbound fraction in the incubation
    source: str = ""

    @property
    def clint_ul_min_mg(self) -> float:
        """Unbound intrinsic clearance in the linear range, uL/min/mg protein."""
        return self.vmax_pmol_min_mg / self.km_um / self.fu_inc


@dataclass
class Target:
    """A molecular target the drug binds in a tissue (for the molecular -> body bridge)."""
    name: str
    tissue: str
    kd_nm: float
    source: str = ""
    effect_label: str = "occupancy"


@dataclass
class Compound:
    name: str
    mw: float                     # g/mol
    logp: float                   # octanol:water
    ionization: Ionization
    fu: float                     # unbound fraction in plasma
    bp: float | None = None       # blood:plasma ratio (None = predict)
    enzymes: list = field(default_factory=list)   # EnzymeProcess
    clint_hepatic_l_h: float | None = None        # direct unbound hepatic CLint (overrides enzymes)
    renal_filtration: float = 1.0                 # multiple of fu*GFR (1 = filtration only)
    peff_1e4_cm_s: float | None = None            # human jejunal effective permeability, 1e-4 cm/s
    gut_enzymes: tuple = ("CYP3A4",)              # enzymes active in enterocytes (Qgut model)
    target: Target | None = None
    sources: dict = field(default_factory=dict)
    notes: str = ""


# --------------------------------------------------------------------------
# Library
# --------------------------------------------------------------------------
_OSP_MDZ = "Open Systems Pharmacology Midazolam-Model (github.com/Open-Systems-Pharmacology/Midazolam-Model)"

MIDAZOLAM = Compound(
    name="midazolam",
    mw=325.78,
    logp=2.897,
    ionization=Ionization("base", 6.2),
    fu=0.031,
    bp=None,
    enzymes=[
        EnzymeProcess("CYP3A4", 850.0, 4.0, source=f"in vitro HLM Vmax and Km as entered in {_OSP_MDZ}"),
        EnzymeProcess("UGT1A4", 276.0, 37.8, source="Klieber 2008, via " + _OSP_MDZ),
    ],
    renal_filtration=1.0,
    peff_1e4_cm_s=4.0,
    target=Target("GABA-A receptor (benzodiazepine site, GABRG2)", "brain", 1.8,
                  source="Buhr 1997 (Kd 1.8 nM), via " + _OSP_MDZ, effect_label="GABA-A occupancy"),
    sources={
        "mw": _OSP_MDZ,
        "logp": "lipophilicity 2.897 (fitted value in " + _OSP_MDZ + ")",
        "pka": "basic pKa 6.2 (" + _OSP_MDZ + "; acidic pKa 10.95 ignored at physiological pH)",
        "fu": "0.031, Gertz 2010, via " + _OSP_MDZ,
        "bp": "predicted from red-cell partitioning (Rodgers & Rowland); literature ~0.6",
        "peff": "ASSUMED 4e-4 cm/s (BCS class I high permeability; not measured)",
        "gut_metabolism": "Qgut model (Yang 2007) with intestinal CYP3A4 70.5 nmol vs hepatic 137 pmol/mg",
    },
    notes="CYP3A4 probe substrate; extensive hepatic and intestinal first-pass metabolism.",
)

LIBRARY = {"midazolam": MIDAZOLAM}
