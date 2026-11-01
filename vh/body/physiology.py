"""Human physiology: organ masses, blood flows, and virtual individuals.

Reference values are the ICRP Publication 89 reference adult male and female
(ICRP 2002, "Basic anatomical and physiological data for use in radiological
protection: reference values"). Tissue masses are used as volumes (1 kg ~ 1 L):
Rodgers & Rowland partition coefficients are defined per gram of tissue, so
amount = Kp * C_plasma * mass is consistent without density corrections.

Model compartments (perfusion-limited, well-stirred):
lung, heart, brain, muscle, adipose, skin, bone, kidney, liver, gut, spleen,
pancreas, rest of body, arterial blood, venous blood. Gut, spleen and pancreas
drain into the liver through the portal vein.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

TISSUES = ("lung", "heart", "brain", "muscle", "adipose", "skin", "bone", "kidney",
           "liver", "gut", "spleen", "pancreas", "rest")
PORTAL = ("gut", "spleen", "pancreas")   # drain into the liver

# ICRP 89 reference adult (masses in kg; flows as % of cardiac output).
# gut = stomach + small intestine + large intestine walls (0.15 + 0.65 + 0.37 kg, male).
# Flow to "rest" collects bronchial, thyroid, lymph nodes, gonads, adrenals,
# bladder and other minor tissues so that flows sum to 100 %.
_ICRP89 = {
    "male": dict(
        body_mass=73.0, height=176.0, cardiac_output_l_min=6.5, hematocrit=0.45, blood_l=5.3,
        mass=dict(lung=0.50, heart=0.33, brain=1.45, muscle=29.0, adipose=18.2, skin=3.3, bone=10.5,
                  kidney=0.31, liver=1.80, gut=1.17, spleen=0.15, pancreas=0.14),
        flow_pct=dict(heart=4.0, brain=12.0, muscle=17.0, adipose=5.0, skin=5.0, bone=5.0,
                      kidney=19.0, liver_arterial=6.5, gut=15.0, spleen=3.0, pancreas=1.0, rest=7.5),
    ),
    "female": dict(
        body_mass=60.0, height=163.0, cardiac_output_l_min=5.9, hematocrit=0.40, blood_l=3.9,
        mass=dict(lung=0.42, heart=0.25, brain=1.30, muscle=17.5, adipose=22.5, skin=2.3, bone=7.8,
                  kidney=0.275, liver=1.40, gut=1.06, spleen=0.13, pancreas=0.12),
        flow_pct=dict(heart=5.0, brain=12.0, muscle=12.0, adipose=8.5, skin=5.0, bone=5.0,
                      kidney=17.0, liver_arterial=6.5, gut=17.0, spleen=3.0, pancreas=1.0, rest=8.0),
    ),
}


@dataclass
class Individual:
    """One virtual human. Volumes in L (= kg), flows in L/h."""

    sex: str
    age: float
    body_mass: float
    height: float
    volumes: dict
    flows: dict                 # per tissue (liver = hepatic artery only)
    cardiac_output: float       # L/h
    hematocrit: float
    v_arterial: float
    v_venous: float
    gfr: float                  # L/h
    mppgl: float = 32.0         # mg microsomal protein per g liver (Barter 2007)
    enzyme_scale: dict = field(default_factory=dict)   # per-enzyme activity multipliers (variability)
    label: str = "reference"

    @property
    def liver_flow_total(self) -> float:
        return self.flows["liver"] + sum(self.flows[t] for t in PORTAL)

    def describe(self) -> str:
        return (f"{self.label}: {self.sex}, {self.age:.0f} y, {self.body_mass:.1f} kg, {self.height:.0f} cm | "
                f"CO {self.cardiac_output / 60:.2f} L/min, liver blood flow {self.liver_flow_total / 60:.2f} L/min, "
                f"GFR {self.gfr / 0.06:.0f} mL/min, adipose {self.volumes['adipose']:.1f} kg")


def _lean_body_weight(sex, mass, height_cm):
    """Janmahasatian et al. 2005 (Clin Pharmacokinet 44:1051)."""
    bmi = mass / (height_cm / 100.0) ** 2
    if sex == "male":
        return 9270.0 * mass / (6680.0 + 216.0 * bmi)
    return 9270.0 * mass / (8780.0 + 244.0 * bmi)


def _bsa(mass, height_cm):
    """Mosteller body surface area, m^2."""
    return np.sqrt(mass * height_cm / 3600.0)


def reference_individual(sex: str = "male") -> Individual:
    return make_individual(sex=sex, age=30.0, label=f"ICRP 89 reference {sex}")


def make_individual(sex="male", age=30.0, body_mass=None, height=None, label=None, enzyme_scale=None) -> Individual:
    """Scale the ICRP reference to a given body size.

    Lean tissues scale with lean body weight, adipose with fat mass, blood
    flows and blood volume with body surface area. GFR follows a simple
    age decline (about -1 mL/min/year after 40, a common approximation) and
    scales with body surface area (normalized to 1.73 m^2 at 120 mL/min, male).
    """
    ref = _ICRP89[sex]
    m = ref["body_mass"] if body_mass is None else float(body_mass)
    h = ref["height"] if height is None else float(height)
    lbw_ref = _lean_body_weight(sex, ref["body_mass"], ref["height"])
    lbw = _lean_body_weight(sex, m, h)
    fat_ref = ref["body_mass"] - lbw_ref
    fat = max(m - lbw, 1.0)
    bsa_ratio = _bsa(m, h) / _bsa(ref["body_mass"], ref["height"])

    vol = {}
    for t, mass in ref["mass"].items():
        vol[t] = mass * (fat / fat_ref if t == "adipose" else lbw / lbw_ref)
    blood = ref["blood_l"] * bsa_ratio
    modeled = sum(vol.values()) + blood * 1.06
    vol["rest"] = max(m - modeled, 0.02 * m)     # minor organs and connective tissue

    co = ref["cardiac_output_l_min"] * 60.0 * bsa_ratio
    flows = {t: co * pct / 100.0 for t, pct in ref["flow_pct"].items() if t != "liver_arterial"}
    flows["liver"] = co * ref["flow_pct"]["liver_arterial"] / 100.0
    flows["lung"] = co

    gfr_ml_min = (120.0 if sex == "male" else 110.0) * _bsa(m, h) / 1.73
    if age > 40:
        gfr_ml_min -= 1.0 * (age - 40.0)
    return Individual(
        sex=sex, age=float(age), body_mass=m, height=h, volumes=vol, flows=flows, cardiac_output=co,
        hematocrit=ref["hematocrit"], v_arterial=0.25 * blood, v_venous=0.75 * blood,
        gfr=max(gfr_ml_min, 10.0) * 0.06, enzyme_scale=dict(enzyme_scale or {}),
        label=label or f"{sex}, {age:.0f} y, {m:.0f} kg",
    )


def sample_population(n: int, seed: int = 0, sex_mix: float = 0.5, age_range=(20, 50),
                      enzyme_cv: float = 0.4) -> list[Individual]:
    """Virtual population with body-size and enzyme-activity variability.

    Heights and BMIs are drawn from normal distributions typical of adult
    Western populations; enzyme activity multipliers are log-normal with the
    given coefficient of variation (CYP3A4 activity varies roughly 30-50 %
    between people).
    """
    rng = np.random.default_rng(seed)
    out = []
    sigma = np.sqrt(np.log(1 + enzyme_cv**2))
    for i in range(n):
        sex = "male" if rng.random() < sex_mix else "female"
        age = rng.uniform(*age_range)
        h = rng.normal(176 if sex == "male" else 163, 7)
        bmi = float(np.clip(rng.normal(24.5, 3.5), 17, 40))
        m = bmi * (h / 100) ** 2
        scale = {e: float(np.exp(rng.normal(-0.5 * sigma**2, sigma))) for e in ("CYP3A4", "UGT1A4", "CYP1A2", "hepatic")}
        out.append(make_individual(sex, age, m, h, label=f"virtual #{i + 1}", enzyme_scale=scale))
    return out


def with_overrides(ind: Individual, **kw) -> Individual:
    return replace(ind, **kw)
