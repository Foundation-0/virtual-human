"""Physical constants in the engine's unit system.

Units throughout the molecular engine follow the common MD convention:
length nm, time ps, mass amu (g/mol), energy kJ/mol, charge e, angle rad.
"""

# Boltzmann constant x Avogadro (molar gas constant), kJ/(mol K). CODATA 2018.
KB = 0.00831446261815324

# Coulomb constant 1/(4 pi eps0) in kJ mol^-1 nm e^-2. CODATA 2018 value
# (identical to the constant OpenMM 8 uses, measured empirically).
ONE_4PI_EPS0 = 138.93545764438198

# GB-OBC constants (Onufriev, Bashford & Case 2004, "OBC2" set).
OBC_DIELECTRIC_OFFSET = 0.009  # nm
OBC2_ALPHA, OBC2_BETA, OBC2_GAMMA = 1.0, 0.8, 4.85
ACE_PROBE_RADIUS = 0.14  # nm, water probe for the nonpolar (ACE) term
