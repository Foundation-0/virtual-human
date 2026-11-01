"""Virtual Human: a multi-scale simulation stack, built from scratch.

Layers
------
vh.molecular  Atom-level physics engine (JAX). Proteins in implicit water,
              AMBER ff14SB + GB-OBC2. Validated term by term against OpenMM.
vh.body       Whole-body physiology / drug disposition (PBPK) engine.
vh.bridge     Hands molecular-scale quantities (binding, unbound fraction)
              up to the body model (target occupancy, effect).

See README.md for what each layer can and cannot claim.
"""

__version__ = "0.1.0"
