# Why not simulate every atom of a body?

Order-of-magnitude estimate for classical all-atom molecular dynamics (MD), with deliberately
generous assumptions.

| Quantity | Value | Note |
|---|---|---|
| Atoms in a 70 kg human | ~7 × 10²⁷ | mostly H and O in water |
| Time step | ~2 fs | set by the fastest bond vibrations (X–H) |
| Steps per simulated second | 5 × 10¹⁴ | 1 s / 2 fs |
| Work per atom per step | ≥ 10³ FLOP | real codes with cutoffs and Ewald sums use more |
| **Work per simulated second of one body** | **≳ 3.5 × 10⁴⁵ FLOP** | |
| Fastest supercomputers (2025–26) | ~2 × 10¹⁸ FLOP/s | exascale class |
| **Wall time for 1 s of body** | **~2 × 10²⁷ s ≈ 6 × 10¹⁹ years** | ~4 billion × the age of the universe |
| Real time would need | ~10²⁷ × more compute than the fastest machine | |
| One human cell (~10¹⁴ atoms), 1 s | ~5 × 10³¹ FLOP → ~800,000 years | same machine |

And classical MD can't break or form chemical bonds, so drug metabolism, which is
chemistry, would need quantum methods that cost far more per atom.

## What this means for the design

Molecular accuracy is applied where molecules decide the outcome, then
coarse-grained upward:

1. **Atoms** (`vh.molecular`): protein structure and motion, and drug–target
   interactions. Systems of 10²–10⁶ atoms for nanoseconds to microseconds.
2. **Bridge** (`vh.bridge`): molecular outputs (binding constants, unbound
   fractions, and in future rates computed by free-energy simulations) become
   parameters of the body model.
3. **Body** (`vh.body`): organs, blood flow, distribution and elimination over
   hours to days. Exact and faster than real time.

This is how drug development already uses simulation (physiologically based
pharmacokinetic models appear in regulatory submissions). The difference here is
one stack, built from scratch, where the molecular layer is first-class and
differentiable.
