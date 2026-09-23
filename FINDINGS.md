# Findings

Results as they land, each against the gate registered in [`PREREG.md`](PREREG.md).
Gates 0–2 concern the apparatus and were run before the pre-registration was
written; that is stated at the top of `PREREG.md` and again here. **No agent has
been run.**

---

## Gate 0: the world grid, the probes and the null arm

`scripts/gate0_grid.py`, `results/gate0_grid.json`. 2026-09-02. **PASS.**

The physics layer reproduces the published 2D H/P results independently: the
ceiling over all 65,536 chains is 9, reached by 143 of them, and the standard
16-mer benchmark `HPHPPHHPHPPHPHHP` scores 6.

| world | fam | tokens | bits | GA median | GA at ceiling | null rho iid | null rho perm | counts rho perm | headroom |
|---|---|---|---|---|---|---|---|---|---|
| `wobble` | yes | 8 | 4.00 | 9.0 | 0.85 | 0.797 | 0.242 | 0.000 | 0.76 |
| `offset_base` | no | 8 | 4.00 | 9.0 | 0.90 | 0.785 | 0.211 | 0.000 | 0.79 |
| `codon_table` | yes | 78 | 64.00 | 9.0 | 0.95 | 0.275 | 0.032 | 0.000 | 0.97 |
| `offset_frame` | no | 78 | 64.00 | 9.0 | 0.90 | 0.301 | 0.085 | 0.000 | 0.92 |
| `xor_pair` | no | 13 | 4.00 | 9.0 | 1.00 | 0.339 | 0.038 | 0.000 | 0.96 |
| `index_gated` | no | 26 | 4.75 | 9.0 | 0.60 | 0.135 | 0.040 | 0.000 | 0.94 |

Both pairs are matched precisely on tokens and bits, and within 1.0 on GA median.
Every world's fitness ceiling is certified at 9 by exhaustive reachability, not
by sampling.

The `counts rho perm` column is the gate that licenses the arrangement claim: a
predictor built on in-frame composition features is pinned to a single constant
across the whole probe, so it cannot rank it at all. This is exact, and the test
suite asserts the features are literally equal rather than merely uninformative.

### Two gates failed on the way here, and both changed the design

**G0.6 rejected the first unfamiliar high-complexity world.** `scrambled_table`
read the residue table at `(a·codon + b·position) mod 64`, and carried 75 bits
against `codon_table`'s 64, it was harder as well as stranger, which is the
confound the project exists to remove. It was replaced by `offset_frame`, the
same 64-entry table read at offsets 0, 2 and 4, which is the same program with
different read positions: 78 tokens against 78, 64 bits against 64.

**G0.4 rejected the first null arm.** The surrogate was built on an in-frame
basis (position-wise nucleotides and codons) and scored 0.447 on
`codon_table`'s arrangement probe against 0.094 on `offset_frame`'s. The null
was carrying the reading-frame prior that the pair contrast is supposed to
measure, so a margin against it would have reported the feature basis's prior
as the agent's. The basis was rebuilt to contain every k-tuple at every start
and every dilation. Measured asymmetry within a pair fell to at most 0.053
(G0.7), and the pair statistic is a difference in differences on top of that.

Neither is a result. Both are the gate doing its job, and the second one would
have produced a clean, well-supported, entirely artefactual confirmation of the
published claim.

---

## Gate 1: the score axis is saturated at the published budget

`scripts/gate1_budget.py`, `results/gate1_budget.json`. 2026-09-02. 40 seeds.

Probability that **undirected random search** reaches the ceiling of 9:

| world | 25 | 50 | 100 | 250 | 500 |
|---|---|---|---|---|---|
| `wobble` | 0.00 | 0.07 | 0.20 | 0.38 | 0.68 |
| `offset_base` | 0.03 | 0.10 | 0.25 | 0.33 | 0.68 |
| `codon_table` | 0.15 | 0.15 | 0.15 | 0.38 | 0.62 |
| `offset_frame` | 0.07 | 0.10 | 0.28 | 0.45 | 0.78 |
| `xor_pair` | 0.07 | 0.10 | 0.15 | 0.47 | 0.85 |
| `index_gated` | 0.00 | 0.00 | 0.03 | 0.07 | 0.15 |

Mean best fitness at budget 500, out of a ceiling of 9:

| strategy | range across worlds |
|---|---|
| random | 8.10 – 8.85 |
| (1+1)-EA | 8.82 – 9.00 |
| genetic | 8.82 – 8.97 |

**At 500 rounds there is less than one point of range on a nine-point scale
between a searcher with no understanding and the ceiling.** A protein-fitness
sandbox scored this way cannot distinguish an agent that inferred the code from
an agent that sampled at random, because both finish at 8.8.

The axis separates at 50–100 rounds: random search averages 7.38–8.10 at 50
against a GA's 7.62–8.35, and the ceiling is reached in 0–15% of random runs.
Every agent session in this project is therefore scored at budget 50 as well as
at 500, and no claim about optimisation quality is made from the 500 column
alone.

This is `HARNESS` H6 collecting again, and it is the same shape as `purchase`: measured against a competent dumb search at matched
budget, the uplift is not there.

---

## Gate 2: what a number on the arrangement probe means

`scripts/gate2_ladder.py`, `results/gate2_ladder.json`. 2026-09-02. Spearman rho,
median over 5 probe seeds, 400 sequences each.

| world | null | wrong decoder | decoder, naive physics | decoder, exact physics |
|---|---|---|---|---|
| `wobble` | 0.166 | −0.044 | 0.393 | 1.000 |
| `offset_base` | 0.151 | −0.052 | 0.469 | 1.000 |
| `codon_table` | 0.071 | 0.015 | 0.467 | 1.000 |
| `offset_frame` | 0.020 | 0.003 | 0.402 | 1.000 |
| `xor_pair` | −0.002 | −0.032 | 0.468 | 1.000 |
| `index_gated` | 0.055 | −0.027 | 0.408 | 1.000 |

Three things this fixes.

**The metric resolves partial understanding.** The rung between the null and the
ceiling (having the rule but only a crude additive model of the folding) sits
at 0.39–0.47 in every world. An agent's number can be placed against that,
rather than against zero.

**Having the physics without the rule is worth nothing.** The `wrong decoder`
column applies the exact fitness function to residues decoded by the wrong
world's rule and returns 0.00 ± 0.05 everywhere. The probe is not rewarding
general competence about chains; it is rewarding the specific rule.

**The ladder is flat across the grid.** The rungs sit within about 0.08 of each
other across all six worlds, so a margin in `wobble` and a margin in
`offset_frame` are on the same scale. Without this, the pair contrast would not
be a contrast.

---

## Arm B: the agent arm

**Unrun.** The session interface, the framing axis, the checkpoint submission
protocol and the scorer are written and tested end to end against a stand-in
that hill-climbs and then submits the true `wobble` decoder with a naive energy
model. That stand-in scores 0.527 on the arrangement probe against a null of
−0.030, which is the expected place on the Gate 2 ladder and is the only
evidence so far that the scoring path works.

Nothing in this file is a statement about any language model.
