# Pantograph — proposal

Drafted 2026-09-02.

> A pantograph is a linkage that traces a figure exactly. It has no
> representation of the figure. Nothing about its output distinguishes it from
> a draughtsman.

## Thesis

Science sandboxes (arXiv:2608.30165) do the right thing: they put an agent in a
loop with a sealed world and ask whether it can learn the rules rather than
merely hit the metric. Their headline is that agents often cannot — that they
optimise a score without understanding the system, and that their reasoning
degrades on rules outside familiar biological priors.

Both halves of that headline rest on measurements that this project argues are
not yet load-bearing.

**The understanding half is adjudicated by reading the notebook.** The paper is
explicit that "the central object of evaluation is the agent's reasoning, not
simply its final score", and equally explicit that there is no automated
quantitative metric for it: humans, or AI judges, inspect the free-form record
of hypotheses. Under `../HARNESS.md` that is a single-channel adjudicator (H10)
with no null arm (P6) and no trivial baseline (H6). It cannot separate an agent
that states a rule it never used from one that used a rule it never stated, and
it has no denominator: there is no reading of a notebook that corresponds to
*the amount of structure the agent's own queries already contained*.

**The prior half is confounded with everything else about the rules.** The
published dry oracles contrast `GC balance` and `alternating purine` against
`Collatz`, `Fibonacci positions` and a hidden English cipher. The familiar rules
are also the short ones, the compositional ones, and the ones whose landscapes
hill-climb. H8 says a quantity the verdict could be a function of, other than
the intended one, has to be named and held fixed. Description length and
searchability are those quantities, and they are not held fixed.

The claim of this project is that when both are fixed, the interesting result
survives in a form that can be defended, and that a third result — one the
published design cannot see — appears immediately.

## The third result, which is already in hand

Before any agent is asked to compete on a score, the score has to discriminate.
The published protein-fitness sandbox gives 500 rounds on a 16-residue H/P
chain. Of the 65,536 H/P chains, 143 sit at the ceiling of 9 contacts.
Undirected random sampling therefore reaches the ceiling with probability
around 0.67 in 500 draws, and Gate 1 measures 0.15–0.85 across the grid, with a
genetic algorithm at 8.82–8.97 out of 9.

**The quantitative axis of that sandbox has under one point of dynamic range
between an agent and a coin.** Every conclusion about whether agents "optimise
well" on it is a conclusion about a saturated metric. This is `purchase`'s
finding in a different substrate — measured uplift is null against a competent
dumb search — and it is `HARNESS` H6 collecting again.

That is not an argument against the framework. It is an argument for reporting
the trivial baseline in the same table, and for choosing the budget at which
the axis still separates: Gate 1 puts that at 50–100 rounds, not 500.

## What is built

A sandbox of the same shape as CodonBox — a hidden encoding from nucleotides to
a two-letter residue alphabet, a fixed shared physics on the decoded chain, a
query budget, a notebook — with three changes.

### 1. Understanding is a prediction, not a reading

At fixed checkpoints the agent submits `predict(seqs)`, executable code. It is
scored out of sample on probe sets it never queried, against a ridge surrogate
fitted to the agent's *own query history*. The surrogate is the null arm: it is
everything the agent's data already contains, with no rule in it anywhere.
Understanding is the margin over it.

The notebook is still collected. The disagreement between what the notebook
claims and what the predictor achieves is then itself a measurable quantity,
and it is the direct test of the published adjudicator.

### 2. The arrangement probe

A ridge on composition can rank random sequences well without any rule, so an
in-distribution probe cannot separate understanding from fitting. The primary
probe is instead a set of **permutations of one fixed multiset of codons**,
filtered to hold the residue composition fixed as well. Every count feature is
constant across it by construction — Gate 0 asserts this exactly, not
approximately — so nothing is left to predict from except the arrangement of
residues along the chain, which is what the fold depends on.

Gate 2 calibrates what a score on it means:

| rung | what it has | rho |
|---|---|---|
| null | a surrogate fitted to 500 of its own queries | 0.00 – 0.17 |
| wrong decoder | the exact physics, the wrong rule | ≈ 0 |
| decoder, naive physics | the true rule, a crude additive stand-in for folding | 0.39 – 0.47 |
| decoder, exact physics | both | 1.00 |

The ladder is flat across all six worlds, so the metric is comparable across
the grid rather than only within a world.

### 3. Matched pairs, varying only the reading frame

| | familiar | unfamiliar | tokens | bits |
|---|---|---|---|---|
| low complexity | `wobble` — a table over base 0 of codon *i* | `offset_base` — a table over base 0 of codon *i+1* | 8 = 8 | 4 = 4 |
| high complexity | `codon_table` — a table over the three bases of codon *i* | `offset_frame` — a table over bases at offsets 0, 2, 4 | 78 = 78 | 64 = 64 |

Within a pair the two worlds are the same program with the read offsets
changed. Identical description length, identical number of bits to determine,
identical fitness ceiling, and — Gate 0 — indistinguishable to a hill-climber.
The only difference is whether the rule respects the reading frame.

Two further worlds, `xor_pair` and `index_gated`, are unfamiliar in the way the
published grid is unfamiliar (a bitwise operation, a modular one) and are kept
so the frame contrast can be checked against a second axis.

It is not an accident that the matched contrast is a frameshift. In `crosstalk`
the frameshift control was the only one that turned out to be decisive about
codon-level leakage; here it is the only operation that can make a rule
off-prior without making it longer.

The framing axis from the published work is retained: every world is presented
either in biological vocabulary (ACGT, codon, residue, folding) or in a
symbolic one (0123, token, element, score), with the oracle identical. Framing
and rule are then crossed, and whether the prior is invoked by vocabulary or
carried by the rule becomes separable.

## What would make this a finding, and what would kill it

The design's own null result is the interesting one. If, once complexity and
searchability are matched, an agent's understanding on `wobble` and
`offset_base` is the same, then the published off-prior collapse is a statement
about rule complexity and not about biological priors, and the paper's most
quotable sentence needs an amendment. If instead the gap survives the matching,
the claim is stronger than the paper could show, because the pair rules out the
alternative explanations by construction.

Either outcome is reportable. The one that would void the project is the
understanding metric failing to resolve anything — every agent at the null on
every world — which Gate 2's ladder makes checkable in advance and which it
currently passes.

## Relation to the rest of the tree

- `carryover` — the same distinction in a different substrate: executing a
  protocol successfully is a property of the instruction stream, not of the
  experiment. Optimising a fitness score is a property of the search, not of
  the science.
- `palimpsest` — readout versus representation. There the question is whether
  unlearning removes knowledge or only the answer; here whether a notebook
  reports a rule or only a score.
- `purchase` — uplift measured against directed evolution was null. Gate 1 is
  the same measurement made on the sandbox's own axis.
- `crosstalk` — the frameshift control, and H7: the split is the experiment.
  Here the probe is the experiment.
