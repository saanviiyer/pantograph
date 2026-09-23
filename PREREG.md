# Pre-registration

## Provenance, stated first, because half of this file is not blind

Gates 0, 1 and 2 were **run before this file was written**. They are apparatus
checks, not results about any agent, but they are not blind and they are not
described as such anywhere in the project. What they established, and what a
reader should therefore discount:

- the score axis is saturated at 500 rounds and separates at 50–100 (Gate 1);
- the arrangement probe pins a composition-only predictor to a constant exactly
  and leaves 0.76–0.99 of headroom over the position-aware surrogate (Gate 0);
- the interpretation ladder is flat across the grid (Gate 2);
- the matched pairs are matched on tokens, bits and searchability (Gate 0);
- the surrogate's own frame sensitivity across a pair is at most 0.053, so the
  null arm does not carry the prior under test (Gate 0, G0.7).

**The agent arm has never been executed.** No language model has been placed in
the loop, no session has been run, and no submitted predictor has been scored.
Everything below about agent behaviour is blind in the ordinary sense.

The honest description is *apparatus open, agent arm pre-registered*, and it is
written that way in any paper this becomes.

---

## Hypotheses

**H1 (the score axis is not the science axis).** At budget 500, the best fitness
found by a frontier agent does not exceed the genetic-algorithm baseline by more
than 0.3 points in any world, on medians over 10 sessions. The GA has no
representation of the rule whatsoever. *Predicted to hold.* If it holds, no
sentence in this project may describe an agent's fitness score as evidence of
understanding, and the score axis is reported at budget 50 thereafter.

**H2 (the notebook over-credits).** For at least 25% of sessions, an LLM judge
reading only the notebook — the published adjudication procedure, reproduced —
rates the agent as having discovered the rule, while the agent's own submitted
predictor scores below the `decoder_naive` rung of 0.39 on the arrangement
probe. The direction is registered: over-crediting, not under-crediting.

**H3 (understanding is not bought by querying).** Median agent margin over the
surrogate null on the arrangement probe is above 0.15 in `wobble` and
`codon_table`, and is not distinguishable from 0 in `index_gated`. This is the
published result, restated as a number.

**H4 (the confound test — the load-bearing prediction).** With description
length, bits and searchability held fixed, the familiar member of a matched
pair is *not* materially easier:

> agent margin on `wobble` − margin on `offset_base` ≤ 0.10, and
> agent margin on `codon_table` − margin on `offset_frame` ≤ 0.10,

with bootstrap intervals over sessions covering 0. **This is predicted to
hold**, against the published claim. If it holds, the reported deterioration on
off-prior systems is attributable to rule complexity rather than to
familiarity, and the finding of this project is a negative one about the prior
explanation. If it fails — if the frame contrast survives matching — the
published claim is stronger than its own design could show, and that is the
finding instead.

**H5 (framing is weaker than rule).** The `bio` minus `symbolic` difference in
margin, pooled over worlds, is smaller in absolute value than the largest
between-world difference. Prior invocation by vocabulary buys less than the
rule's own structure.

**H6 (budget is spent on search, not on tests).** Median fraction of an agent's
queries that lie within Hamming distance 3 of a previously queried sequence
exceeds 0.5 — the brute-force signature the published work describes
qualitatively, made countable.

---

## Outcome measures, fixed now

**Primary.** Margin on the arrangement probe:
`margin = rho(agent predictor) − rho(surrogate fitted to that agent's own queries)`,
Spearman, on 400 held-out permutation sequences, at the final checkpoint.
Bootstrap percentile intervals over **sessions**, not over probe sequences.

**Primary for H4.** The difference of margins within a matched pair, which is a
difference in differences and therefore already nets out the surrogate's own
sensitivity to the reading frame.

**Secondary.** Margin on the `iid` and `ood` probes; top-10 precision on each
probe (H12: the predictor is consumed as a ranking of what to build next, so
the tail is reported beside the bulk statistic); best fitness at budgets 50 and
500; the judge-versus-margin disagreement rate; the near-duplicate query
fraction.

**Never reported as a finding.** Absolute rho without the Gate 2 ladder beside
it. A margin is meaningful only against the rungs `null`, `wrong_decoder`,
`decoder_naive`, `decoder_exact`; a bare correlation is not.

---

## Bands

| quantity | band | reading |
|---|---|---|
| margin, arrangement probe | ≤ 0.05 | the notebook is a readout; the queries contained everything |
| | 0.05 – 0.39 | partial rule, below a correct decoder with naive physics |
| | 0.39 – 0.80 | the rule is substantially recovered |
| | > 0.80 | the rule and the physics are both recovered |
| within-pair margin difference | ≤ 0.10 | familiarity does not survive complexity matching (H4 holds) |
| | > 0.20 with CI excluding 0 | the frame prior is real at matched complexity |
| judge/margin disagreement | ≥ 0.25 | the published adjudicator over-credits (H2 holds) |
| agent minus GA best fitness | ≤ 0.3 | the score axis carries no agent-specific signal |

## Stopping rules

- A submission that fails to execute scores 0 and is reported as an execution
  failure with its own denominator (H11), never dropped.
- A session where the agent spends fewer than 80% of its budget is reported
  separately; it is not a run of the protocol.
- If Gate 0 fails on a rebuilt world grid, no agent arm is run against that
  grid. A FAIL is a stop, not a caveat.

## Named nuisances

Per H8, quantities the verdict could be a function of other than the intended
one, and what is done about each.

| nuisance | held how |
|---|---|
| rule description length | equal token counts within a pair, by construction, asserted in tests |
| bits to determine the rule | equal within a pair, asserted in tests |
| searchability | Gate 0 requires GA medians within 1.0 within a pair |
| fitness ceiling | equal at 9 in every world, certified by exhaustive reachability |
| the null arm's own frame prior | the surrogate basis contains every dilation, and the pair statistic is a difference in differences; measured asymmetry ≤ 0.053 |
| coding ability confounded with understanding | bounded by the Gate 2 `decoder_naive` rung; reported, not removed |
| probe composition | the arrangement probe pins in-frame composition features to a constant exactly |
| interpreter seeding | every seed via SHA-256; all three gates byte-identical under two `PYTHONHASHSEED` values |
