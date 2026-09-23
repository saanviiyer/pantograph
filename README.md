# pantograph

*A pantograph traces a figure exactly and understands nothing about it.*

A science sandbox in which **what an agent found** and **what an agent
understood** are two separate measurements, and the second one is a number
rather than a reading of the agent's notebook.

Built on the science-sandboxes framework (arXiv:2608.30165), which instantiates
agentic scientific discovery as repeated cycles of experiment, feedback and
hypothesis revision, and reports that agents "successfully optimized a
quantitative metric without understanding the rules underlying the system",
with reasoning that deteriorated on rules outside familiar biological priors.
This project takes that finding seriously enough to try to break it, by fixing
the two things that make it hard to check.

| | published sandbox | here |
|---|---|---|
| score axis | best fitness in M rounds | same, plus the budget at which it stops being saturated by chance |
| understanding axis | humans or an AI judge read the agent's lab notebook | the agent submits an **executable predictor**, scored on sequences it never queried, against a **surrogate fitted to its own query history** |
| familiar vs unfamiliar rules | contrasted across rules that also differ in length, arithmetic and locality | contrasted across **matched pairs**: identical program, identical token count, identical bits to determine, differing only in reading frame |

## What has run

`results/`, all three gates, reproducible under any `PYTHONHASHSEED`.

- **Gate 0** — the world grid, the probes and the null arm. **PASS.**
- **Gate 1** — the budget ladder. **The score axis is saturated.** At the
  published budget of 500 rounds, undirected random search reaches the fitness
  ceiling in 15–85% of runs, and a genetic algorithm with no understanding at
  all averages 8.82–8.97 out of 9.
- **Gate 2** — the interpretation ladder for the understanding metric.

## What has not run

**The agent arm has never been executed.** `src/pantograph/session.py` and
`scripts/score_session.py` are wired and tested end to end against a stand-in,
and no language model has been put in the loop. Every number in `FINDINGS.md`
is a property of the apparatus. Nothing here is yet a statement about any agent.

## What this harness cannot see

Written before the results, per `../HARNESS.md` H14.

- **It is entirely dry.** Every world is invented. The H/P lattice is a fixed
  physics shared across worlds so that differences between worlds are
  differences in the hidden rule; it is not a claim about folding.
- **The fitness ceiling is 9 and the chain is 16 residues.** Conclusions about
  the score axis are conclusions about a small landscape.
- **`familiar` is operationalised as reading frame.** The matched pairs vary
  whether the rule respects codon boundaries and nothing else. That is one
  biological prior. A result about it is not a result about biological priors
  in general.
- **The understanding metric measures a submitted predictor**, so it cannot
  distinguish an agent that has a rule and cannot code it from one that has no
  rule. Gate 2's ladder bounds how much that costs: a correct decoder paired
  with a crude stand-in for the physics scores about 0.4, not 1.0.
- **Rule discovery is scored by ranking, not by reading the rule.** An agent
  that recovers a rule equivalent to the true one under relabeling scores the
  same as one that recovers it exactly, which is intended, and an agent that
  recovers a different rule with the same predictions is not distinguished from
  a correct one, which is not.

## Layout

```
src/pantograph/
  hp.py             the shared physics: 2D H/P lattice fitness for all 2**16 chains
  dsl.py            a register machine for hidden rules, so complexity is computed
  worlds.py         the rule grid: two matched pairs plus two off-prior controls
  oracle.py         the sealed world, the query budget, the logged history
  probes.py         held-out probe sets, including the arrangement probe
  understanding.py  the surrogate null arm and the out-of-sample metric
  session.py        the agent's side of the boundary; notebook and submissions
  baselines.py      random search, (1+1)-EA, GA — the score-axis nulls
scripts/
  gate0_grid.py     apparatus gates
  gate1_budget.py   does the score axis discriminate?
  gate2_ladder.py   what does a given rho mean?
  score_session.py  score one agent run on both axes
```

## Run

```bash
python3 -m pytest tests -q && python3 scripts/gate0_grid.py && python3 scripts/gate1_budget.py && python3 scripts/gate2_ladder.py
```
