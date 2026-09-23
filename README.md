# pantograph

A pantograph copies a figure line for line and knows nothing about it.

This project is a science sandbox. It measures what an agent found and what an agent understood as two separate quantities. The second quantity is a number, so no one has to read the agent's notebook to get it.

It builds on the science-sandboxes work (arXiv:2608.30165). That work runs agentic scientific discovery as repeated cycles of experiment, feedback and hypothesis revision. It reports that agents "successfully optimized a quantitative metric without understanding the rules underlying the system", and that their reasoning got worse on rules outside familiar biological priors. This project tries to break that finding. It fixes the two things that make the finding hard to check.

| | published sandbox | here |
|---|---|---|
| score axis | best fitness in M rounds | the same, plus the budget at which chance no longer saturates it |
| understanding axis | humans or an AI judge read the agent's lab notebook | the agent submits an executable predictor. It is scored on sequences it never queried, against a surrogate fitted to its own query history |
| familiar vs unfamiliar rules | compared across rules that also differ in length, arithmetic and locality | compared across matched pairs with the same program, token count and bits to determine. The pairs differ only in reading frame |

## Results

All three apparatus gates have run. The outputs are in `results/` and reproduce under any `PYTHONHASHSEED`. Details are in [`FINDINGS.md`](FINDINGS.md).

Gate 0 checks the world grid, the probes and the null arm. It passes.

Gate 1 is the budget ladder. It shows the score axis is saturated. At the published budget of 500 rounds, undirected random search reaches the fitness ceiling in 15 to 85% of runs. A genetic algorithm with no understanding averages 8.82 to 8.97 out of 9.

Gate 2 calibrates the understanding metric (Spearman rho on held-out probe sequences). A correct decoder with exact physics scores 1.000 in every world. A correct decoder with a crude additive model of folding scores 0.39 to 0.47. Exact physics applied through the wrong world's decoder scores 0.00 ± 0.05. So the metric rewards the specific rule and resolves partial understanding.

## What has not run

The agent arm has never run. `src/pantograph/session.py` and `scripts/score_session.py` are wired and tested end to end against a stand-in. No language model has been in the loop. Every number in `FINDINGS.md` is a property of the apparatus. Nothing here is a statement about any agent yet.

## Limits

These were written before the results, per H14 of the shared research checklist.

- It is fully dry. Every world is invented. The H/P lattice is a fixed physics shared across worlds, so differences between worlds come from the hidden rule. It makes no claim about folding.
- The fitness ceiling is 9 and the chain is 16 residues. Conclusions about the score axis apply to a small fitness landscape.
- "Familiar" is defined as reading frame. The matched pairs vary only in whether the rule respects codon boundaries. That is one biological prior. A result about it does not cover biological priors in general.
- The understanding metric scores a submitted predictor. It cannot tell apart an agent that has a rule but cannot code it from an agent with no rule. Gate 2 bounds the cost. A correct decoder paired with a crude stand-in for the physics scores about 0.4, where a perfect one scores 1.0.
- Rule discovery is scored by ranking, without reading the rule. An agent that recovers a rule equal to the true one under relabeling scores the same as one that recovers it literally. That is intended. An agent that recovers a different rule with the same predictions also scores the same as a correct one. That is a known gap.

## Install and run

```bash
pip install -e ".[dev]"
python3 -m pytest tests -q
python3 scripts/gate0_grid.py
python3 scripts/gate1_budget.py
python3 scripts/gate2_ladder.py
```

The first run computes the fitness table for all 2^16 chains and caches it in `cache/hp16.npz`. The cache is not in the repo.

## Layout

```
PROPOSAL.md         the claim and the design
PREREG.md           gates and bands, fixed before the runs
FINDINGS.md         gate results
src/pantograph/
  hp.py             the shared physics: 2D H/P lattice fitness for all 2**16 chains
  dsl.py            a register machine for hidden rules, so complexity is computed
  worlds.py         the rule grid: two matched pairs plus two off-prior controls
  oracle.py         the sealed world, the query budget, the logged history
  probes.py         held-out probe sets, including the arrangement probe
  understanding.py  the surrogate null arm and the out-of-sample metric
  session.py        the agent's side of the boundary, with notebook and submissions
  baselines.py      random search, (1+1)-EA and a GA as score-axis nulls
scripts/
  gate0_grid.py     apparatus gates
  gate1_budget.py   does the score axis discriminate?
  gate2_ladder.py   what does a given rho mean?
  score_session.py  score one agent run on both axes
results/            gate outputs (JSON)
```
