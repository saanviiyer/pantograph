"""Gate 1 — does the score axis have any headroom at all?

The published protein-fitness sandbox gives the agent 500 rounds on a 16-residue
H/P chain. Before any agent is asked to compete on that axis, the axis has to be
shown to discriminate: a searcher with no understanding whatsoever must not
already be at the ceiling. HARNESS H6.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pantograph import baselines, oracle, worlds  # noqa: E402

BUDGETS = (25, 50, 100, 250, 500)
SEEDS = 40
OUT = Path(__file__).resolve().parents[1] / "results"


def main() -> int:
    rows = []
    for name in worlds.GRID:
        w = worlds.build(name, seed=0)
        for b in BUDGETS:
            for strat, fn in baselines.STRATEGIES.items():
                best = []
                for s in range(SEEDS):
                    oc = oracle.Oracle(w, budget=b)
                    fn(oc, np.random.default_rng(oracle.seed_of(name, strat, b, s)))
                    best.append(oc.best)
                best = np.asarray(best)
                rows.append({"world": name, "budget": b, "strategy": strat,
                             "median": float(np.median(best)),
                             "mean": round(float(best.mean()), 2),
                             "p_ceiling": round(float((best >= 9).mean()), 3)})
    OUT.mkdir(exist_ok=True)
    (OUT / "gate1_budget.json").write_text(json.dumps({"seeds": SEEDS, "rows": rows}, indent=2))

    print("P(random search reaches the ceiling of 9), by budget\n")
    hdr = f"{'world':14s}" + "".join(f"{b:>8d}" for b in BUDGETS)
    print(hdr)
    print("-" * len(hdr))
    for name in worlds.GRID:
        line = f"{name:14s}"
        for b in BUDGETS:
            r = next(x for x in rows if x["world"] == name and x["budget"] == b
                     and x["strategy"] == "random")
            line += f"{r['p_ceiling']:8.2f}"
        print(line)

    print("\nmean best fitness, budget 500, by strategy\n")
    hdr2 = f"{'world':14s}" + "".join(f"{s:>12s}" for s in baselines.STRATEGIES)
    print(hdr2)
    print("-" * len(hdr2))
    for name in worlds.GRID:
        line = f"{name:14s}"
        for strat in baselines.STRATEGIES:
            r = next(x for x in rows if x["world"] == name and x["budget"] == 500
                     and x["strategy"] == strat)
            line += f"{r['mean']:12.2f}"
        print(line)

    sat = [r for r in rows if r["strategy"] == "random" and r["budget"] == 500]
    worst = min(r["p_ceiling"] for r in sat)
    print(f"\nAt the published budget of 500, undirected random search reaches the "
          f"ceiling in at least {worst:.0%} of runs in every world.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
