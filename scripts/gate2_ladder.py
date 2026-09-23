"""Gate 2 — what does a given rho on the arrangement probe actually mean?

A number is uninterpretable without the rungs around it. `crosstalk` has a proxy
ladder and `carryover` a detector ladder for the same reason. Here the rungs are
predictors built from known amounts of the world:

  null            a ridge fitted to 500 random queries; no rule, no physics
  wrong_decoder   the exact physics, applied to a decoder from the wrong world
  decoder_naive   the true decoder, with a crude additive stand-in for folding
  decoder_exact   the true decoder and the true physics

An agent's submitted predictor is read against this ladder, not against zero.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pantograph import baselines, hp, oracle, probes, worlds  # noqa: E402
from pantograph import understanding as U  # noqa: E402

PROBE_N = 400
SEEDS = 5
OUT = Path(__file__).resolve().parents[1] / "results"
PAIRS = [(i, j) for i in range(hp.N_RES) for j in range(i + 2, hp.N_RES)]


def naive_energy(res: np.ndarray) -> np.ndarray:
    """Every H-H pair counts, ignoring whether any fold realises them all. The
    rung for an agent that has the code but not the chain."""
    h = res.astype(np.float64)
    return sum(h[:, i] * h[:, j] for i, j in PAIRS)


def main() -> int:
    tab = hp.table()
    rows = []
    for name in worlds.GRID:
        w = worlds.build(name, seed=0)
        wrong = worlds.build("index_gated" if name != "index_gated" else "xor_pair", seed=1)
        acc: dict[str, list[float]] = {k: [] for k in
                                       ("null", "wrong_decoder", "decoder_naive", "decoder_exact")}
        for s in range(SEEDS):
            rng = np.random.default_rng(oracle.seed_of("ladder", name, s))
            px = probes.perm(PROBE_N, w.n_nt, w.k, rng, world=w)
            ref = oracle.Oracle(w, budget=1)
            py = ref.truth(px)

            oc = oracle.Oracle(w, budget=500)
            baselines.random_search(oc, np.random.default_rng(oracle.seed_of("ladderq", name, s)))
            hx, hy = oc.history
            acc["null"].append(U.rho(U.Ridge(w.k, basis="stride").fit(hx, hy)(px), py))

            codes = (wrong.residues(px).astype(np.uint32)
                     << np.arange(hp.N_RES, dtype=np.uint32)).sum(axis=1)
            acc["wrong_decoder"].append(U.rho(tab[codes], py))
            acc["decoder_naive"].append(U.rho(naive_energy(w.residues(px)), py))
            acc["decoder_exact"].append(U.rho(ref.truth(px), py))

        row = {"world": name, "familiar": w.familiar}
        row.update({k: round(float(np.median(v)), 3) for k, v in acc.items()})
        rows.append(row)

    OUT.mkdir(exist_ok=True)
    (OUT / "gate2_ladder.json").write_text(json.dumps({"probe_n": PROBE_N, "seeds": SEEDS,
                                                       "rows": rows}, indent=2))
    hdr = (f"{'world':14s} {'fam':>3s} | {'null':>6s} {'wrong_dec':>10s} "
           f"{'dec_naive':>10s} {'dec_exact':>10s}")
    print("Spearman rho on the arrangement probe\n")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['world']:14s} {str(r['familiar'])[0]:>3s} | {r['null']:6.3f} "
              f"{r['wrong_decoder']:10.3f} {r['decoder_naive']:10.3f} {r['decoder_exact']:10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
