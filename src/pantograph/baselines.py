"""Score-axis baselines. HARNESS H6: the trivial baseline goes in the same
table as the method, and it wins surprisingly often.

`purchase` measured uplift against directed evolution and found it null. The
same discipline applies here: an agent's best-found fitness means nothing until
a hill-climber with no understanding whatsoever has been run at the same budget
on the same world.
"""

from __future__ import annotations

import numpy as np

from .oracle import Oracle


def random_search(oc: Oracle, rng: np.random.Generator, batch: int = 50) -> int:
    while oc.used < oc.budget:
        n = min(batch, oc.budget - oc.used)
        oc.query(rng.integers(0, 4, size=(n, oc.world.n_nt), dtype=np.uint8))
    return oc.best


def hill_climb(oc: Oracle, rng: np.random.Generator, rate: float | None = None) -> int:
    """(1+1)-EA. Mutate each nucleotide with probability 1/n, accept ties."""
    n_nt = oc.world.n_nt
    rate = 1.0 / n_nt if rate is None else rate
    cur = rng.integers(0, 4, size=(1, n_nt), dtype=np.uint8)
    cur_y = int(oc.query(cur)[0])
    while oc.used < oc.budget:
        cand = cur.copy()
        m = rng.random(n_nt) < rate
        if not m.any():
            m[rng.integers(0, n_nt)] = True
        cand[0, m] = rng.integers(0, 4, size=int(m.sum()), dtype=np.uint8)
        y = int(oc.query(cand)[0])
        if y >= cur_y:
            cur, cur_y = cand, y
    return oc.best


def genetic(oc: Oracle, rng: np.random.Generator, pop: int = 20, elite: int = 5) -> int:
    """A small GA with uniform crossover, at the same budget."""
    n_nt = oc.world.n_nt
    P = rng.integers(0, 4, size=(pop, n_nt), dtype=np.uint8)
    Y = oc.query(P)
    while oc.used < oc.budget:
        order = np.argsort(-Y)
        P, Y = P[order], Y[order]
        n = min(pop, oc.budget - oc.used)
        if n <= 0:
            break
        pa = P[rng.integers(0, elite, size=n)]
        pb = P[rng.integers(0, elite, size=n)]
        mask = rng.random((n, n_nt)) < 0.5
        kids = np.where(mask, pa, pb).astype(np.uint8)
        mut = rng.random((n, n_nt)) < 1.0 / n_nt
        kids[mut] = rng.integers(0, 4, size=int(mut.sum()), dtype=np.uint8)
        ky = oc.query(kids)
        P = np.concatenate([P[:elite], kids])
        Y = np.concatenate([Y[:elite], ky])
    return oc.best


STRATEGIES = {"random": random_search, "hill_climb": hill_climb, "genetic": genetic}
