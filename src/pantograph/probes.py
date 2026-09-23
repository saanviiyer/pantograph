"""Held-out probe sets, fixed before any agent runs.

Understanding is scored as prediction on sequences the agent never queried.
Two probes, because the contrast between them is the whole discriminator:

  `iid`  uniform sequences. A surrogate fitted to the agent's own queries can
         do well here without any rule knowledge, so a high score means little.
  `perm` composition-matched permutations of one fixed codon multiset. Every
         count feature is constant across the set by construction, so the only
         thing left to predict from is the *arrangement* of residues along the
         chain, which is what the fold depends on. A rule plus the shared
         physics gets this exactly; an additive surrogate cannot.
  `ood`  low-complexity sequences — restricted sub-alphabets, homopolymer runs,
         short periods. A distribution shift, kept as a secondary probe.

HARNESS H9: a control that cannot fail is decoration. This one can fail — if
the OOD probe turns out to be predictable from composition alone, the gate in
`GATES.md` says so and the probe is rebuilt.
"""

from __future__ import annotations

import numpy as np


def iid(n: int, n_nt: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 4, size=(n, n_nt), dtype=np.uint8)


def ood(n: int, n_nt: int, rng: np.random.Generator) -> np.ndarray:
    """Three constructions in equal thirds: restricted alphabet, homopolymer
    blocks, and short-period repeats."""
    out = np.zeros((n, n_nt), dtype=np.uint8)
    for i in range(n):
        mode = i % 3
        if mode == 0:
            sub = rng.choice(4, size=rng.integers(2, 4), replace=False)
            out[i] = rng.choice(sub, size=n_nt)
        elif mode == 1:
            pos, cur = 0, 0
            while pos < n_nt:
                run = int(rng.integers(3, 9))
                out[i, pos:pos + run] = rng.integers(0, 4)
                pos += run
                cur += 1
        else:
            period = int(rng.integers(2, 7))
            motif = rng.integers(0, 4, size=period)
            out[i] = np.tile(motif, n_nt // period + 1)[:n_nt]
    return out


def perm(n: int, n_nt: int, k: int, rng: np.random.Generator,
         base: np.ndarray | None = None, world=None) -> np.ndarray:
    """Permutations of a single fixed multiset of codons.

    Codon counts, nucleotide counts and every k-mer count are identical across
    the whole set. A predictor that reads composition is pinned to one value
    here and scores rho = 0 by construction, which is the point: this probe
    only rewards knowing how the sequence maps to an *arrangement* of residues.
    """
    n_res = n_nt // k
    if base is None:
        base = rng.integers(0, 4, size=(n_res, k), dtype=np.uint8)
    pool = np.zeros((20 * n, n_nt), dtype=np.uint8)
    for i in range(len(pool)):
        pool[i] = base[rng.permutation(n_res)].reshape(-1)
    if world is not None:
        # Worlds whose rule reads the residue index do not preserve the residue
        # multiset under a codon permutation. Hold it fixed by rejection, so
        # that in every world the probe varies arrangement and nothing else.
        h = world.residues(pool).sum(axis=1)
        keep = pool[h == np.bincount(h).argmax()]
        if len(keep) >= n:
            pool = keep
    return pool[:n]


def off_hull(history_x: np.ndarray, cand: np.ndarray, k: int, frac: float = 0.25) -> np.ndarray:
    """The candidates whose codon composition is furthest from anything the
    agent actually queried. Built after the run, from the run's own history."""
    def comp(x):
        n = len(x)
        c = x.reshape(n, -1, k).astype(np.int64)
        v = np.zeros(c.shape[:2], dtype=np.int64)
        for p in range(k):
            v = v * 4 + c[:, :, p]
        out = np.zeros((n, 4 ** k), dtype=np.float64)
        for i in range(n):
            out[i] = np.bincount(v[i], minlength=4 ** k)
        return out / out.sum(axis=1, keepdims=True)

    h, c = comp(history_x), comp(cand)
    d = np.sqrt(((c[:, None, :] - h[None, :, :]) ** 2).sum(-1)).min(axis=1)
    take = max(1, int(len(cand) * frac))
    return cand[np.argsort(-d)[:take]]
