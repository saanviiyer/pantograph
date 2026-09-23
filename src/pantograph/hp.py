"""The physics layer: 2D HP lattice fitness for 16-mers.

This module is *scenery*, in the sense of HARNESS P1. Nothing about the
contribution depends on the HP model being a good model of folding; it is here
because it supplies a fitness function that is (a) fixed, (b) published, (c)
identical across every world, so that any difference between worlds is a
difference in the hidden encoding and not in the physics.

Fitness of an H/P string is the maximum, over all self-avoiding walks of the
chain on the square lattice, of the number of H-H contacts between residues
that are lattice-adjacent but not chain-adjacent. For N = 16 the optimum is 9.

The whole 2**16 table is precomputed once and cached.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

N_RES = 16
CACHE = Path(__file__).resolve().parents[2] / "cache" / "hp16.npz"

_DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def _pair_index() -> dict[tuple[int, int], int]:
    """Non-consecutive residue pairs, numbered."""
    return {(i, j): n for n, (i, j) in enumerate(
        (i, j) for i in range(N_RES) for j in range(i + 2, N_RES))}


def enumerate_contact_maps() -> list[int]:
    """Distinct contact maps over self-avoiding walks of N_RES nodes.

    Returned as bitmasks over the non-consecutive pair index. Reflections and
    rotations of a walk have identical contact maps, so the first step is fixed
    to East and the first turn to North without loss.
    """
    pidx = _pair_index()
    seen: set[int] = set()
    occ: dict[tuple[int, int], int] = {(0, 0): 0}
    path = [(0, 0)]

    def walk(n: int, mask: int, turned: bool) -> None:
        if n == N_RES:
            seen.add(mask)
            return
        x, y = path[-1]
        for d, (dx, dy) in enumerate(_DIRS):
            if n == 1 and d != 0:
                continue                       # first step East
            if not turned and d == 3:
                continue                       # first turn is North, not South
            p = (x + dx, y + dy)
            if p in occ:
                continue
            add = 0
            for ex, ey in _DIRS:
                q = (p[0] + ex, p[1] + ey)
                m = occ.get(q)
                if m is not None and m < n - 1:
                    add |= 1 << pidx[(m, n)]
            occ[p] = n
            path.append(p)
            walk(n + 1, mask | add, turned or d != 0)
            path.pop()
            del occ[p]

    walk(1, 0, False)
    return sorted(seen, key=lambda m: -bin(m).count("1"))


def maximal(masks: list[int]) -> list[int]:
    """Drop any contact map that is a subset of another; it can never win."""
    keep: list[int] = []
    for m in masks:
        if not any(m & k == m for k in keep):
            keep.append(m)
    return keep


def _build() -> np.ndarray:
    masks = maximal(enumerate_contact_maps())
    pidx = _pair_index()
    inv = {v: k for k, v in pidx.items()}

    codes = np.arange(1 << N_RES, dtype=np.uint32)
    bits = ((codes[:, None] >> np.arange(N_RES)[None, :]) & 1).astype(np.uint8)

    best = np.zeros(1 << N_RES, dtype=np.uint8)
    for m in sorted(masks, key=lambda m: -bin(m).count("1")):
        p = bin(m).count("1")
        active = np.flatnonzero(best < p)
        if active.size == 0:
            break
        pairs = [inv[b] for b in range(len(pidx)) if (m >> b) & 1]
        i = np.fromiter((q[0] for q in pairs), dtype=np.int64)
        j = np.fromiter((q[1] for q in pairs), dtype=np.int64)
        sub = bits[active]
        cnt = (sub[:, i] & sub[:, j]).sum(axis=1).astype(np.uint8)
        np.maximum(best[active], cnt, out=cnt)
        best[active] = cnt
    return best


def table() -> np.ndarray:
    """uint8[65536]: fitness of every H/P string, bit r set == residue r is H."""
    if CACHE.exists():
        return np.load(CACHE)["fitness"]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    fit = _build()
    np.savez_compressed(CACHE, fitness=fit)
    return fit


def code(residues) -> int:
    """H/P sequence (1 == H) to table index."""
    return int(sum(int(b) << r for r, b in enumerate(residues)))


def fitness(residues, tab: np.ndarray | None = None) -> int:
    return int((table() if tab is None else tab)[code(residues)])
