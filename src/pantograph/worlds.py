"""The rule grid.

The science-sandboxes result this project is built on reports that agent
reasoning "deteriorated when they encountered systems whose rules fell outside
familiar biological priors". In the published grid, familiarity is not
separated from description length or from where in the input the rule reads:
`GC balance` is biological *and* short *and* compositional, `Collatz` is
unfamiliar *and* long *and* arithmetic. HARNESS H8 says a quantity the verdict
could be a function of, other than the intended one, has to be named and held
fixed. Complexity is that quantity.

So the grid is two matched pairs. Within a pair the rule template, the token
count and the number of bits to be determined are identical; the only thing
that changes is whether the quantity the rule reads is one a biologist would
reach for.

  familiar, 4 bits, 8 tokens    wobble        table over base 0 of codon i
  unfamiliar, 4 bits, 8 tokens  offset_base   table over base 0 of codon i+1

  familiar, 64 bits, 78 tokens  codon_table   table over the three bases of
                                              codon i
  unfamiliar, 64 bits, 78 tok.  offset_frame  table over three bases read at
                                              stride 2, across codon boundaries

Within a pair the two worlds are the *same program* with the read offsets
changed. Identical token count, identical number of bits to determine,
identical fitness ceiling, and — Gate 0 checks this — indistinguishable to a
hill-climber. The only thing that differs is whether the rule respects the
reading frame. That is as close as this design can get to varying "familiar
biological prior" and nothing else, and it is not an accident that the contrast
is a frameshift: in `crosstalk`, frameshift was the only control that turned
out to be decisive about codon-level leakage.

`xor_pair` and `index_gated` are not part of a pair. They are unfamiliar rules
of the kind the published grid is full of — a bitwise operation and a modular
one — kept so the project has cells that resemble the prior work directly, and
so that the frame contrast can be checked against a second axis of
unfamiliarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log2

import numpy as np

from . import dsl, hp

N_RES = hp.N_RES


@dataclass(frozen=True)
class World:
    name: str
    familiar: bool
    prog: list
    k: int                      # nucleotides per residue
    j: int = 4                  # alphabet size
    bits: float = 0.0           # bits to determine the rule within its family
    note: str = ""

    @property
    def n_nt(self) -> int:
        return N_RES * self.k

    @property
    def tokens(self) -> int:
        return dsl.tokens(self.prog)

    def residues(self, x: np.ndarray) -> np.ndarray:
        return dsl.run(self.prog, x, N_RES, self.k, self.j)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _balanced_table(n: int, seed: int) -> tuple[int, ...]:
    """A binary table with exactly half H, so composition carries no signal."""
    t = np.zeros(n, dtype=np.int64)
    t[: n // 2] = 1
    _rng(seed).shuffle(t)
    return tuple(int(v) for v in t)


def wobble(seed: int = 0) -> World:
    t = _balanced_table(4, seed)
    prog = [("NT", 0), ("LUT", 0, t)]
    return World("wobble", True, prog, k=3, bits=4.0,
                 note="first base decides; the other two are silent")


def xor_pair(seed: int = 0) -> World:
    t = _balanced_table(4, seed)
    prog = [("NT", 0), ("NT", 2), ("XOR", 0, 1), ("LUT", 2, t)]
    return World("xor_pair", False, prog, k=3, bits=4.0,
                 note="first XOR third base decides; the middle base is silent")


def offset_base(seed: int = 0) -> World:
    t = _balanced_table(4, seed)
    prog = [("NTG", 3), ("LUT", 0, t)]
    return World("offset_base", False, prog, k=3, bits=4.0,
                 note="the first base of the *next* codon decides; wraps at the end")


def codon_table(seed: int = 0) -> World:
    t = _balanced_table(64, seed)
    prog = [("NTG", 0), ("NTG", 1), ("NTG", 2),
            ("PACK", 0, 1), ("PACK", 3, 2), ("LUT", 4, t)]
    return World("codon_table", True, prog, k=3, bits=64.0,
                 note="an arbitrary balanced table over all 64 codons")


def offset_frame(seed: int = 0) -> World:
    t = _balanced_table(64, seed)
    prog = [("NTG", 0), ("NTG", 2), ("NTG", 4),
            ("PACK", 0, 1), ("PACK", 3, 2), ("LUT", 4, t)]
    return World("offset_frame", False, prog, k=3, bits=64.0,
                 note="the same size of table, read at bases 0, 2 and 4 from the "
                      "codon start, so the triple straddles the next codon")


def index_gated(seed: int = 0, m: int = 5, t_thr: int = 2) -> World:
    prog = [("NT", 0), ("NT", 1), ("NT", 2),
            ("PACK", 0, 1), ("PACK", 3, 2),
            ("IDX",), ("ADD", 4, 5),
            ("CONST", m), ("MOD", 6, 7),
            ("CONST", t_thr), ("LT", 8, 9)]
    space = sum(mm - 1 for mm in range(3, 9))
    return World("index_gated", False, prog, k=3, bits=log2(space),
                 note="H when (codon value + position) mod m is below t")


GRID = {
    "wobble": wobble,
    "offset_base": offset_base,
    "codon_table": codon_table,
    "offset_frame": offset_frame,
    "xor_pair": xor_pair,
    "index_gated": index_gated,
}

PAIRS = [("wobble", "offset_base"), ("codon_table", "offset_frame")]


def build(name: str, seed: int = 0) -> World:
    return GRID[name](seed)


def reachable(w: World) -> np.ndarray:
    """bool[n_res, 2]: can residue i be made P (0) and H (1)?

    Every family in the grid decodes residue i from codon i and i alone, so
    this is exact rather than sampled: enumerate all j**k codons at each
    position. If both are reachable everywhere then every one of the 2**16 H/P
    strings is reachable and the world's fitness ceiling is the model's own, 9.

    Offset worlds read across a codon boundary, so a constant-codon probe is
    not enough to certify them; random sequences are added and the union is
    taken. The check is still a lower bound that can fail, which is the point.
    """
    n_codon = w.j ** w.k
    xs = np.zeros((n_codon, w.n_nt), dtype=np.uint8)
    for c in range(n_codon):
        digits = [(c // (w.j ** (w.k - 1 - p))) % w.j for p in range(w.k)]
        xs[c] = np.tile(digits, N_RES)
    extra = np.random.default_rng(20260902).integers(
        0, w.j, size=(20000, w.n_nt), dtype=np.uint8)
    r = w.residues(np.concatenate([xs, extra]))
    return np.stack([(~r).any(axis=0), r.any(axis=0)], axis=1)
