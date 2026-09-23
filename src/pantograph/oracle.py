"""The sandbox: a sealed world, a query budget, and a logged history.

The agent sees fitness and nothing else. It never sees residues, the program,
the table, or the identity of the world. The oracle is the only channel.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import hp
from .worlds import World

BASES = "ACGT"


def seed_of(*parts) -> int:
    """Stable across processes. `carryover` shipped three headline numbers from
    one command because this was Python's salted hash()."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") % (2 ** 32)


def encode(seqs: list[str]) -> np.ndarray:
    lut = np.full(256, 255, dtype=np.uint8)
    for i, b in enumerate(BASES):
        lut[ord(b)] = i
    a = np.frombuffer("".join(seqs).encode(), dtype=np.uint8)
    out = lut[a].reshape(len(seqs), -1)
    if (out == 255).any():
        raise ValueError("sequence contains a character outside ACGT")
    return out


def decode(x: np.ndarray) -> list[str]:
    return ["".join(BASES[i] for i in row) for row in np.asarray(x)]


@dataclass
class Oracle:
    world: World
    budget: int = 500
    tab: np.ndarray = field(default_factory=hp.table, repr=False)
    used: int = 0
    history_x: list = field(default_factory=list, repr=False)
    history_y: list = field(default_factory=list, repr=False)

    def truth(self, x: np.ndarray) -> np.ndarray:
        """Fitness without spending budget. For probes and gates only; never
        reachable from the agent's side of the boundary."""
        r = self.world.residues(np.asarray(x, dtype=np.uint8))
        codes = (r.astype(np.uint32) << np.arange(hp.N_RES, dtype=np.uint32)).sum(axis=1)
        return self.tab[codes].astype(np.int64)

    def query(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.uint8))
        if self.used + len(x) > self.budget:
            raise RuntimeError(f"budget exhausted: {self.used}/{self.budget}")
        y = self.truth(x)
        self.used += len(x)
        self.history_x.append(x.copy())
        self.history_y.append(y.copy())
        return y

    @property
    def history(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.history_x:
            return (np.zeros((0, self.world.n_nt), dtype=np.uint8),
                    np.zeros(0, dtype=np.int64))
        return np.concatenate(self.history_x), np.concatenate(self.history_y)

    @property
    def best(self) -> int:
        _, y = self.history
        return int(y.max()) if len(y) else -1

    def dump(self, path: Path) -> None:
        x, y = self.history
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for s, v in zip(decode(x), y.tolist()):
                fh.write(json.dumps({"seq": s, "fitness": int(v)}) + "\n")
