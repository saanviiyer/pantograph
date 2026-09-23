"""A tiny register machine for hidden encoding rules.

A world's rule is a *program*, not a Python function, so that two things the
project depends on are computable rather than asserted:

  - `tokens(prog)`   description length, in instruction + argument tokens
  - `bits(world)`    how many bits of the world an agent has to determine to
                     pin the rule down within its own family template

`crosstalk` and `weathervane` both paid for reporting a quantity that was a
function of a design choice nobody had written down. Here the design choice is
the rule's complexity, and the whole contribution is a contrast that holds it
fixed while familiarity varies, so it is measured in the code.

Programs run vectorised: input is uint8[B, n_res * k], output is bool[B, n_res].
"""

from __future__ import annotations

import numpy as np

BINARY = {
    "ADD": lambda a, b: a + b,
    "MUL": lambda a, b: a * b,
    "MOD": lambda a, b: np.remainder(a, np.maximum(b, 1)),
    "XOR": lambda a, b: np.bitwise_xor(a, b),
    "EQ": lambda a, b: (a == b).astype(np.int64),
    "LT": lambda a, b: (a < b).astype(np.int64),
}

Instr = tuple  # ("OP", args...)


def tokens(prog: list[Instr]) -> int:
    """Description length: one token per opcode, one per argument.

    A lookup table's entries are arguments, so a 64-entry table costs 64. That
    is the point: an arbitrary codon table really is a long rule, and any claim
    that unfamiliar rules are harder has to survive saying so.
    """
    n = 0
    for ins in prog:
        n += 1
        for a in ins[1:]:
            n += len(a) if isinstance(a, (tuple, list)) else 1
    return n


def run(prog: list[Instr], x: np.ndarray, n_res: int, k: int, j: int) -> np.ndarray:
    """Evaluate a program. Returns bool[B, n_res]; True == H."""
    x3 = x.reshape(len(x), n_res, k).astype(np.int64)
    idx = np.broadcast_to(np.arange(n_res)[None, :], x3.shape[:2])
    reg: list[np.ndarray] = []
    for ins in prog:
        op, args = ins[0], ins[1:]
        if op == "NT":
            reg.append(x3[:, :, args[0]])
        elif op == "NTG":
            # A nucleotide at an absolute offset from this residue's codon
            # start, wrapping at the end of the sequence. Offsets inside
            # [0, k) read in frame; anything else reads across a codon
            # boundary, which is the only thing the matched pairs vary.
            pos = (np.arange(n_res) * k + args[0]) % (n_res * k)
            reg.append(x[:, pos].astype(np.int64))
        elif op == "IDX":
            reg.append(idx)
        elif op == "CONST":
            reg.append(np.full(x3.shape[:2], args[0], dtype=np.int64))
        elif op == "PACK":
            reg.append(reg[args[0]] * j + reg[args[1]])
        elif op == "LUT":
            reg.append(np.asarray(args[1], dtype=np.int64)[reg[args[0]]])
        elif op in BINARY:
            # Binary ops take two register indices. Constants go through CONST,
            # so an argument is never ambiguous between a value and a slot.
            reg.append(BINARY[op](reg[args[0]], reg[args[1]]))
        else:  # pragma: no cover - guarded by the world constructors
            raise ValueError(f"unknown op {op}")
    return reg[-1].astype(bool)
