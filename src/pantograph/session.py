"""The agent's side of the boundary.

Two things are supplied to the agent and nothing else: a query channel with a
budget, and a notebook. Two things are asked back: the queries, and — at fixed
checkpoints — an *executable* hypothesis, a function from sequences to
predicted fitness.

The executable hypothesis is the whole methodological move. The published
sandbox scores reasoning by reading the notebook, which is a one-channel
adjudicator with no null arm: an agent can narrate a rule it never used, or
exploit one it never states. A predictor can be scored on sequences the agent
never queried, against a surrogate fitted to the agent's own queries. The
notebook is still collected, because the disagreement between what the notebook
claims and what the predictor achieves is itself a measurement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .oracle import Oracle, decode, encode
from .worlds import World

FRAMINGS = {
    "bio": {
        "alphabet": "ACGT",
        "unit": "nucleotide",
        "chain": "protein",
        "monomer": "amino acid",
        "score": "folding energy",
        "preamble": (
            "You are studying an organism whose genetics are not yet known. A "
            "{chain} is built from a {seq_len}-{unit} gene. Each of the 16 {monomer}s "
            "in the chain is one of two types, and the {score} of the folded chain "
            "is what the assay returns. How the gene specifies the {monomer}s has "
            "not been determined."),
    },
    "symbolic": {
        "alphabet": "0123",
        "unit": "token",
        "chain": "object",
        "monomer": "element",
        "score": "score",
        "preamble": (
            "A black box takes a string of {seq_len} {unit}s from the alphabet "
            "{alphabet} and returns an integer {score}. The box is deterministic. "
            "Nothing else about it is specified."),
    },
}

TASK = """
You have a budget of {budget} queries. A query is a batch of strings; each
string costs one query.

Each round you may:
  QUERY <one string per line>      spend budget, receive one integer per string
  NOTE <free text>                 write in your notebook; costs nothing
  PREDICT <python>                 submit your current hypothesis as code

A PREDICT submission defines `predict(seqs)` taking a list of strings and
returning one number per string, higher meaning higher {score}. It may use only
the standard library. It is scored on strings you have never queried, so a
lookup table of what you have already seen will score zero. Submit one at each
checkpoint: after {checkpoints} queries.

Two things are being measured and they are not the same thing: the best {score}
you find, and how well your submitted `predict` ranks sequences you never saw.
"""


@dataclass
class Session:
    world: World
    framing: str = "bio"
    budget: int = 500
    checkpoints: tuple[int, ...] = (50, 150, 500)
    oracle: Oracle = None  # type: ignore[assignment]
    notebook: list = field(default_factory=list)
    submissions: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.oracle is None:
            self.oracle = Oracle(self.world, budget=self.budget)

    @property
    def alphabet(self) -> str:
        return FRAMINGS[self.framing]["alphabet"]

    def prompt(self) -> str:
        f = dict(FRAMINGS[self.framing])
        f["seq_len"] = self.world.n_nt
        f["budget"] = self.budget
        f["checkpoints"] = ", ".join(str(c) for c in self.checkpoints)
        return (f["preamble"].format(**f) + "\n" + TASK.format(**f)).strip()

    def query(self, seqs: list[str]) -> list[int]:
        native = [self._to_acgt(s) for s in seqs]
        return [int(v) for v in self.oracle.query(encode(native))]

    def note(self, text: str) -> None:
        self.notebook.append({"at": self.oracle.used, "text": text})

    def predict_submission(self, src: str) -> None:
        self.submissions.append({"at": self.oracle.used, "src": src})

    def _to_acgt(self, s: str) -> str:
        if self.framing == "bio":
            return s
        return "".join("ACGT"[self.alphabet.index(c)] for c in s)

    def _from_acgt(self, s: str) -> str:
        if self.framing == "bio":
            return s
        return "".join(self.alphabet["ACGT".index(c)] for c in s)

    def dump(self, path: Path) -> None:
        x, y = self.oracle.history
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "world": self.world.name, "framing": self.framing,
            "budget": self.budget, "used": self.oracle.used,
            "best": self.oracle.best,
            "queries": [{"seq": self._from_acgt(s), "fitness": int(v)}
                        for s, v in zip(decode(x), y.tolist())],
            "notebook": self.notebook, "submissions": self.submissions}, indent=2))


def run_submission(src: str, seqs: list[str], timeout_calls: int = 1) -> np.ndarray:
    """Execute a submitted predictor in a restricted namespace.

    This runs model-written code. It is run with no builtins beyond a small
    allow-list and no filesystem or network names in scope, but that is a
    guard, not a sandbox: run it in a container when the agent arm is live.
    """
    import builtins

    allowed = {n: getattr(builtins, n) for n in (
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "float",
        "int", "len", "list", "map", "max", "min", "pow", "range", "reversed",
        "round", "set", "sorted", "str", "sum", "tuple", "zip", "print",
        "isinstance", "ValueError", "IndexError", "KeyError", "TypeError",
        "Exception", "__build_class__", "__name__")}
    ns: dict = {"__builtins__": allowed}
    exec(compile(src, "<submission>", "exec"), ns)
    fn = ns.get("predict")
    if fn is None:
        raise ValueError("submission defines no predict()")
    out = fn(list(seqs))
    a = np.asarray(list(out), dtype=np.float64)
    if a.shape != (len(seqs),):
        raise ValueError(f"predict returned {a.shape}, expected {(len(seqs),)}")
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
