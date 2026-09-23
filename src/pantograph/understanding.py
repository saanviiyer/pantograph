"""The understanding axis.

The published sandbox scores scientific reasoning by reading the agent's lab
notebook: "the central object of evaluation is the agent's reasoning, not
simply its final score". Under HARNESS that is a single-channel adjudicator
with no null arm — a notebook can state a rule the agent never used, and an
agent can exploit a rule it never states.

Here understanding is a prediction, scored out of sample:

  the agent submits an executable predictor  f: sequence -> fitness
  it is scored on probe sets it never queried
  and it is scored *against a null*: a ridge surrogate fitted to the agent's
  own query history, which by construction has no understanding at all.

`understanding = rho(agent) - rho(surrogate)`, reported separately on the IID
and the OOD probe. The OOD column is the one that can only be bought with a
rule; this is the same readout-versus-representation distinction `palimpsest`
turns on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


def features(x: np.ndarray, k: int) -> np.ndarray:
    """Position-wise one-hot over nucleotides, plus position-wise one-hot over
    codons. Strictly more expressive than the 6-mer ridge the published dry
    oracles score with, so the null arm is not a straw man."""
    x = np.asarray(x, dtype=np.int64)
    n, n_nt = x.shape
    nt = np.zeros((n, n_nt, 4), dtype=np.float64)
    nt[np.arange(n)[:, None], np.arange(n_nt)[None, :], x] = 1.0
    c = x.reshape(n, -1, k)
    v = np.zeros(c.shape[:2], dtype=np.int64)
    for p in range(k):
        v = v * 4 + c[:, :, p]
    n_res, n_cod = v.shape[1], 4 ** k
    cod = np.zeros((n, n_res, n_cod), dtype=np.float64)
    cod[np.arange(n)[:, None], np.arange(n_res)[None, :], v] = 1.0
    return np.concatenate([nt.reshape(n, -1), cod.reshape(n, -1)], axis=1)


def stride_features(x: np.ndarray, k: int) -> np.ndarray:
    """Frame-agnostic: position-wise nucleotides, plus every k-tuple of
    positions at every start and every dilation in 1..k-1, wrapping.

    The in-frame basis carries the same biological prior the experiment is
    trying to measure — it can express "codon i" and cannot express "the triple
    that straddles codon i". A null arm built on it is weaker in exactly the
    worlds the hypothesis says the agent should be weaker in, which would hand
    the pair contrast a confound (HARNESS H8). This basis contains the reading
    used by every world in the grid, and contains none of them preferentially.
    """
    x = np.asarray(x, dtype=np.int64)
    n, n_nt = x.shape
    nt = np.zeros((n, n_nt, 4), dtype=np.float64)
    nt[np.arange(n)[:, None], np.arange(n_nt)[None, :], x] = 1.0
    blocks = [nt.reshape(n, -1)]
    rows, cols = np.arange(n)[:, None], np.arange(n_nt)[None, :]
    for dil in range(1, k):
        v = np.zeros((n, n_nt), dtype=np.int64)
        for p in range(k):
            v = v * 4 + x[:, (np.arange(n_nt) + p * dil) % n_nt]
        km = np.zeros((n, n_nt, 4 ** k), dtype=np.float64)
        km[rows, cols, v] = 1.0
        blocks.append(km.reshape(n, -1))
    return np.concatenate(blocks, axis=1)


def count_features(x: np.ndarray, k: int, frame: bool = True) -> np.ndarray:
    """Composition only, no position.

    With `frame=True` these are nucleotide counts and in-frame codon counts,
    every one of which is constant across the `perm` probe by construction, so
    a predictor built on them cannot rank it at all. Gate 0 asserts exactly
    that, and it is what licenses the claim that the probe isolates
    arrangement. With `frame=False` the counts are taken over all offsets and
    dilations, which a codon permutation does disturb slightly; that version is
    reported with a band rather than asserted at zero.
    """
    f = (features if frame else stride_features)(x, k)
    n, n_nt = np.asarray(x).shape
    nt = f[:, : n_nt * 4].reshape(n, n_nt, 4).sum(axis=1)
    rest = f[:, n_nt * 4:]
    km = rest.reshape(n, -1, 4 ** k).sum(axis=1)
    return np.concatenate([nt, km], axis=1)


@dataclass
class Ridge:
    k: int
    lam: float = 1.0
    w: np.ndarray | None = None
    mu: float = 0.0

    basis: str = "stride"          # "stride" (frame-agnostic) | "frame" | "counts"

    xfit: np.ndarray | None = None

    def _F(self, x: np.ndarray) -> np.ndarray:
        if self.basis == "stride":
            return stride_features(x, self.k)
        if self.basis == "frame":
            return features(x, self.k)
        if self.basis == "counts_frame":
            return count_features(x, self.k, frame=True)
        if self.basis == "counts":
            return count_features(x, self.k, frame=False)
        raise ValueError(self.basis)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Ridge":
        F = self._F(x)
        self.mu = float(np.mean(y))
        yc = np.asarray(y, dtype=np.float64) - self.mu
        if F.shape[1] > F.shape[0]:
            # Dual form. The frame-agnostic basis has more columns than the
            # agent has queries, and solving in the primal would be both slow
            # and worse conditioned.
            K = F @ F.T + self.lam * np.eye(F.shape[0])
            self.w = np.linalg.solve(K, yc)
            self.xfit = np.asarray(x, dtype=np.uint8)
        else:
            A = F.T @ F + self.lam * np.eye(F.shape[1])
            self.w = np.linalg.solve(A, F.T @ yc)
            self.xfit = None
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        F = self._F(x)
        if self.xfit is not None:
            return F @ self._F(self.xfit).T @ self.w + self.mu
        return F @ self.w + self.mu


def rho(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    if np.allclose(pred, pred[0]):
        return 0.0
    return float(spearmanr(pred, true).statistic)


def top_k_precision(pred: np.ndarray, true: np.ndarray, k: int = 10) -> float:
    """H12: the decision, not the average. A predictor is consumed as a ranking
    of what to build next, so the tail is reported beside the bulk metric."""
    order = np.argsort(-np.asarray(pred, dtype=np.float64))[:k]
    cut = np.sort(true)[-k]
    return float((np.asarray(true)[order] >= cut).mean())


def evaluate(predict, probes: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    out = {}
    for name, (px, py) in probes.items():
        p = np.asarray(predict(px), dtype=np.float64)
        out[name] = {"rho": rho(p, py), "top10": top_k_precision(p, py)}
    return out
