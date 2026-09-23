"""Score one agent session on both axes.

Reads a session dump written by `pantograph.session.Session.dump` and reports,
for every checkpoint submission, the two numbers the project exists to keep
apart: what the agent found, and what the agent understood.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pantograph import oracle, probes, session, worlds  # noqa: E402
from pantograph import understanding as U  # noqa: E402

PROBE_N = 400


def build_probes(w, seed: int):
    rng = np.random.default_rng(oracle.seed_of("probe", w.name, seed))
    ref = oracle.Oracle(w, budget=1)
    out = {}
    for name, x in (("iid", probes.iid(PROBE_N, w.n_nt, rng)),
                    ("perm", probes.perm(PROBE_N, w.n_nt, w.k, rng, world=w)),
                    ("ood", probes.ood(PROBE_N, w.n_nt, rng))):
        out[name] = (x, ref.truth(x))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", type=Path)
    ap.add_argument("--probe-seed", type=int, default=0)
    ap.add_argument("--world-seed", type=int, default=0)
    a = ap.parse_args()

    d = json.loads(a.dump.read_text())
    w = worlds.build(d["world"], seed=a.world_seed)
    P = build_probes(w, a.probe_seed)
    sess = session.Session(w, framing=d["framing"], budget=d["budget"])

    hx = oracle.encode([sess._to_acgt(q["seq"]) for q in d["queries"]])
    hy = np.asarray([q["fitness"] for q in d["queries"]], dtype=np.int64)

    rows = []
    for sub in d["submissions"]:
        at = sub["at"]
        null = U.Ridge(w.k, basis="stride").fit(hx[:at], hy[:at])
        row = {"at": at, "best_so_far": int(hy[:at].max())}
        for pname, (px, py) in P.items():
            names = [sess._from_acgt(s) for s in oracle.decode(px)]
            try:
                pred = session.run_submission(sub["src"], names)
                ar = U.rho(pred, py)
                atk = U.top_k_precision(pred, py)
            except Exception as exc:                     # a broken submission is a result
                ar, atk = 0.0, 0.0
                row.setdefault("errors", []).append(f"{pname}: {type(exc).__name__}: {exc}")
            nr = U.rho(null(px), py)
            row[pname] = {"agent_rho": round(ar, 3), "null_rho": round(nr, 3),
                          "gap": round(ar - nr, 3), "agent_top10": round(atk, 3)}
        rows.append(row)

    out = {"world": d["world"], "familiar": w.familiar, "framing": d["framing"],
           "bits": w.bits, "tokens": w.tokens,
           "best": int(hy.max()), "used": int(len(hy)), "checkpoints": rows}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
