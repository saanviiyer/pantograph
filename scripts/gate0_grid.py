"""Gate 0 — the apparatus, before any agent runs.

Every check here can fail. Nothing in this script involves a language model;
its whole purpose is to establish that the worlds, the probes and the null arm
mean what the pre-registration says they mean, so that an agent result later on
is a statement about the agent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pantograph import baselines, hp, oracle, probes, worlds  # noqa: E402
from pantograph import understanding as U  # noqa: E402

BUDGET = 500
SEEDS = 20
PROBE_N = 400
OUT = Path(__file__).resolve().parents[1] / "results"

BENCH16 = ("HPHPPHHPHPPHPHHP", 6)   # published 2D HP 16-mer optimum


def build_probes(w, seed):
    rng = np.random.default_rng(oracle.seed_of("probe", w.name, seed))
    ref = oracle.Oracle(w, budget=1)
    out = {}
    for name, x in (
        ("iid", probes.iid(PROBE_N, w.n_nt, rng)),
        ("perm", probes.perm(PROBE_N, w.n_nt, w.k, rng, world=w)),
        ("ood", probes.ood(PROBE_N, w.n_nt, rng)),
    ):
        out[name] = (x, ref.truth(x))
    return out


def main() -> int:
    res: dict = {"budget": BUDGET, "seeds": SEEDS, "probe_n": PROBE_N}
    fails: list[str] = []

    # G0.1 the physics layer
    tab = hp.table()
    seq, want = BENCH16
    got = hp.fitness([c == "H" for c in seq], tab)
    res["physics"] = {"ceiling": int(tab.max()), "n_at_ceiling": int((tab == tab.max()).sum()),
                      "mean": round(float(tab.mean()), 4),
                      "benchmark_seq": seq, "benchmark_fitness": got, "benchmark_expected": want}
    if tab.max() != 9 or got != want:
        fails.append("G0.1 physics: HP table does not reproduce the published 16-mer optimum")

    rows = []
    for name in worlds.GRID:
        w = worlds.build(name, seed=0)
        row: dict = {"world": name, "familiar": w.familiar, "tokens": w.tokens,
                     "bits": round(w.bits, 2), "note": w.note}

        # G0.2 reachability -> the ceiling is the same 9 in every world
        reach = worlds.reachable(w)
        row["reach_all"] = bool(reach.all())
        if not reach.all():
            fails.append(f"G0.2 reachability: {name} cannot make both residues at every position")

        # G0.3 searchability, at the agent's budget, with no understanding
        for strat, fn in baselines.STRATEGIES.items():
            best = []
            for s in range(SEEDS):
                oc = oracle.Oracle(w, budget=BUDGET)
                fn(oc, np.random.default_rng(oracle.seed_of(name, strat, s)))
                best.append(oc.best)
            row[f"{strat}_median"] = float(np.median(best))
            row[f"{strat}_max"] = int(np.max(best))
            row[f"{strat}_solved"] = float(np.mean(np.asarray(best) >= 9))

        # G0.4 the null arm is measured, not assumed.
        # Two nulls: a composition-only ridge, which the `perm` probe is built
        # to pin to a constant, and a position-aware ridge, which is the real
        # null the agent has to beat. The first is a pass/fail; the second is
        # reported, not thresholded — it is allowed to be informative, and the
        # understanding metric is the margin over it.
        nulls = {p: [] for p in ("iid", "perm", "ood")}
        cnulls = {p: [] for p in ("iid", "perm", "ood")}
        fnull: list[float] = []
        for s in range(5):
            oc = oracle.Oracle(w, budget=BUDGET)
            baselines.random_search(oc, np.random.default_rng(oracle.seed_of(name, "null", s)))
            hx, hy = oc.history
            sur = U.Ridge(w.k, basis="stride").fit(hx, hy)
            fsur = U.Ridge(w.k, basis="frame").fit(hx, hy)
            csur = U.Ridge(w.k, basis="counts_frame").fit(hx, hy)
            for p, (px, py) in build_probes(w, s).items():
                nulls[p].append(U.rho(sur(px), py))
                cnulls[p].append(U.rho(csur(px), py))
                if p == "perm":
                    fnull.append(U.rho(fsur(px), py))
        for p, v in nulls.items():
            row[f"null_rho_{p}"] = round(float(np.median(v)), 3)
        row["counts_rho_perm"] = round(float(np.median(cnulls["perm"])), 3)
        row["frame_null_rho_perm"] = round(float(np.median(fnull)), 3)
        row["counts_rho_iid"] = round(float(np.median(cnulls["iid"])), 3)
        if abs(row["counts_rho_perm"]) > 1e-9:
            fails.append(f"G0.4 probe: composition still ranks {name}/perm "
                         f"({row['counts_rho_perm']}) — the probe is not composition-matched")

        # G0.5 the probe is solvable by a rule, and has spread
        pr = build_probes(w, 0)
        row["perm_sd"] = round(float(pr["perm"][1].std()), 3)
        row["rule_rho_perm"] = round(U.rho(oracle.Oracle(w, budget=1).truth(pr["perm"][0]),
                                           pr["perm"][1]), 3)
        if row["rule_rho_perm"] < 0.999:
            fails.append(f"G0.5 probe: the true rule does not solve {name}/perm")
        if row["perm_sd"] < 0.4:
            fails.append(f"G0.5 probe: {name}/perm has no fitness spread to rank")
        row["headroom_perm"] = round(row["rule_rho_perm"] - row["null_rho_perm"], 3)
        if row["headroom_perm"] < 0.5:
            fails.append(f"G0.5 probe: only {row['headroom_perm']} of headroom over the "
                         f"surrogate on {name}/perm; the metric cannot resolve understanding")
        rows.append(row)

    res["worlds"] = rows

    # G0.7 the null arm must not carry the prior under test.
    # If the in-frame basis is much better in the familiar world of a pair than
    # in its unfamiliar twin, then a gap measured against it would report the
    # basis's prior as the agent's. The frame-agnostic basis is the one the
    # headroom gate uses; this check reports how much that mattered.
    res["null_basis_asymmetry"] = []
    for a, b in worlds.PAIRS:
        ra = next(r for r in rows if r["world"] == a)
        rb = next(r for r in rows if r["world"] == b)
        res["null_basis_asymmetry"].append({
            "pair": [a, b],
            "frame_basis": [ra["frame_null_rho_perm"], rb["frame_null_rho_perm"]],
            "stride_basis": [ra["null_rho_perm"], rb["null_rho_perm"]],
            "frame_gap": round(ra["frame_null_rho_perm"] - rb["frame_null_rho_perm"], 3),
            "stride_gap": round(ra["null_rho_perm"] - rb["null_rho_perm"], 3)})

    # G0.6 the matched pairs really are matched on bits
    res["pairs"] = []
    for a, b in worlds.PAIRS:
        ra = next(r for r in rows if r["world"] == a)
        rb = next(r for r in rows if r["world"] == b)
        d = {"familiar": a, "unfamiliar": b,
             "bits": [ra["bits"], rb["bits"]], "tokens": [ra["tokens"], rb["tokens"]],
             "genetic_median": [ra["genetic_median"], rb["genetic_median"]],
             "null_rho_perm": [ra["null_rho_perm"], rb["null_rho_perm"]]}
        d["bits_matched"] = abs(ra["bits"] - rb["bits"]) <= 0.05 * max(ra["bits"], rb["bits"])
        d["search_matched"] = abs(ra["genetic_median"] - rb["genetic_median"]) <= 1.0
        res["pairs"].append(d)
        if not d["bits_matched"]:
            fails.append(f"G0.6 pairing: {a}/{b} are not matched on bits")
        if not d["search_matched"]:
            fails.append(f"G0.6 pairing: {a}/{b} are not matched on searchability")

    res["fails"] = fails
    res["verdict"] = "PASS" if not fails else "FAIL"
    OUT.mkdir(exist_ok=True)
    (OUT / "gate0_grid.json").write_text(json.dumps(res, indent=2))

    hdr = (f"{'world':14s} {'fam':>3s} {'tok':>4s} {'bits':>6s} | {'rand':>5s} {'hill':>5s} "
           f"{'GA':>4s} {'GA@9':>5s} | {'n_iid':>6s} {'n_perm':>7s} {'cnt_perm':>9s} {'head':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['world']:14s} {str(r['familiar'])[0]:>3s} {r['tokens']:4d} {r['bits']:6.2f} | "
              f"{r['random_median']:5.1f} {r['hill_climb_median']:5.1f} {r['genetic_median']:4.1f} "
              f"{r['genetic_solved']:5.2f} | {r['null_rho_iid']:6.3f} {r['null_rho_perm']:7.3f} "
              f"{r['counts_rho_perm']:9.3f} {r['headroom_perm']:5.2f}")
    print(f"\nverdict: {res['verdict']}")
    for f in fails:
        print("  FAIL " + f)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
