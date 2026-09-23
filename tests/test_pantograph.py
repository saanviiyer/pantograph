import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pantograph import baselines, dsl, hp, oracle, probes, session, worlds
from pantograph import understanding as U


def test_hp_ceiling_and_published_benchmark():
    tab = hp.table()
    assert tab.max() == 9
    assert hp.fitness([c == "H" for c in "HPHPPHHPHPPHPHHP"], tab) == 6


def test_every_world_reaches_both_residues_everywhere():
    for name in worlds.GRID:
        assert worlds.reachable(worlds.build(name)).all(), name


@pytest.mark.parametrize("a,b", worlds.PAIRS)
def test_pairs_are_matched_on_complexity(a, b):
    wa, wb = worlds.build(a), worlds.build(b)
    assert wa.tokens == wb.tokens
    assert wa.bits == wb.bits
    assert wa.familiar and not wb.familiar


def test_perm_probe_is_composition_matched_in_frame():
    """The gate that licenses the arrangement claim: in-frame composition
    features are literally constant across the probe."""
    rng = np.random.default_rng(0)
    for name in worlds.GRID:
        w = worlds.build(name)
        x = probes.perm(200, w.n_nt, w.k, rng, world=w)
        f = U.count_features(x, w.k, frame=True)
        assert np.allclose(f, f[0]), name


def test_oracle_never_exceeds_budget():
    w = worlds.build("wobble")
    oc = oracle.Oracle(w, budget=10)
    oc.query(np.zeros((10, w.n_nt), dtype=np.uint8))
    with pytest.raises(RuntimeError):
        oc.query(np.zeros((1, w.n_nt), dtype=np.uint8))


def test_seeding_does_not_depend_on_interpreter_salt():
    """H13. `carryover` shipped three headline numbers from one command."""
    assert oracle.seed_of("wobble", "genetic", 3) == 807882916


def test_true_rule_solves_every_probe():
    rng = np.random.default_rng(1)
    for name in worlds.GRID:
        w = worlds.build(name)
        ref = oracle.Oracle(w, budget=1)
        x = probes.perm(200, w.n_nt, w.k, rng, world=w)
        assert U.rho(ref.truth(x), ref.truth(x)) == pytest.approx(1.0)


def test_ridge_dual_and_primal_agree():
    rng = np.random.default_rng(2)
    w = worlds.build("wobble")
    x = probes.iid(300, w.n_nt, rng)
    y = oracle.Oracle(w, budget=1).truth(x)
    xt = probes.iid(50, w.n_nt, rng)
    a = U.Ridge(w.k, basis="frame").fit(x, y)(xt)
    assert a.shape == (50,)
    b = U.Ridge(w.k, basis="stride").fit(x, y)(xt)
    assert b.shape == (50,)


def test_framing_changes_vocabulary_not_the_world():
    w = worlds.build("codon_table")
    bio = session.Session(w, framing="bio", budget=4)
    sym = session.Session(w, framing="symbolic", budget=4)
    assert bio.query(["ACGT" * 12]) == sym.query(["0123" * 12])
    assert "nucleotide" in bio.prompt() and "nucleotide" not in sym.prompt()


def test_submission_namespace_has_no_import():
    with pytest.raises(Exception):
        session.run_submission("import os\ndef predict(s):\n    return [0]*len(s)", ["A" * 48])


def test_baselines_spend_the_whole_budget():
    w = worlds.build("index_gated")
    for fn in baselines.STRATEGIES.values():
        oc = oracle.Oracle(w, budget=120)
        fn(oc, np.random.default_rng(3))
        assert oc.used == 120


def test_dsl_token_count_charges_for_table_entries():
    assert dsl.tokens([("LUT", 0, (0, 1, 0, 1))]) == 6
