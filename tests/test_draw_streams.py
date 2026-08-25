"""Draws from two separately fitted cohorts must not share one noise stream.

The defect this file was written for: every gallery entry's ``predict`` and the
shared ``predict_at`` used to start ``np.random.default_rng(seed)`` from
scratch, once per fitted cohort. A study that fits twenty-five companies and
passes one seed - which is exactly what the notebooks in ``analysis/`` do - then
had draw ``i`` of every company reading the same underlying random numbers, so
the companies' ultimates moved up and down together. Notebook 3c measured a mean
implied cross-cohort correlation of 0.255 for ``mack`` where independent draws
would give roughly 0.01 of sampling noise at 10,000 draws.

Each cohort's own distribution was never wrong. What was wrong is anything read
across cohorts *within a draw*: a company total, a panel total, the spread of
either, and any calibration statistic computed from those sums.

The tests below come in two layers.

**Behavior.** Fit two genuinely different cohorts - different segment identity
AND different loss amounts - pass both the same seed, and require the two draw
vectors to be uncorrelated. The threshold is ``abs(corr) < 0.1``; at 4,000 draws
one sampling standard deviation of a sample correlation is about 0.016, so 0.1
is more than six of them away from zero and the test does not flicker. Each of
these assertions fails on the pre-fix code, with the measured correlations noted
in the individual docstrings. Every behavior test also repeats the call and
requires the draws back bit for bit, because a derived stream that is not
reproducible would trade one defect for a worse one.

**Unit.** ``kernels.rng`` itself: which arguments change a stream, which do not,
and a frozen entropy list that pins the canonical text across platforms and
releases.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ibnr import gallery
from ibnr.gallery.statistical.clark.model import Clark
from ibnr.gallery.statistical.copula_glm.model import CopulaGLM
from ibnr.gallery.statistical.sur.model import SUR
from ibnr.kernels.holdout import CellIndex, next_diagonal
from ibnr.kernels.rng import cohort_stream, heldout_stream

from .conftest import make_cohort_triangle, make_multiline_triangle

START = 2010

#: Two cohorts drawing with one seed must land no further from zero correlation
#: than sampling noise. At 4,000 draws the standard deviation of a sample
#: correlation is about 1/sqrt(4000) = 0.016, so this is a six-sigma band.
MAX_CORR = 0.1


def corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two equal-length draw vectors."""
    return float(np.corrcoef(np.asarray(a, dtype=float), np.asarray(b, dtype=float))[0, 1])


def total_column(pred) -> np.ndarray:
    """The draws of a predictive distribution's ``total`` target.

    Read by label rather than by position: the total is what the defect
    actually damages, because it is a sum taken WITHIN each draw, so shared
    noise across cohorts survives into it while each cohort's own margin looks
    perfectly healthy.
    """
    labels = list(pred.targets["label"])
    return pred.samples[:, labels.index("total")]


# =============================================================================
# deterministic/mack - the entry notebook 3c measured
# =============================================================================

MACK_N_W = 6
MACK_AS_OF = "2015-12-31"


def full_square(n_w: int = MACK_N_W, seed: int = 7, level: float = 1000.0) -> np.ndarray:
    """A COMPLETE run-off square generated under Mack's own dynamics.

    The same builder ``tests/test_mack_heldout.py`` uses, with a ``level``
    multiplier added so two cohorts can differ in loss amount as well as in
    identity. ``as_of`` cuts the training staircase out of the square, which
    leaves the next diagonal as real observed data rather than a simulation.
    """
    rng = np.random.default_rng(seed)
    factors = np.array([1.5, 1.2, 1.1, 1.05, 1.02])
    cum = np.empty((n_w, n_w))
    cum[:, 0] = level * rng.uniform(0.9, 1.1, size=n_w)
    for j in range(n_w - 1):
        cum[:, j + 1] = factors[j] * cum[:, j] + np.sqrt(cum[:, j]) * rng.standard_normal(n_w) * 3.0
    return cum


@pytest.fixture(scope="module")
def mack_triangles():
    """Two single-cohort triangles that differ in BOTH segment and amounts.

    Differing in only one of the two would leave the test unable to say which
    part of the cohort identity the stream is keyed on.
    """
    a = make_cohort_triangle(None, full_square(seed=7), start_year=START, segment={"lob": "wkcomp"})
    b = make_cohort_triangle(
        None, full_square(seed=21, level=2500.0), start_year=START, segment={"lob": "othliab"}
    )
    return a, b


@pytest.fixture(scope="module")
def mack_entries(mack_triangles):
    tri_a, tri_b = mack_triangles
    return tuple(
        gallery.fit("mack", t, loss_field="paid_loss", as_of=MACK_AS_OF, heldout_n_draws=4000)
        for t in (tri_a, tri_b)
    )


def test_mack_predict_does_not_share_noise_across_fits(mack_triangles, mack_entries):
    """Two mack fits handed one seed must draw independent ultimates.

    Pre-fix both calls restarted the same generator, and the two total columns
    came back correlated at 0.53 - the gamma step law's rejection sampling is
    all that kept it below 1.0. Refitting the first triangle from scratch and
    reproducing its draws exactly is the other half: deriving a per-cohort
    stream must not cost reproducibility.
    """
    tri_a, _ = mack_triangles
    entry_a, entry_b = mack_entries
    draws_a = total_column(entry_a.predict(seed=7, n_draws=4000))
    draws_b = total_column(entry_b.predict(seed=7, n_draws=4000))
    assert abs(corr(draws_a, draws_b)) < MAX_CORR

    refit = gallery.fit("mack", tri_a, loss_field="paid_loss", as_of=MACK_AS_OF)
    again = total_column(refit.predict(seed=7, n_draws=4000))
    np.testing.assert_array_equal(again, draws_a)


def test_mack_predict_at_does_not_share_noise_across_fits(mack_triangles, mack_entries):
    """The shared held-out path, which every entry's CRPS column runs through.

    Both triangles carry the same origins and development lags, so cell 0 of
    one diagonal is the same position as cell 0 of the other and the two draw
    vectors are directly comparable. Pre-fix correlation at that cell: 0.53.
    """
    tri_a, tri_b = mack_triangles
    entry_a, entry_b = mack_entries
    cells_a = next_diagonal(tri_a, as_of=MACK_AS_OF, fields="paid_loss")
    cells_b = next_diagonal(tri_b, as_of=MACK_AS_OF, fields="paid_loss")
    assert cells_a.n_cells == cells_b.n_cells

    draws_a = entry_a.predict_at(cells_a, seed=7)
    draws_b = entry_b.predict_at(cells_b, seed=7)
    assert abs(corr(draws_a[:, 0], draws_b[:, 0])) < MAX_CORR
    np.testing.assert_array_equal(entry_a.predict_at(cells_a, seed=7), draws_a)


def test_mack_cdr_distribution_does_not_share_noise_across_fits(mack_entries):
    """The one-year claims development result, drawn per cohort.

    A capital figure read off several companies at once is a sum within each
    draw, so this is the number the defect damages most directly. Pre-fix
    correlation of the two total columns: 0.88.
    """
    entry_a, entry_b = mack_entries
    draws_a = total_column(entry_a.cdr_distribution(seed=9, n_draws=4000))
    draws_b = total_column(entry_b.cdr_distribution(seed=9, n_draws=4000))
    assert abs(corr(draws_a, draws_b)) < MAX_CORR
    np.testing.assert_array_equal(
        total_column(entry_a.cdr_distribution(seed=9, n_draws=4000)), draws_a
    )


# =============================================================================
# statistical/sur - two companies, several lines each
# =============================================================================

MULTI_N_W = 8
MULTI_CUTOFF = dt.date(START + MULTI_N_W - 1, 12, 31)


def chain_ladder_square(rng, *, n_w: int = MULTI_N_W, level: float = 1000.0) -> np.ndarray:
    """Full ``(n_lob, n_w, n_d)`` square from a chain-ladder process.

    The generator ``tests/test_sur.py`` simulates from, at zero cross-line
    correlation - the dependence structure is not what this file is about.
    """
    factors = np.array([1.5, 1.2, 1.1, 1.05])
    sigmas = np.array([2.0, 3.0])
    n_lob, n_d = len(sigmas), len(factors) + 1
    cum = np.empty((n_lob, n_w, n_d))
    cum[:, :, 0] = level * rng.uniform(0.8, 1.2, size=(n_lob, n_w))
    for d in range(n_d - 1):
        eta = rng.standard_normal((n_lob, n_w)) * sigmas[:, None]
        cum[:, :, d + 1] = factors[d] * cum[:, :, d] + np.sqrt(cum[:, :, d]) * eta
    return cum


def multiline_triangle(cum, *, company: str, premium: float | None = None):
    lobs = {f"lob_{k}": cum[k] for k in range(cum.shape[0])}
    prem = (
        None
        if premium is None
        else {f"lob_{k}": np.full(cum.shape[1], premium) for k in range(cum.shape[0])}
    )
    return make_multiline_triangle(
        "duckdb", lobs, premium_by_lob=prem, start_year=START, company=company
    )


@pytest.fixture(scope="module")
def sur_entries():
    a = multiline_triangle(chain_ladder_square(np.random.default_rng(1)), company="0001")
    b = multiline_triangle(
        chain_ladder_square(np.random.default_rng(2), level=3000.0), company="0002"
    )
    return SUR().fit(a, as_of=MULTI_CUTOFF), SUR().fit(b, as_of=MULTI_CUTOFF)


def test_sur_predict_does_not_share_noise_across_fits(sur_entries):
    """Two companies' SUR fits handed one seed draw independent grand totals.

    Both fits have the same number of lines, origins and development steps, so
    pre-fix they consumed the generator in lockstep - measured correlation of
    the two grand totals: 0.86.
    """
    entry_a, entry_b = sur_entries
    draws_a = total_column(entry_a.predict(n_draws=4000, seed=3))
    draws_b = total_column(entry_b.predict(n_draws=4000, seed=3))
    assert abs(corr(draws_a, draws_b)) < MAX_CORR
    np.testing.assert_array_equal(total_column(entry_a.predict(n_draws=4000, seed=3)), draws_a)


# =============================================================================
# statistical/copula_glm
# =============================================================================

DEV_LEVEL = np.log(np.array([0.45, 0.25, 0.15, 0.08, 0.05]))


def lognormal_square(
    rng, *, n_w: int = MULTI_N_W, rho: float = 0.6, premium: float = 1000.0
) -> np.ndarray:
    """Full square simulated from the copula model's own process.

    Same builder as ``tests/test_copula_glm.py``: correlated lognormal
    incremental loss ratios, cumulated last so the dependence sits where the
    model looks for it.
    """
    sigmas = np.array([0.10, 0.15])
    n_lob, n_d = len(sigmas), len(DEV_LEVEL)
    corr_matrix = np.full((n_lob, n_lob), rho)
    np.fill_diagonal(corr_matrix, 1.0)
    chol = np.linalg.cholesky(corr_matrix)
    alpha = rng.normal(0.0, 0.05, size=n_w)
    incr = np.empty((n_lob, n_w, n_d))
    for w in range(n_w):
        for d in range(n_d):
            z = chol @ rng.standard_normal(n_lob)
            incr[:, w, d] = premium * np.exp(DEV_LEVEL[d] + alpha[w] + sigmas * z)
    return np.cumsum(incr, axis=2)


@pytest.fixture(scope="module")
def copula_entries():
    a = multiline_triangle(
        lognormal_square(np.random.default_rng(11)), company="0001", premium=1000.0
    )
    b = multiline_triangle(
        lognormal_square(np.random.default_rng(12), premium=4000.0),
        company="0002",
        premium=4000.0,
    )
    return CopulaGLM().fit(a, as_of=MULTI_CUTOFF), CopulaGLM().fit(b, as_of=MULTI_CUTOFF)


def test_copula_glm_predict_does_not_share_noise_across_fits(copula_entries):
    """Two copula_glm fits handed one seed draw independent grand totals.

    ``param_uncertainty="plugin"`` keeps the run short; it also makes the
    pre-fix damage plainest, because every draw is then a fixed transform of
    the shared normal scores. Measured correlation before the fix: 0.98.
    """
    entry_a, entry_b = copula_entries
    kwargs = {"n_draws": 2000, "seed": 5, "param_uncertainty": "plugin"}
    draws_a = total_column(entry_a.predict(**kwargs))
    draws_b = total_column(entry_b.predict(**kwargs))
    assert abs(corr(draws_a, draws_b)) < MAX_CORR
    np.testing.assert_array_equal(total_column(entry_a.predict(**kwargs)), draws_a)


# =============================================================================
# statistical/clark
# =============================================================================

CLARK_N_W = 6
CLARK_CUTOFF = dt.date(START + CLARK_N_W - 1, 12, 31)


def emergence_square(rng, *, n_w: int = CLARK_N_W, level: float = 1000.0) -> np.ndarray:
    """A cumulative square with a decaying emergence pattern.

    Clark fits a growth curve to that pattern under an over-dispersed Poisson
    likelihood, so every increment has to stay positive - which rules out the
    Mack-style additive noise the other builders here use.
    """
    pattern = np.array([0.45, 0.25, 0.15, 0.08, 0.05, 0.02])
    ultimate = level * rng.uniform(0.85, 1.15, size=n_w)
    incr = ultimate[:, None] * pattern[None, :] * rng.uniform(0.9, 1.1, size=(n_w, len(pattern)))
    return np.cumsum(incr, axis=1)


@pytest.fixture(scope="module")
def clark_entries():
    a = make_cohort_triangle(
        None,
        emergence_square(np.random.default_rng(31)),
        start_year=START,
        segment={"lob": "wkcomp"},
    )
    b = make_cohort_triangle(
        None,
        emergence_square(np.random.default_rng(32), level=5000.0),
        start_year=START,
        segment={"lob": "othliab"},
    )
    fit_kwargs = {
        "loss_field": "paid_loss",
        "premium_field": None,
        "method": "ldf",
        "as_of": CLARK_CUTOFF,
    }
    return Clark().fit(a, **fit_kwargs), Clark().fit(b, **fit_kwargs)


def test_clark_predict_does_not_share_noise_across_fits(clark_entries):
    """Two Clark fits handed one seed draw independent ultimates.

    Clark's draws are a multivariate normal parameter sample followed by
    over-dispersed Poisson process noise; the Poisson step's rejection sampling
    is what held the pre-fix correlation down to 0.49 rather than near 1.0.
    """
    entry_a, entry_b = clark_entries
    draws_a = total_column(entry_a.predict(n_draws=1000, seed=2))
    draws_b = total_column(entry_b.predict(n_draws=1000, seed=2))
    assert abs(corr(draws_a, draws_b)) < MAX_CORR
    np.testing.assert_array_equal(total_column(entry_a.predict(n_draws=1000, seed=2)), draws_a)


# =============================================================================
# kernels.rng - the derivation itself
# =============================================================================

ONE_COHORT = [{"company": "A", "lob": "x"}]


def first_draws(stream, n: int = 4) -> np.ndarray:
    """What a generator built on ``stream`` produces, as the stream's fingerprint.

    Comparing draws rather than the SeedSequence keeps these tests about the
    thing that matters - whether two calls sample the same numbers - so they
    would still hold if the derivation changed how it packs its entropy.
    """
    return np.random.default_rng(stream).standard_normal(n)


def test_identical_arguments_give_one_stream():
    """The whole point: same cohort, same seed, same draws - every time."""
    args = {"label": "predict", "cohorts": ONE_COHORT, "field": "paid_loss"}
    np.testing.assert_array_equal(
        first_draws(cohort_stream(7, **args)), first_draws(cohort_stream(7, **args))
    )


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param({"label": "predict_at"}, id="label"),
        pytest.param({"field": "reported_loss"}, id="field"),
        pytest.param({"as_of": dt.date(2015, 12, 31)}, id="as_of"),
        pytest.param({"cohorts": [{"company": "B", "lob": "x"}]}, id="cohort_value"),
        pytest.param({"cohorts": [{"company": "A", "lob": "y"}]}, id="cohort_second_value"),
        pytest.param({"cohorts": [{"company": "A"}]}, id="cohort_columns"),
        pytest.param({"cohorts": [*ONE_COHORT, {"company": "B", "lob": "x"}]}, id="cohort_count"),
    ],
)
def test_each_key_axis_changes_the_stream(changed):
    """Every part of the cohort identity is actually read.

    One axis at a time, because a derivation that ignored one of them would
    still pass a test that varied several together.
    """
    base = {"label": "predict", "cohorts": ONE_COHORT, "field": "paid_loss", "as_of": None}
    assert not np.array_equal(
        first_draws(cohort_stream(7, **base)), first_draws(cohort_stream(7, **{**base, **changed}))
    )


def test_a_different_seed_changes_the_stream():
    """The caller's own seed still does what a seed does."""
    args = {"label": "predict", "cohorts": ONE_COHORT}
    assert not np.array_equal(
        first_draws(cohort_stream(7, **args)), first_draws(cohort_stream(8, **args))
    )


def test_cohort_order_and_key_order_do_not_change_the_stream():
    """Two ways of writing the same cohort identity are the same identity.

    A caller assembling cohorts from a dataframe has no control over row order
    or column order, so letting either move the stream would make a rerun of the
    same study produce different numbers for no reason a reader could see.
    """
    listed = [{"company": "A", "lob": "x"}, {"company": "B", "lob": "y"}]
    reversed_rows = list(reversed(listed))
    reversed_keys = [{"lob": c["lob"], "company": c["company"]} for c in listed]
    want = first_draws(cohort_stream(7, label="predict", cohorts=listed))
    np.testing.assert_array_equal(
        first_draws(cohort_stream(7, label="predict", cohorts=reversed_rows)), want
    )
    np.testing.assert_array_equal(
        first_draws(cohort_stream(7, label="predict", cohorts=reversed_keys)), want
    )


def test_a_bare_mapping_is_one_cohort():
    """A dict is iterable and iterating it yields its KEYS, so a single cohort
    passed on its own has to be recognized before anything is iterated."""
    np.testing.assert_array_equal(
        first_draws(cohort_stream(7, label="predict", cohorts=ONE_COHORT[0])),
        first_draws(cohort_stream(7, label="predict", cohorts=ONE_COHORT)),
    )


def test_empty_cohort_identities_are_accepted():
    """An unsegmented triangle's cohort identity is the empty mapping, and a
    caller may legitimately have no cohorts at all."""
    assert cohort_stream(7, label="predict", cohorts={}) is not None
    assert cohort_stream(7, label="predict", cohorts=[]) is not None
    assert not np.array_equal(
        first_draws(cohort_stream(7, label="predict", cohorts={})),
        first_draws(cohort_stream(7, label="predict", cohorts=[])),
    )


def test_none_stays_none():
    """``seed=None`` must reach ``default_rng`` untouched, or a caller asking
    for fresh operating-system entropy would silently get a fixed stream."""
    assert cohort_stream(None, label="predict", cohorts=ONE_COHORT) is None


def test_an_explicit_generator_passes_through_unchanged():
    """A caller who built their own stream gets that stream, not one derived
    from it - the object itself, by identity."""
    generator = np.random.default_rng(3)
    assert cohort_stream(generator, label="predict", cohorts=ONE_COHORT) is generator
    sequence = np.random.SeedSequence(12345)
    assert cohort_stream(sequence, label="predict", cohorts=ONE_COHORT) is sequence


def test_the_canonical_text_is_frozen():
    """GOLDEN PIN: this entropy list must not change.

    It freezes the canonical text and the way its sha256 digest becomes entropy,
    across platforms and across releases. Every published study is reproducible
    only as long as this holds: change it accidentally and every result computed
    from a given seed is silently redrawn, with nothing in any output to say so.
    Changing it on purpose is a breaking change and belongs in the changelog.
    """
    stream = cohort_stream(7, label="predict", cohorts=ONE_COHORT, field="paid_loss")
    assert list(stream.entropy) == [
        7,
        2889875121,
        4166576583,
        3426702354,
        1495552616,
        717613194,
        1874045580,
        2803470726,
        2846847470,
    ]


# -- heldout_stream ----------------------------------------------------------


def test_heldout_stream_resolves_the_only_field(mack_triangles):
    """Held-out cells of a single-field triangle need no ``field=``, and the
    resolved value is the one the cells carry - not some default."""
    tri_a, _ = mack_triangles
    cells = next_diagonal(tri_a, as_of=MACK_AS_OF, fields="paid_loss")
    np.testing.assert_array_equal(
        first_draws(heldout_stream(7, cells)),
        first_draws(heldout_stream(7, cells, field="paid_loss")),
    )


def test_heldout_stream_refuses_several_fields_without_one_named(mack_triangles):
    """Two fields and no choice made is ambiguous, and answering anyway would
    key the stream on whichever field happened to sort first."""
    tri_a, _ = mack_triangles
    cells = next_diagonal(tri_a, as_of=MACK_AS_OF, fields="paid_loss")
    both = replace(
        cells,
        frame=pd.concat(
            [cells.frame, cells.frame.assign(field="reported_loss")], ignore_index=True
        ),
    )
    with pytest.raises(ValueError, match="span fields"):
        heldout_stream(7, both)


def test_heldout_stream_honors_an_explicit_field(mack_triangles):
    """``field=`` is read, not decoration: naming a different field must move
    the stream, or ``predict_at``'s own ``field=`` would be inert here."""
    tri_a, _ = mack_triangles
    cells = next_diagonal(tri_a, as_of=MACK_AS_OF, fields="paid_loss")
    assert not np.array_equal(
        first_draws(heldout_stream(7, cells, field="paid_loss")),
        first_draws(heldout_stream(7, cells, field="reported_loss")),
    )


def test_heldout_stream_separates_two_cohorts(mack_triangles):
    """Different cells, different cohorts, one seed - different streams. This
    is the behavior test above reduced to the derivation alone."""
    tri_a, tri_b = mack_triangles
    cells_a = next_diagonal(tri_a, as_of=MACK_AS_OF, fields="paid_loss")
    cells_b = next_diagonal(tri_b, as_of=MACK_AS_OF, fields="paid_loss")
    assert not np.array_equal(
        first_draws(heldout_stream(7, cells_a)), first_draws(heldout_stream(7, cells_b))
    )


def test_the_heldout_construction_is_frozen(mack_triangles):
    """GOLDEN PIN for the held-out derivation, label included.

    The ``cohort_stream`` pin above freezes the text format; this one freezes
    everything ``heldout_stream`` feeds it - the ``predict_at`` label, the field
    resolved from the cells, the cutoff, the cells' segment identity. Without
    it, ``predict_at`` silently adopting another method's label would pass every
    other test here (both sides of each equality rebuild through
    ``heldout_stream``) while putting a fit's held-out draws and its run-off
    draws back on one stream. Independently recomputed from the documented
    construction: sha256 of ``predict_at``, ``paid_loss``, ``2015-12-31`` and
    ``lob=wkcomp`` joined with the part separator, read as little-endian words.
    """
    tri_a, _ = mack_triangles
    cells = next_diagonal(tri_a, as_of=MACK_AS_OF, fields="paid_loss")
    assert list(heldout_stream(7, cells).entropy) == [
        7,
        3640912721,
        1898873467,
        3582603879,
        2720242411,
        49762666,
        3290713026,
        2856782108,
        3746394144,
    ]


def test_heldout_stream_refuses_a_bare_cell_index():
    """A ``CellIndex`` carries no cohort, field or cutoff, so it cannot key a
    stream - and silently keying on nothing would put every cohort back on one."""
    idx = CellIndex(
        w=np.array([1]),
        d=np.array([2]),
        value=np.array([1.0]),
        prev_value=np.array([1.0]),
        premium=np.array([np.nan]),
    )
    with pytest.raises(TypeError, match="HoldoutCells"):
        heldout_stream(7, idx)
