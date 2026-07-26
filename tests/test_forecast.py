"""The forecast object: reducing held-out arrays to a board without lying.

This layer has no external reference to check itself against. Upstream every
number is pinned by something - ``holdout.py`` checks cells against the fit's own
contract, ``densities.py`` integrates each measure to 1, and the CSR scorer is
checked against the fit's own ``log_lik``. Here the inputs are already correct
and the only job is to reduce them, so **every wrong answer this module can
produce is a finite float of the right sign and the right order of magnitude**.

That shapes the tests. A value-only assertion ("is it negative?", "does it
rank?") passes on almost every mutation, so each test below names the specific
broken version it must catch, and several assert that the *wrong* answer is
still plausible - which is the evidence that no weaker check would do.

Four groups:

1. **The reduction.** ``logmeanexp`` versus the mean of the logs, and ``-inf``
   versus NaN.
2. **The panel.** Alignment by key, the intersection, and the two independent
   capability axes.
3. **The N/A rule.** Missing is ``pd.NA`` and never ``0.0`` - which on this
   board is simultaneously the best ELPD and the best CRPS there is.
4. **The refusals.** Everything ``align_panel`` will not let a caller do.
"""

from __future__ import annotations

import datetime as dt
import io

import numpy as np
import pandas as pd
import pytest

from ibnr.kernels.forecast import (
    CAPABILITIES,
    SCORE_DIRECTION,
    Absence,
    CohortForecast,
    ForecastPanel,
    align_panel,
    leaderboard,
    logmeanexp,
    resolve_field,
)
from ibnr.kernels.holdout import HoldoutCells, next_diagonal
from ibnr.kernels.scores import crps
from ibnr.triangle.core import Triangle

TASK = "paid_next_diagonal_v1"
N_DRAWS = 400


# -- fixtures ----------------------------------------------------------------


def _rows(company, *, fields=("paid_loss",), n=5, through=6, start=2010, scale=1.0):
    out = []
    for w in range(1, n + 1):
        for d in range(1, n + 1):
            if w + d - 1 > through:
                continue
            for f in fields:
                out.append(
                    {
                        "company": company,
                        "origin_period": dt.date(start + w - 1, 1, 1),
                        "dev_lag": 12 * d,
                        "eval_date": dt.date(start + w + d - 2, 12, 31),
                        "field": f,
                        "value": float((1000 * w + 100 * d) * scale),
                    }
                )
    return out


def _triangle(company, **kw) -> Triangle:
    return Triangle.from_long(pd.DataFrame(_rows(company, **kw)), measure="cumulative")


def _cells(company="CO_A", *, as_of="2014-12-31", **kw) -> HoldoutCells:
    fields = kw.get("fields", ("paid_loss",))
    return next_diagonal(_triangle(company, **kw), as_of=as_of, fields=list(fields))


@pytest.fixture
def cells_a() -> HoldoutCells:
    return _cells("CO_A")


@pytest.fixture
def cells_b() -> HoldoutCells:
    return _cells("CO_B")


def _density(cells: HoldoutCells, *, rng, shift=0.0, n_draws=N_DRAWS) -> np.ndarray:
    """A plausible (n_draws, n_cells) log density on the amount scale."""
    return rng.normal(-np.log(cells.values) - 2.0 + shift, 0.3, size=(n_draws, cells.n_cells))


def _draws(cells: HoldoutCells, *, rng, sd=0.25, n_draws=N_DRAWS) -> np.ndarray:
    return np.exp(rng.normal(np.log(cells.values), sd, size=(n_draws, cells.n_cells)))


def _forecast(model, cells, *, rng, shift=0.0, sd=0.25, density=True, draws=True):
    kw = {}
    kw["log_density"] = _density(cells, rng=rng, shift=shift) if density else None
    kw["draws"] = _draws(cells, rng=rng, sd=sd) if draws else None
    if not density:
        kw["density_absence"] = Absence("no_predictive_density", "test")
    if not draws:
        kw["draws_absence"] = Absence("no_cell_sampler", "test")
    return CohortForecast(model=model, task=TASK, cells=cells, field="paid_loss", **kw)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260726)


# =============================================================================
# 1. the reduction
# =============================================================================


def test_logmeanexp_is_not_the_mean_of_the_logs(rng):
    """THE test of this module. ``ll.mean(axis=0)`` is one character away and
    estimates ``E[log p]`` rather than ``log E[p]``.

    Mutation: ``return a.mean(axis=axis)``. The assertions on the mutant's own
    output are the point - they show that the wrong answer is negative, finite,
    and correctly ordered, so nothing but this comparison catches it.
    """
    ll = rng.normal(-10.0, 0.5, size=(8000, 5))
    got, naive = logmeanexp(ll), ll.mean(axis=0)

    assert (got > naive).all(), "Jensen: log E[p] > E[log p] whenever p varies"
    # the gap is Var/2 to second order, so it is model-dependent and does NOT
    # cancel between two models on the same cells
    assert np.allclose(got - naive, ll.var(axis=0) / 2.0, rtol=0.05)

    # what the mutant would look like: entirely plausible
    assert (naive < 0).all() and np.isfinite(naive).all()


def test_the_gap_widens_with_the_spread_so_it_cannot_cancel(rng):
    """A constant offset would cancel in any comparison. This one does not: it
    grows with the posterior spread of the pointwise log density, which favours
    whichever model has the larger fitted variance.

    Mutation: the same ``a.mean(axis=axis)``; this test states WHY that matters.
    """
    tight = rng.normal(-10.0, 0.25, size=(8000, 4))
    loose = rng.normal(-10.0, 2.0, size=(8000, 4))
    gap_tight = float((logmeanexp(tight) - tight.mean(axis=0)).mean())
    gap_loose = float((logmeanexp(loose) - loose.mean(axis=0)).mean())
    assert gap_loose > 8 * gap_tight


@pytest.mark.parametrize("n_draws", [2, 1000, 10000])
def test_a_constant_log_density_reduces_to_that_constant(n_draws):
    """``logmeanexp`` of a constant is the constant, at any draw count.

    Mutation: ``logsumexp(a, axis) - np.log(a.shape[1])`` (the OTHER axis). It
    fails by a different amount at each draw count, which is what identifies it
    as a wrong axis rather than a wrong constant.
    """
    a = np.full((n_draws, 3), -7.25)
    assert np.allclose(logmeanexp(a), -7.25)


def test_all_minus_inf_is_minus_inf_and_never_nan():
    """A cell every draw gives zero density is a verdict: the model assigned
    zero probability to something that happened.

    Mutation: the textbook max-shift ``m + log(sum(exp(a - m)))``. Measured: it
    returns NaN with only a RuntimeWarning, and pandas then SKIPS that cell, so
    the model quietly scores on fewer cells than everyone else.
    """
    got = logmeanexp(np.full((16, 2), -np.inf))
    assert np.isneginf(got).all()
    assert not np.isnan(got).any()


def test_one_minus_inf_draw_is_absorbed_not_propagated():
    """Part of the posterior giving zero density is ordinary; the cell still has
    a finite predictive density."""
    a = np.array([[-np.inf, -1.0], [-2.0, -1.0], [-3.0, -1.0]])
    got = logmeanexp(a)
    assert np.isfinite(got).all()
    assert np.isclose(got[0], np.log((np.exp(-2.0) + np.exp(-3.0)) / 3.0))


def test_nan_is_refused_rather_than_skipped():
    """NaN is never a verdict here - it is a bug, usually a hand-rolled
    logmeanexp two layers away. Refusing beats letting an aggregation skip it."""
    with pytest.raises(ValueError, match="NaN"):
        logmeanexp(np.array([[1.0, np.nan], [1.0, 1.0]]))


def test_a_single_draw_is_refused():
    """At S=1 logmeanexp IS the mean of the logs - the exact estimand it exists
    to avoid - and the result is a plug-in density, not a predictive one."""
    with pytest.raises(ValueError, match="at least 2 draws"):
        logmeanexp(np.array([[-1.0, -2.0]]))


def test_extreme_log_densities_do_not_underflow():
    """CSR's ``sig[d]`` shrinks with development, so a badly missed deep-dev cell
    reaches several hundred negative nats. ``log(mean(exp(ll)))`` computed
    naively underflows the whole column to ``-inf``.

    Mutation: ``np.log(np.exp(a).mean(axis=axis))``.
    """
    a = np.full((100, 1), -800.0)
    a[0, 0] = -799.0
    got = logmeanexp(a)
    assert np.isfinite(got).all(), "must not underflow to -inf"
    assert -801.0 < float(got[0]) < -798.0


# =============================================================================
# 2. the panel
# =============================================================================


def test_pointwise_elpd_adds_no_second_measure_carry(cells_a, rng):
    """The forecast object reduces; it does not convert. The measure carry
    happens once, in ``ScoresHeldout.log_lik_at``.

    Mutation: any ``- np.log(value)`` inside ``pointwise_elpd``. A value-only
    check ("is it negative?") passes on the mutant.
    """
    ld = _density(cells_a, rng=rng)
    f = CohortForecast(
        model="m",
        task=TASK,
        cells=cells_a,
        field="paid_loss",
        log_density=ld,
        draws_absence=Absence("no_cell_sampler"),
    )
    assert np.allclose(f.pointwise_elpd(), logmeanexp(ld, axis=0))


def test_pointwise_crps_matches_the_kernel_against_the_realized_outcomes(cells_a, rng):
    """CRPS is computed by ``kernels.scores.crps`` against ``cells.values`` -
    implemented once, per CLAUDE.md decision 5.

    Mutation: score against ``cells.increments`` instead. That is finite, smooth,
    and wrong by the training-diagonal anchor.
    """
    dr = _draws(cells_a, rng=rng)
    f = CohortForecast(
        model="m",
        task=TASK,
        cells=cells_a,
        field="paid_loss",
        draws=dr,
        density_absence=Absence("no_predictive_density"),
    )
    assert np.allclose(f.pointwise_crps(), crps(dr, cells_a.values))
    # the mutant's answer is off by the anchor and still entirely plausible
    wrong = crps(dr, cells_a.increments)
    assert np.isfinite(wrong).all() and (wrong > 0).all()
    assert not np.allclose(wrong, f.pointwise_crps())


def test_cells_are_joined_by_key_never_by_position(cells_a, cells_b, rng):
    """Two models' arrays are equal-length unlabelled vectors. If the panel
    aligned by position, reversing one model's cohort order would leave every
    total unchanged while the per-cell join underneath was garbage.

    Mutation: build ``pointwise`` with ``reset_index`` order instead of the key.
    """
    a1 = _forecast("m1", cells_a, rng=rng)
    b1 = _forecast("m1", cells_b, rng=rng)
    a2 = _forecast("m2", cells_a, rng=rng, shift=-0.5)
    b2 = _forecast("m2", cells_b, rng=rng, shift=-0.5)

    forward = leaderboard(align_panel([a1, b1, a2, b2]))
    backward = leaderboard(align_panel([b2, a2, b1, a1]))
    pd.testing.assert_frame_equal(forward, backward)

    # and the join really is keyed: one row per (model, key), and each key names
    # exactly one cell with one outcome
    panel = align_panel([a1, b1, a2, b2])
    assert not panel.pointwise.duplicated(["model", "key"]).any()
    assert panel.cells["key"].is_unique
    assert (panel.cells.groupby("key")["value"].nunique() == 1).all()


def test_key_frame_rows_match_the_array_columns_on_a_two_field_cohort():
    """The one seam where columns and keys are matched by POSITION. A two-field
    HoldoutCells is the fixture that can actually break it: the narrowing removes
    rows, so a mismatched selection changes which cell column *i* describes.

    Mutation: ``_narrowed`` returning ``cells.frame`` unfiltered, or sorting
    differently. A width-only assertion passes whenever the counts coincide,
    which is why this compares VALUES.
    """
    cells = _cells("CO_A", fields=("paid_loss", "reported_loss"))
    assert set(cells.frame["field"]) == {"paid_loss", "reported_loss"}

    paid = cells.frame[cells.frame["field"] == "paid_loss"]
    # a per-cell offset so a mis-narrowed selection cannot coincide, plus real
    # draw-to-draw variation so the zero-variance guard is not what fires
    ld = np.arange(len(paid))[None, :] - 3.0 + np.linspace(0, 0.5, 8)[:, None]
    f = CohortForecast(
        model="m",
        task=TASK,
        cells=cells,
        field="paid_loss",
        log_density=ld,
        draws_absence=Absence("no_cell_sampler"),
    )
    assert f.n_cells == len(paid)
    assert np.array_equal(f.key_frame["value"].to_numpy(), paid["value"].to_numpy())
    assert (f.key_frame["field"] == "paid_loss").all()


def test_a_width_mismatch_is_refused(cells_a, rng):
    """The width check is what pins the arrays to the keys."""
    with pytest.raises(ValueError, match="columns but field"):
        CohortForecast(
            model="m",
            task=TASK,
            cells=cells_a,
            field="paid_loss",
            log_density=_density(cells_a, rng=rng)[:, :-1],
            draws_absence=Absence("no_cell_sampler"),
        )


def test_a_member_refusing_a_cohort_shrinks_the_panel_for_everyone(cells_a, cells_b, rng):
    """Ethan's call, 2026-07-26: the panel shrinks and the cost is itemised.

    Mutation: take the UNION instead of the intersection. The assertion on the
    mutant is the ranking the guard exists to prevent - the refusing model scores
    higher precisely because it skipped the cells it could not handle.
    """
    complete = [_forecast("complete", c, rng=rng) for c in (cells_a, cells_b)]
    brittle = [
        _forecast("brittle", cells_a, rng=rng),
        CohortForecast(
            model="brittle",
            task=TASK,
            cells=cells_b,
            field="paid_loss",
            density_absence=Absence("scoring_refused", "non-positive loss"),
            draws_absence=Absence("scoring_refused", "non-positive loss"),
        ),
    ]
    panel = align_panel([*complete, *brittle])
    board = leaderboard(panel)

    assert panel.n_cells_for("elpd") == cells_a.n_cells, "only the surviving cohort"
    assert (board["n_cells_elpd"] == cells_a.n_cells).all(), "one n for every model"
    assert set(panel.dropped["missing_from"].explode()) == {"brittle"}
    assert len(panel.dropped) == 2 * cells_b.n_cells  # one row per score per cell
    dropped_for = dict(zip(board["model"], board["n_cells_elpd_dropped"], strict=True))
    assert dropped_for["complete"] == cells_b.n_cells
    assert dropped_for["brittle"] == 0


def test_the_two_capabilities_have_separate_panels(cells_a, cells_b, rng):
    """The reason ODP can be on the board at all.

    ODP has draws and no usable density. On ONE shared panel its refusals would
    delete cells from another model's ELPD - a column ODP does not appear in.

    Mutation: one shared panel over the union of both memberships. Then the ELPD
    panel shrinks too, which this test's first assertion catches.
    """
    csr = [_forecast("meyers_csr", c, rng=rng) for c in (cells_a, cells_b)]
    odp = [
        _forecast("odp", cells_a, rng=rng, density=False),
        CohortForecast(
            model="odp",
            task=TASK,
            cells=cells_b,
            field="paid_loss",
            density_absence=Absence("no_predictive_density", "quasi-likelihood"),
            draws_absence=Absence("scoring_refused", "negative increment"),
        ),
    ]
    panel = align_panel([*csr, *odp])

    assert panel.elpd_members == ("meyers_csr",)
    assert panel.crps_members == ("meyers_csr", "odp")
    assert panel.n_cells_for("elpd") == cells_a.n_cells + cells_b.n_cells, (
        "ODP is not an ELPD member, so its refusal must cost the ELPD panel nothing"
    )
    assert panel.n_cells_for("crps") == cells_a.n_cells, "but the CRPS panel does shrink"
    assert panel.elpd_fingerprint != panel.crps_fingerprint

    board = leaderboard(panel)
    row = board[board["model"] == "odp"].iloc[0]
    assert pd.isna(row["elpd"]) and row["elpd_status"] == "na: no_predictive_density"
    assert not pd.isna(row["crps"]) and row["crps_status"] == "scored"


def test_adding_a_non_member_changes_nothing(cells_a, cells_b, rng):
    """A model that is not a member of a score's panel cannot move that column.

    Mutation: make membership ``all_models`` rather than the eligible ones.
    """
    csr = [_forecast("meyers_csr", c, rng=rng) for c in (cells_a, cells_b)]
    alone = leaderboard(align_panel(csr))

    mack = [
        CohortForecast.unavailable(
            model="mack",
            task=TASK,
            cells=c,
            density_reason="no_predictive_density",
            draws_reason="scorer_not_implemented",
        )
        for c in (cells_a, cells_b)
    ]
    together = leaderboard(align_panel([*csr, *mack]))
    mine = together[together["model"] == "meyers_csr"].reset_index(drop=True)

    for column in ("elpd", "crps", "n_cells_elpd", "n_cells_crps", "elpd_fingerprint"):
        assert alone[column].iloc[0] == mine[column].iloc[0], column


# =============================================================================
# 3. the N/A rule
# =============================================================================


def test_a_missing_score_is_na_and_never_zero(cells_a, cells_b, rng):
    """0.0 on this board is simultaneously the best ELPD and the best CRPS.

    Mutation: replace the Python reduction in ``leaderboard`` with
    ``pointwise.groupby("model")["elpd"].sum()``. Measured on pandas 2.3.3: an
    all-missing group sums to 0.0, a nullable ``Float64`` dtype does NOT fix it
    (only ``min_count=1`` does), and ``idxmax()`` then returns the model that was
    never scored at all.
    """
    csr = [_forecast("meyers_csr", c, rng=rng) for c in (cells_a, cells_b)]
    absent = [
        CohortForecast.unavailable(
            model="ccl",
            task=TASK,
            cells=c,
            density_reason="scorer_not_implemented",
            draws_reason="scorer_not_implemented",
        )
        for c in (cells_a, cells_b)
    ]
    board = leaderboard(align_panel([*csr, *absent]))

    ccl = board[board["model"] == "ccl"].iloc[0]
    assert pd.isna(ccl["elpd"]) and pd.isna(ccl["crps"])
    # `pd.NA == 0` is itself NA, not False - which is exactly why a 0.0 here
    # would slip through a casual truthiness check
    assert ccl["elpd"] is pd.NA and ccl["crps"] is pd.NA

    # the model with no scores must not win either column
    assert board.loc[board["elpd"].idxmax(), "model"] == "meyers_csr"
    assert board.loc[board["crps"].idxmin(), "model"] == "meyers_csr"
    assert board["elpd"].max() < 0
    assert board["crps"].min() > 0

    # and it is still ON the board, not dropped
    assert set(board["model"]) == {"ccl", "meyers_csr"}


def test_the_na_survives_a_csv_round_trip(cells_a, rng):
    """A published board is a CSV. A ``fillna(0)`` in a writer would be caught
    here and nowhere else."""
    board = leaderboard(
        align_panel(
            [
                _forecast("m", cells_a, rng=rng),
                CohortForecast.unavailable(
                    model="none",
                    task=TASK,
                    cells=cells_a,
                    density_reason="no_predictive_density",
                    draws_reason="no_cell_sampler",
                ),
            ]
        )
    )
    buf = io.StringIO()
    board.to_csv(buf, index=False)
    back = pd.read_csv(io.StringIO(buf.getvalue()))
    row = back[back["model"] == "none"].iloc[0]
    assert pd.isna(row["elpd"]) and pd.isna(row["crps"])


def test_minus_inf_propagates_and_is_explained_on_the_same_row(cells_a, rng):
    """Ethan's call, 2026-07-26: propagate. A model that gave the outcome zero
    density ranks last, and the board says how many cells did it.

    Mutation: ``np.nan_to_num`` on the pointwise vector, or clipping. Either
    makes the total finite and the model merely mediocre.
    """
    ld = _density(cells_a, rng=rng)
    ld[:, 0] = -np.inf
    f = CohortForecast(
        model="zero",
        task=TASK,
        cells=cells_a,
        field="paid_loss",
        log_density=ld,
        draws_absence=Absence("no_cell_sampler"),
    )
    board = leaderboard(align_panel([f]))
    row = board.iloc[0]
    assert np.isneginf(float(row["elpd"]))
    assert row["n_cells_zero_density"] == 1
    assert row["elpd_status"] == "scored"


def test_absence_reasons_come_from_a_closed_vocabulary():
    with pytest.raises(ValueError, match="reason must be one of"):
        Absence("because_i_said_so")


def test_a_reason_cannot_name_the_wrong_capability():
    """The board PRINTS these. ``no_predictive_density`` on the draws axis reads
    as 'cannot draw', which is false for every ODP and bootstrap entry."""
    with pytest.raises(ValueError, match="describes the 'density' capability"):
        Absence("no_predictive_density").check_axis("draws")
    with pytest.raises(ValueError, match="describes the 'draws' capability"):
        Absence("no_cell_sampler").check_axis("density")
    # the axis-neutral reason is fine on both
    Absence("scorer_not_implemented").check_axis("density")
    Absence("scorer_not_implemented").check_axis("draws")


def test_unavailable_has_no_default_reason(cells_a):
    """A default would put ``no_predictive_density`` next to ``meyers_ccl`` - a
    lognormal model - which is a false statement the board would print."""
    with pytest.raises(TypeError):
        CohortForecast.unavailable(model="m", task=TASK, cells=cells_a)  # type: ignore[call-arg]


def test_an_array_and_an_absence_cannot_both_be_set(cells_a, rng):
    with pytest.raises(ValueError, match="exactly one of log_density"):
        CohortForecast(
            model="m",
            task=TASK,
            cells=cells_a,
            field="paid_loss",
            log_density=_density(cells_a, rng=rng),
            density_absence=Absence("scoring_refused"),
            draws_absence=Absence("no_cell_sampler"),
        )


def test_neither_an_array_nor_an_absence_is_refused(cells_a):
    with pytest.raises(ValueError, match="exactly one of draws"):
        CohortForecast(
            model="m",
            task=TASK,
            cells=cells_a,
            field="paid_loss",
            density_absence=Absence("scoring_refused"),
        )


def test_a_model_level_absence_must_be_uniform(cells_a, cells_b, rng):
    """ "This entry has no density" is a claim about the ENTRY. Declaring it on
    one cohort while scoring another means one of the two is a lie."""
    mixed = [
        _forecast("m", cells_a, rng=rng),
        CohortForecast(
            model="m",
            task=TASK,
            cells=cells_b,
            field="paid_loss",
            density_absence=Absence("no_predictive_density"),
            draws_absence=Absence("no_cell_sampler"),
        ),
    ]
    with pytest.raises(ValueError, match="on some cohorts and offers density on others"):
        align_panel(mixed)


# =============================================================================
# 4. the refusals
# =============================================================================


def test_a_duplicated_forecast_is_refused(cells_a, rng):
    """Built the realistic way - a results list concatenated with itself, which
    is what a re-run appended to a CSV looks like.

    Mutation: drop the ``clash`` check; the model's cells then count twice in its
    own sum and once in everyone else's, which reads as a model difference.
    """
    f = _forecast("m", cells_a, rng=rng)
    other = _forecast("n", cells_a, rng=rng)
    with pytest.raises(ValueError, match="twice"):
        align_panel([f, other, f])


def test_a_disagreeing_observed_value_is_refused(cells_a, rng):
    """Two models reporting different outcomes at the same key. The reachable
    cause is Meyers' ``pmax(paid, 1)`` clamp, applied for the lognormal entries
    only - a clamped run and an unclamped one produce identical keys and
    different values at exactly the cells that matter. It is also what catches a
    units mismatch, which would shift every log density by a constant large
    enough to decide the ranking.
    """
    clamped = _cells("CO_A")
    scaled = _cells("CO_A", scale=1000.0)
    with pytest.raises(ValueError, match="disagreeing value"):
        align_panel([_forecast("m", clamped, rng=rng), _forecast("n", scaled, rng=rng)])


def test_disagreeing_train_origins_are_refused(cells_a, rng):
    """The check every design proposal missed, and it is one line.

    Two fits over different origin windows pass every other guard - same keys,
    same outcomes, same eval_date, same exclusion counts. CLAUDE.md already
    records this bug class inflating an outcome aggregate 2.4x.
    """
    wide = _cells("CO_A")
    narrow = next_diagonal(
        _triangle("CO_A"),
        as_of="2014-12-31",
        fields="paid_loss",
        origins=[dt.date(y, 1, 1) for y in range(2011, 2015)],
    )
    common = set(wide.frame["dev_lag"]) & set(narrow.frame["dev_lag"])
    assert common, "fixture must overlap, or the keys differ and another guard fires"
    assert wide.train_origins != narrow.train_origins

    with pytest.raises(ValueError, match="train_origins"):
        align_panel([_forecast("m", wide, rng=rng), _forecast("n", narrow, rng=rng)])


def test_a_mixed_task_is_refused(cells_a, rng):
    """The paid and reported boards stay separate."""
    a = _forecast("m", cells_a, rng=rng)
    b = CohortForecast(
        model="n",
        task="reported_next_diagonal_v1",
        cells=cells_a,
        field="paid_loss",
        log_density=_density(cells_a, rng=rng),
        draws_absence=Absence("no_cell_sampler"),
    )
    with pytest.raises(ValueError, match="disagree on task"):
        align_panel([a, b])


def test_a_mixed_cutoff_is_refused(cells_a, rng):
    """Two cutoffs is two panels - and they are not even the same shape."""
    early = _cells("CO_A", as_of="2013-12-31")
    assert early.as_of != cells_a.as_of
    with pytest.raises(ValueError, match="disagree on as_of"):
        align_panel([_forecast("m", cells_a, rng=rng), _forecast("n", early, rng=rng)])


def test_a_mixed_segment_schema_is_refused(cells_a, rng):
    """An unsegmented fit is an aggregate and cannot be compared to one cohort's
    cells - the same rule ``index_into`` applies one level down."""
    plain_rows = [{k: v for k, v in r.items() if k != "company"} for r in _rows("CO_A")]
    plain = next_diagonal(
        Triangle.from_long(pd.DataFrame(plain_rows), measure="cumulative"),
        as_of="2014-12-31",
        fields="paid_loss",
    )
    assert plain.segments == ()
    with pytest.raises(ValueError, match="disagree on segment schema"):
        align_panel([_forecast("m", cells_a, rng=rng), _forecast("n", plain, rng=rng)])


def test_zero_variance_draws_are_refused(cells_a):
    """A repeated point estimate is not a posterior. Both reductions then return
    a plug-in value that is systematically overconfident and looks normal."""
    flat = np.tile(np.full(cells_a.n_cells, -4.0), (50, 1))
    with pytest.raises(ValueError, match="zero variance"):
        CohortForecast(
            model="m",
            task=TASK,
            cells=cells_a,
            field="paid_loss",
            log_density=flat,
            draws_absence=Absence("no_cell_sampler"),
        )


def test_an_all_minus_inf_column_does_not_trip_the_zero_variance_check(cells_a, rng):
    """A ``-inf`` column has no variance to speak of but is a verdict, not a
    plug-in. Mutation: drop the ``live`` mask; this fixture then raises
    spuriously and a legitimate zero-density cell becomes unreportable."""
    ld = _density(cells_a, rng=rng)
    ld[:, 0] = -np.inf
    f = CohortForecast(
        model="m",
        task=TASK,
        cells=cells_a,
        field="paid_loss",
        log_density=ld,
        draws_absence=Absence("no_cell_sampler"),
    )
    assert np.isneginf(f.pointwise_elpd()[0])


def test_non_finite_draws_are_refused(cells_a, rng):
    """Unlike a log density, where ``-inf`` is the legitimate verdict 'this
    outcome had zero probability', an infinite LOSS is not a forecast."""
    dr = _draws(cells_a, rng=rng)
    dr[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite draw"):
        CohortForecast(
            model="m",
            task=TASK,
            cells=cells_a,
            field="paid_loss",
            draws=dr,
            density_absence=Absence("no_predictive_density"),
        )


def test_a_zero_cell_forecast_is_refused():
    """A cohort whose diagonal was entirely excluded upstream has nothing to
    score. A zero-width forecast would align invisibly and make the model look
    like it covered the cohort."""
    cells = _cells("CO_A")
    empty = HoldoutCells(
        frame=cells.frame.iloc[:0],
        as_of=cells.as_of,
        eval_date=cells.eval_date,
        excluded=cells.excluded,
        train_origins=cells.train_origins,
        segments=cells.segments,
        measure=cells.measure,
    )
    with pytest.raises(ValueError, match="no held-out cells|zero cells"):
        CohortForecast(
            model="m",
            task=TASK,
            cells=empty,
            field="paid_loss",
            log_density=np.zeros((5, 0)),
            draws_absence=Absence("no_cell_sampler"),
        )


def test_an_empty_intersection_raises_with_the_census(cells_a, cells_b, rng):
    """Returning an empty panel would let a board of zeros be published.

    Mutation: return the empty panel instead of raising.
    """
    a_only = _forecast("m", cells_a, rng=rng)
    b_only = _forecast("n", cells_b, rng=rng)
    # both models are members, but they share no cells at all
    with pytest.raises(ValueError, match="intersection over ELPD members is empty"):
        align_panel([a_only, b_only])


def test_align_panel_needs_at_least_one_forecast():
    with pytest.raises(ValueError, match="at least one"):
        align_panel([])


def test_resolve_field_refuses_an_ambiguous_cohort():
    cells = _cells("CO_A", fields=("paid_loss", "reported_loss"))
    with pytest.raises(ValueError, match="span fields"):
        resolve_field(cells)


def test_cells_must_come_from_next_diagonal(rng):
    """A forecast cannot describe cells nothing showed to be held out."""
    with pytest.raises(TypeError, match="must be a HoldoutCells"):
        CohortForecast(
            model="m",
            task=TASK,
            cells=pd.DataFrame({"value": [1.0]}),
            field="paid_loss",
            log_density=np.zeros((4, 1)),
            draws_absence=Absence("no_cell_sampler"),
        )


# =============================================================================
# the board's own contract
# =============================================================================


def test_the_board_has_no_sort_key_and_returns_model_order(cells_a, cells_b, rng):
    """Decided 2026-07-25: ELPD and CRPS are published side by side with no
    default sort - the reader picks. A ``sort_by=`` would make whichever value
    people typed first the house ranking by habit.

    Mutation: add ``sort_by`` with any default.
    """
    items = []
    for name in ("zeta", "alpha", "mu"):
        items += [_forecast(name, c, rng=rng) for c in (cells_a, cells_b)]
    board = leaderboard(align_panel(items))
    assert list(board["model"]) == ["alpha", "mu", "zeta"]

    import inspect

    assert "sort_by" not in inspect.signature(leaderboard).parameters


def test_every_score_column_declares_its_direction(cells_a, rng):
    """The two columns run in OPPOSITE directions. Without a machine-readable
    declaration a renderer composing an average-rank column produces nonsense."""
    board = leaderboard(align_panel([_forecast("m", cells_a, rng=rng)]))
    for column in SCORE_DIRECTION:
        assert column in board.columns
    for score in CAPABILITIES.values():
        assert score in SCORE_DIRECTION
    assert SCORE_DIRECTION["elpd"] != SCORE_DIRECTION["crps"]


def test_the_board_carries_the_member_set_it_depends_on(cells_a, cells_b, rng):
    """The ELPD column is a function of the member set, so the number travels
    with it - the same discipline ``mart_publish_id`` gets."""
    items = [_forecast(n, c, rng=rng) for n in ("a", "b") for c in (cells_a, cells_b)]
    board = leaderboard(align_panel(items))
    assert (board["elpd_members"] == "a,b").all()
    assert board["elpd_fingerprint"].nunique() == 1
    assert len(board["elpd_fingerprint"].iloc[0]) == 16


def test_the_panel_reports_upstream_exclusions(cells_a, rng):
    """A model scoring a shorter diagonal must not look better for it - the
    upstream census is carried onto the panel."""
    panel = align_panel([_forecast("m", cells_a, rng=rng)])
    assert len(panel.excluded) == 1
    assert set(cells_a.exclusion_counts()) <= set(panel.excluded.columns)
    assert int(panel.excluded["n_scorable"].iloc[0]) == cells_a.n_cells


def test_by_cohort_carries_each_score_on_its_own_panel(cells_a, cells_b, rng):
    """The input any future clustered SE or stacking objective reads. The two
    columns of one row can rest on different cell counts, so both are carried."""
    items = [_forecast(n, c, rng=rng) for n in ("a", "b") for c in (cells_a, cells_b)]
    panel = align_panel(items)
    assert len(panel.by_cohort) == 4
    assert set(panel.by_cohort.columns) == {
        "model",
        "cohort",
        "elpd",
        "n_cells_elpd",
        "crps",
        "n_cells_crps",
    }
    # the cohort sums reconstruct the board total
    board = leaderboard(panel)
    for model in ("a", "b"):
        total = panel.by_cohort.loc[panel.by_cohort["model"] == model, "elpd"].sum()
        assert np.isclose(float(total), float(board.loc[board["model"] == model, "elpd"].iloc[0]))


def test_the_panel_type_is_frozen(cells_a, rng):
    panel = align_panel([_forecast("m", cells_a, rng=rng)])
    assert isinstance(panel, ForecastPanel)
    with pytest.raises(Exception, match="frozen|immutable|cannot assign"):
        panel.task = "other"  # type: ignore[misc]
