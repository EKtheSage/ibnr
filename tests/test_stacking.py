"""Stacking over two panels: the weights, the floor, and the pseudo-model.

Like the forecast layer below it, this module has no external reference to
check itself against - every wrong answer it can produce is a finite weight
vector on the simplex and a plausible-looking stacked forecast. So each test
names the specific broken version it must catch:

1. **The weights.** Recovery of an obviously dominant member, the even split,
   and the ``-inf`` floor (bayesblend's own lpd reduction turns a raw ``-inf``
   into NaN - the floor is the only thing between a legitimate zero-density
   verdict and NaN weights).
2. **The stacked density.** The row-concatenation identity: ``logmeanexp`` of
   the offset-stacked arrays equals ``logsumexp(log w + lpd)`` per cell. A
   flipped offset sign still yields a valid-looking CohortForecast.
3. **The pooled draws.** Largest-remainder apportionment, exact and
   deterministic - bayesblend's ``_blend`` resamples stochastically per
   datapoint, which is the behavior this arm deliberately avoids.
4. **The refusals.** as_of ordering, task and member-set mismatches: a weight
   vector fitted over one panel cannot be applied to another silently.

``BayesStacking``/``HierarchicalBayesStacking`` compile Stan, so their tests
live behind ``-m slow`` like every other cmdstan test.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from scipy.special import logsumexp

from ibnr.kernels.forecast import Absence, CohortForecast, align_panel, leaderboard, logmeanexp
from ibnr.kernels.stacking import (
    LPD_FLOOR,
    StackingResult,
    _largest_remainder,
    apply_weights,
    stack,
)

from .test_forecast import TASK, _cells, _density, _draws

WEIGHTS_AS_OF = "2013-12-31"
EVAL_AS_OF = "2014-12-31"


# -- fixtures ----------------------------------------------------------------


def _forecast(
    model,
    cells,
    *,
    rng,
    task=TASK,
    shift=0.0,
    log_density=None,
    draws=None,
    density=True,
    with_draws=True,
    n_draws=400,
):
    """A CohortForecast builder that can override task and inject raw arrays."""
    kw = {}
    if density:
        kw["log_density"] = (
            log_density
            if log_density is not None
            else _density(cells, rng=rng, shift=shift, n_draws=n_draws)
        )
    else:
        kw["density_absence"] = Absence("scoring_refused", "test refusal")
    if with_draws:
        kw["draws"] = draws if draws is not None else _draws(cells, rng=rng, n_draws=n_draws)
    else:
        kw["draws_absence"] = Absence("no_cell_sampler", "test")
    return CohortForecast(model=model, task=task, cells=cells, field="paid_loss", **kw)


def _weights_panel(rng, *, shift_b=0.0, neg_inf_col_b=False, as_of=WEIGHTS_AS_OF, equal=False):
    """Two models on two cohorts at the earlier cutoff, aligned."""
    forecasts = []
    for company in ("CO_A", "CO_B"):
        cells = _cells(company, as_of=as_of)
        ld_a = _density(cells, rng=rng)
        ld_b = ld_a.copy() if equal else _density(cells, rng=rng, shift=shift_b)
        if neg_inf_col_b and company == "CO_A":
            ld_b[:, 0] = -np.inf
        forecasts.append(_forecast("model_a", cells, rng=rng, log_density=ld_a))
        forecasts.append(_forecast("model_b", cells, rng=rng, log_density=ld_b))
    return align_panel(forecasts)


def _evaluation(rng, *, shift_b=0.0, as_of=EVAL_AS_OF, models=("model_a", "model_b"), task=TASK):
    out = []
    for company in ("CO_A", "CO_B"):
        cells = _cells(company, as_of=as_of)
        for model in models:
            shift = shift_b if model == "model_b" else 0.0
            out.append(_forecast(model, cells, rng=rng, shift=shift, task=task))
    return out


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260726)


# =============================================================================
# 1. the weights
# =============================================================================


def test_a_dominant_model_takes_nearly_all_the_weight(rng):
    """Model A's pointwise lpd is uniformly ~3 nats higher, so the log-score
    objective is increasing in w_A everywhere and MLE must go to the corner.

    Mutation: pivot the lpd matrix positionally with the models swapped; the
    weight lands on model_b and this fails.
    """
    pytest.importorskip("bayesblend")
    result = stack(_weights_panel(rng, shift_b=-3.0), _evaluation(rng, shift_b=-3.0))
    assert result.weights["model_a"] > 0.9
    assert result.weights["model_b"] < 0.1
    assert result.method == "mle"
    assert result.model == "stacked_mle"


def test_indistinguishable_models_split_the_weight(rng):
    """Identical lpd columns make the objective flat, so MLE stays at its
    uniform start. A stacking layer that favours either copy is reading
    something other than the scores."""
    pytest.importorskip("bayesblend")
    result = stack(_weights_panel(rng, equal=True), _evaluation(rng))
    assert 0.4 < result.weights["model_a"] < 0.6
    assert 0.4 < result.weights["model_b"] < 0.6


def test_weights_sum_to_one_and_carry_their_provenance(rng):
    pytest.importorskip("bayesblend")
    panel = _weights_panel(rng, shift_b=-1.0)
    result = stack(panel, _evaluation(rng, shift_b=-1.0))
    assert np.isclose(sum(result.weights.values()), 1.0, atol=1e-9)
    assert result.n_cells_weight_fit == panel.n_cells_for("elpd")
    assert result.weights_as_of == panel.as_of
    assert result.weights_fingerprint == panel.elpd_fingerprint


def test_a_non_simplex_weight_vector_is_refused():
    """A rescaled vector shifts the stacked ELPD by log(sum w) per cell and
    still looks entirely plausible - refuse it at the type."""
    with pytest.raises(ValueError, match="sum to"):
        StackingResult(
            method="mle",
            weights={"a": 0.5, "b": 0.6},
            n_cells_weight_fit=8,
            n_floored_neg_inf=0,
            weights_as_of=__import__("datetime").date(2013, 12, 31),
            weights_fingerprint="abc",
            forecasts=(),
        )


def test_a_neg_inf_pointwise_elpd_is_floored_not_nan(rng):
    """bayesblend's compute_lpd is the textbook max shift: a raw ``-inf``
    becomes NaN (measured: Draws.from_lpd([-10, -inf, -9]).lpd -> nan), and the
    NaN then rides the SLSQP solve into the weights.

    Mutation: skip the floor in ``_lpd_matrix``. The weights come back NaN and
    the simplex check in ``_check_weights`` raises - this test asserts the
    working path instead: finite weights, and the count on the record.
    """
    pytest.importorskip("bayesblend")
    result = stack(_weights_panel(rng, neg_inf_col_b=True), _evaluation(rng))
    assert result.n_floored_neg_inf == 1
    assert all(np.isfinite(w) for w in result.weights.values())
    assert np.isclose(sum(result.weights.values()), 1.0)
    # the floored member gave a weight-panel outcome zero density; the floor is
    # low enough that MLE reads that cell as (effectively) zero support
    assert result.weights["model_b"] < 0.5
    assert LPD_FLOOR == -700.0 and np.exp(LPD_FLOOR) > 0.0, (
        "the floor must stay a normal float: exp(-700) ~ 1e-304, not exactly 0"
    )


# =============================================================================
# 2. the stacked density
# =============================================================================


def test_the_concatenation_identity(rng):
    """logmeanexp of the offset-stacked arrays IS the mixture's pointwise log
    density: logsumexp(log w_m + lpd_m) per cell, to 1e-12.

    Members carry DIFFERENT draw counts so the S/S_m part of the offset does
    real work. Mutation: flip the offset sign to ``- log(w*S/S_m)``; the result
    is still a valid CohortForecast with plausible negative values, and only
    this comparison catches it.
    """
    cells = _cells("CO_A", as_of=EVAL_AS_OF)
    ld_a = _density(cells, rng=rng, n_draws=300)
    ld_b = _density(cells, rng=rng, shift=-1.0, n_draws=500)
    a = _forecast("model_a", cells, rng=rng, log_density=ld_a, n_draws=300)
    b = _forecast("model_b", cells, rng=rng, log_density=ld_b, n_draws=500)
    w = {"model_a": 0.3, "model_b": 0.7}

    (stacked,) = apply_weights(w, [a, b], model="stacked_mle")

    lhs = logmeanexp(stacked.log_density, axis=0)
    lpd = np.vstack([logmeanexp(ld_a, axis=0), logmeanexp(ld_b, axis=0)])
    rhs = logsumexp(np.log([w["model_a"], w["model_b"]])[:, None] + lpd, axis=0)
    assert np.allclose(lhs, rhs, rtol=0.0, atol=1e-12)
    assert stacked.log_density.shape == (800, cells.n_cells)


def test_a_zero_weight_member_is_dropped_from_both_arms(rng):
    """log(0) rows would be ``-inf`` everywhere: nothing in the mixture, poison
    in the zero-variance check. And zero draw rows is what a zero weight means.

    Mutation: keep the zero-weight member's rows with the log(0) offset; the
    shape assertion fails first, the allclose second.
    """
    cells = _cells("CO_A", as_of=EVAL_AS_OF)
    a = _forecast("model_a", cells, rng=rng)
    b = _forecast("model_b", cells, rng=rng, shift=-5.0)

    (stacked,) = apply_weights({"model_a": 1.0, "model_b": 0.0}, [a, b], model="stacked_mle")

    # density arm: only A's rows, and with S == S_a the offset is exactly 0
    assert stacked.log_density.shape == a.log_density.shape
    assert np.allclose(stacked.log_density, a.log_density)
    # draws arm: every pooled row is A's, in A's own deterministic selection
    assert np.array_equal(stacked.draws, a.draws)
    assert np.isfinite(stacked.log_density).all() and not np.isnan(stacked.log_density).any()


def test_a_member_refusing_a_cohort_makes_the_stacked_cohort_an_absence(rng):
    """The mixture is defined over all its members or not at all - a stacked
    density quietly built from the surviving member would be that member wearing
    the stack's name."""
    cells_a = _cells("CO_A", as_of=EVAL_AS_OF)
    cells_b = _cells("CO_B", as_of=EVAL_AS_OF)
    forecasts = [
        _forecast("model_a", cells_a, rng=rng),
        _forecast("model_b", cells_a, rng=rng),
        _forecast("model_a", cells_b, rng=rng),
        _forecast("model_b", cells_b, rng=rng, density=False),
    ]
    stacked = apply_weights({"model_a": 0.6, "model_b": 0.4}, forecasts, model="stacked_mle")
    by_cohort = {f.cohort: f for f in stacked}
    assert by_cohort[("CO_A",)].has_density
    refused = by_cohort[("CO_B",)]
    assert not refused.has_density
    assert refused.density_absence.reason == "scoring_refused"
    assert "model_b" in refused.density_absence.detail
    assert refused.has_draws, "the draws arm is independent and every member drew here"


# =============================================================================
# 3. the pooled draws
# =============================================================================


def test_draw_pooling_matches_largest_remainder_exactly(rng):
    """Weights enter the CRPS arm only through row counts, so the counts ARE
    the contract: quotas 127.8/72.2 over target 200 must give 128/72.

    Mutation: apportion with ``round()`` per member (can give 128/72 or lose a
    seat entirely depending on the weights) or plain ``floor`` (199 rows); the
    exact-count assertions fail.
    """
    cells = _cells("CO_A", as_of=EVAL_AS_OF)
    w = {"model_a": 0.639, "model_b": 0.361}
    a = _forecast("model_a", cells, rng=rng, n_draws=100)
    big = _draws(cells, rng=rng, n_draws=100) * 1e9  # rows identifiable by magnitude
    b = _forecast("model_b", cells, rng=rng, draws=big, n_draws=100)

    (stacked,) = apply_weights(w, [a, b], model="stacked_mle")

    assert stacked.draws.shape[0] == 200, "target = min member draw count x member count"
    n_b = int((stacked.draws[:, 0] > 1e6).sum())
    assert (200 - n_b, n_b) == (128, 72)
    assert _largest_remainder(w, 200) == {"model_a": 128, "model_b": 72}
    # and the helper always hands out exactly the target
    assert sum(_largest_remainder({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, 100).values()) == 100


def test_draw_pooling_is_deterministic_across_calls(rng):
    """bayesblend's ``_blend`` resamples with rng.choice per datapoint; ours
    must give byte-identical draws twice. Mutation: pool via rng.choice."""
    cells = _cells("CO_A", as_of=EVAL_AS_OF)
    a = _forecast("model_a", cells, rng=rng, n_draws=90)
    b = _forecast("model_b", cells, rng=rng, n_draws=140)
    w = {"model_a": 0.55, "model_b": 0.45}
    first = apply_weights(w, [a, b], model="s")[0].draws
    second = apply_weights(w, [a, b], model="s")[0].draws
    assert np.array_equal(first, second)
    assert first.shape[0] == 90 * 2


# =============================================================================
# 4. the refusals
# =============================================================================


def test_weights_must_be_fitted_strictly_before_the_evaluation(rng):
    """Weights graded on the cells that chose them measure selection, not
    skill. Both the reversed and the equal cutoff are refused - no bayesblend
    needed, the guard fires before any fitting."""
    late_panel = _weights_panel(rng, as_of=EVAL_AS_OF)
    with pytest.raises(ValueError, match="must precede"):
        stack(late_panel, _evaluation(rng, as_of=WEIGHTS_AS_OF))
    with pytest.raises(ValueError, match="must precede"):
        stack(late_panel, _evaluation(rng, as_of=EVAL_AS_OF))


def test_a_task_mismatch_is_refused(rng):
    with pytest.raises(ValueError, match="task mismatch"):
        stack(_weights_panel(rng), _evaluation(rng, task="reported_next_diagonal_v1"))


def test_a_member_set_mismatch_is_refused(rng):
    """A weight vector fitted over one member set cannot be applied to another:
    a missing member strands its weight, an extra one has none."""
    panel = _weights_panel(rng)
    with pytest.raises(ValueError, match="member-set mismatch"):
        stack(panel, _evaluation(rng, models=("model_a",)))
    with pytest.raises(ValueError, match="member-set mismatch"):
        stack(panel, _evaluation(rng, models=("model_a", "model_b", "model_c")))


def test_a_mixed_evaluation_cutoff_is_refused(rng):
    """One stack reads one panel's worth of forecasts; a mixed-cutoff list is
    two panels interleaved, and align_panel would refuse it too."""
    both = _evaluation(rng, as_of=EVAL_AS_OF) + _evaluation(rng, as_of="2012-12-31")
    with pytest.raises(ValueError, match="disagree on as_of"):
        stack(_weights_panel(rng), both)


def test_an_evaluation_panel_is_refused_with_directions(rng):
    """The stacked pseudo-model needs the members' raw arrays, which a panel
    does not retain - the error must say what to pass instead."""
    weights_panel = _weights_panel(rng)
    eval_panel = align_panel(_evaluation(rng))
    with pytest.raises(TypeError, match="CohortForecast objects themselves"):
        stack(weights_panel, eval_panel)


def test_stacking_needs_at_least_two_members(rng):
    forecasts = [
        _forecast("only", _cells(c, as_of=WEIGHTS_AS_OF), rng=rng) for c in ("CO_A", "CO_B")
    ]
    with pytest.raises(ValueError, match="at least two"):
        stack(align_panel(forecasts), _evaluation(rng, models=("only",)))


# =============================================================================
# end to end, and the import contract
# =============================================================================


def test_end_to_end_the_stacked_model_lands_on_the_board(rng):
    """Two cutoffs -> stack -> align_panel WITH the stacked forecasts ->
    leaderboard. The stacked model is an ordinary row: same guards, same
    columns, and its ELPD cannot sit below every member (the mixture with a
    near-degenerate weight is approximately its best member; a broken offset
    sign or a positional pivot lands it far outside the members' range).
    """
    pytest.importorskip("bayesblend")
    result = stack(_weights_panel(rng, shift_b=-2.0), _evaluation(rng, shift_b=-2.0))
    evaluation = _evaluation(rng, shift_b=-2.0)
    board = leaderboard(align_panel([*evaluation, *result.forecasts]))

    assert set(board["model"]) == {"model_a", "model_b", "stacked_mle"}
    by_model = board.set_index("model")
    stacked_elpd = float(by_model.loc["stacked_mle", "elpd"])
    member_elpds = [float(by_model.loc[m, "elpd"]) for m in ("model_a", "model_b")]
    assert stacked_elpd >= min(member_elpds)
    assert stacked_elpd <= max(member_elpds) + 1.0, (
        "the stack of these members cannot beat its best member by more than noise"
    )
    assert by_model.loc["stacked_mle", "elpd_status"] == "scored"
    assert by_model.loc["stacked_mle", "crps_status"] == "scored"
    assert (board["n_cells_elpd"] == board["n_cells_elpd"].iloc[0]).all(), (
        "the stacked model covers exactly the members' cells, so the panel must not shrink"
    )


def test_stacking_imports_without_bayesblend():
    """bayesblend pulls cmdstanpy + arviz at module scope, so a top-level import
    in kernels/stacking.py would break the core install. A clean interpreter is
    the only honest check (the [bayesian] extra is installed here).

    Mutation: move ``import bayesblend`` to module scope; the subprocess
    assertion fails.
    """
    code = (
        "import sys; "
        "import ibnr.kernels.stacking; "
        "import ibnr.gallery as g; "
        "assert 'bayesblend' not in sys.modules, 'stacking imported bayesblend eagerly'; "
        "assert callable(g.stack) and callable(g.leaderboard)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# =============================================================================
# the Stan-compiling methods, behind -m slow like every other cmdstan test
# =============================================================================


@pytest.mark.slow
def test_bayes_stacking_compiles_and_recovers_the_dominant_model(rng):
    pytest.importorskip("bayesblend")
    pytest.importorskip("cmdstanpy")
    from ibnr.gallery.bayesian._toolchain import ensure_stan_toolchain

    ensure_stan_toolchain()
    result = stack(
        _weights_panel(rng, shift_b=-3.0), _evaluation(rng, shift_b=-3.0), method="bayes", seed=1
    )
    assert result.model == "stacked_bayes"
    assert np.isclose(sum(result.weights.values()), 1.0, atol=1e-6)
    assert result.weights["model_a"] > 0.7, (
        "the Bayesian posterior mean is shrunk toward uniform by the Dirichlet prior, "
        "so the corner is softer than MLE's - but the dominant member must still dominate"
    )


@pytest.mark.slow
def test_hierarchical_stacking_fits_with_the_dev_lag_covariate(rng):
    """Per-cell weights inside the fit, one cell-averaged weight applied - the
    documented simplification. This asserts the applied vector is still a
    simplex and the stacked forecasts still build."""
    pytest.importorskip("bayesblend")
    pytest.importorskip("cmdstanpy")
    from ibnr.gallery.bayesian._toolchain import ensure_stan_toolchain

    ensure_stan_toolchain()
    result = stack(
        _weights_panel(rng, shift_b=-3.0),
        _evaluation(rng, shift_b=-3.0),
        method="hierarchical",
        seed=1,
    )
    assert result.model == "stacked_hierarchical"
    assert np.isclose(sum(result.weights.values()), 1.0, atol=1e-6)
    assert len(result.forecasts) == 2
    assert all(f.has_density and f.has_draws for f in result.forecasts)
