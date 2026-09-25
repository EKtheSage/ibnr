"""Mack's distribution-free chain ladder, implemented natively over a Triangle.

Mack (1993), *Distribution-free calculation of the standard error of chain
ladder reserve estimates* (ASTIN Bulletin 23/2). The model makes no
distributional assumption; it fixes the first two conditional moments of the
development, which is exactly what the one-year claims development result of
``kernels/cdr.py`` needs on top:

    E[C_{i,j+1} | C_{i,0}, ..., C_{i,j}]   = f_j * C_{i,j}
    Var(C_{i,j+1} | C_{i,0}, ..., C_{i,j}) = sigma_j^2 * C_{i,j}

with accident years independent. Estimated on the observed run-off triangle by

    f_j-hat      = sum_i C_{i,j+1} / S_j,        S_j = sum_i C_{i,j}
    sigma_j-hat^2 = 1/(n_j - 1) * sum_i C_{i,j} * (C_{i,j+1}/C_{i,j} - f_j-hat)^2

The volume-weighted (alpha = 1) factor is Mack's estimator and the one
Merz-Wuthrich assume; no other averaging is offered here, because the CDR
formulas downstream are only valid for this one.

THE TWO SUMS DO NOT ALWAYS RUN OVER THE SAME ORIGINS. Both range over the
origins observing both ends of the step, but sigma's summand carries a 1/C_{i,j}
(the weighted residual is (C_{i,j+1} - f_j C_{i,j})^2 / C_{i,j}), so it is
estimable only where C_{i,j} > 0:

    f_j-hat        all pair origins; needs only S_j > 0
    sigma_j-hat^2  the pair origins with C_{i,j} > 0, count ``n_j``, df n_j - 1

They coincide on any triangle with strictly positive cumulatives, which is most
of them. They part on a real and unexceptional cohort - an accident year with
zero paid at 12 months - whose chain-ladder ultimate is perfectly well defined
and whose sigma simply has one fewer observation behind it. Refusing the whole
cohort would throw away a usable reserve estimate; quietly summing 0 * inf into
the variance would be worse. ``MackFit.n_obs`` and ``MackFit.n_pos`` record the
two counts separately so the divergence is visible rather than inferred.

That is the ``zero_cells="observed"`` reading, the default. Under
``zero_cells="missing"`` (chainladder-python's rule, a zero cell is a missing
one) every pair with a zero at either end leaves both sums, so the two sums run
over the same origins again, and an open origin whose latest amount is zero gets
an ultimate and a standard error of 0; see :func:`fit_mack_grid`.

The one thing this file cannot check on the factor path is the LATEST DIAGONAL:
those cells have no observed successor and so enter no step's estimator, yet
every variance formula divides by them. That guard therefore lives on the
variance path - see ``MackFit.require_positive_open_diagonals``.

Why this exists rather than a call into chainladder-python: chainladder is an
optional interop extra (``ibnr[interop]``), and the CDR is a *core* deliverable
that has to run in the duckdb-only install and inside the compute image. The
tie-out tests (``tests/test_mack.py``, marked ``tieout``) pin every quantity
here against ``cl.MackChainladder`` on raa, so "native" never means "different".

Cross-refs: ``kernels/contract.py::cohort_grid`` (the data contract),
``kernels/cdr.py`` (one-year CDR built on ``MackFit``),
``gallery/deterministic/mack`` (the gallery entry that wraps this in a
``PredictiveDistribution``).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

# Nothing at module level here imports pandas or the Triangle layer, because
# ibnr.methods imports this module and must not load ibis, pandas or scipy. The
# Triangle paths (fit_mack, fit_mack_many), the pandas summaries and the
# simulation import what they need when they run.
from ibnr.errors import Refusal, RefusedCell
from ibnr.kernels.grid import ZERO_CELLS, check_grid

if TYPE_CHECKING:
    import pandas as pd

    from ibnr.kernels.holdout import CellIndex
    from ibnr.kernels.predictive import PredictiveDistribution
    from ibnr.triangle.core import Triangle

#: how the variance of the LAST development step is estimated. That step has a
#: single observation, so it has no residual degrees of freedom of its own.
SIGMA_RULES = ("mack", "log_linear")

#: shapes for a simulated development step. Mack's model fixes only the first
#: two conditional moments, so a simulation must add one assumption; all three
#: match those moments and differ in tail shape and support.
PROCESS_LAWS = ("gamma", "normal", "lognormal")


@dataclass(frozen=True)
class MackFit:
    """A fitted distribution-free chain ladder on one cohort.

    Arrays are 0-based on both axes: dev index ``j`` spans ``0 .. n_d - 1`` and
    the development step ``j -> j + 1`` carries ``f[j]``, ``sigma2[j]``,
    ``s[j]`` (its volume denominator), ``n_obs[j]`` (the origins behind the
    FACTOR) and ``n_pos[j]`` (the origins behind the SIGMA), each of length
    ``n_d - 1``. The last two differ only where a pair origin has a zero
    cumulative, and never under ``zero_cells="missing"``; see the module
    docstring. ``zero_cells`` is the reading of a zero cumulative the fit was
    made with, and ``zero_links`` counts the link ratios it left out.

    ``cum`` keeps the observed triangle (NaN outside it); ``full`` is the same
    matrix with the lower triangle filled by the chain-ladder projection, so
    ``full[:, -1]`` is the ultimate and ``full[i, j]`` for ``j > latest_dev[i]``
    is the ``C-hat_{i,j}`` that Mack's and Merz-Wuthrich's variance formulas
    both evaluate at.
    """

    cum: np.ndarray  # (n_w, n_d) observed cumulative, NaN outside the triangle
    obs_mask: np.ndarray  # (n_w, n_d) bool
    latest_dev: np.ndarray  # (n_w,) 0-based dev index of each origin's diagonal cell
    f: np.ndarray  # (n_d - 1,) volume-weighted development factors
    sigma2: np.ndarray  # (n_d - 1,) Mack process variance parameters
    s: np.ndarray  # (n_d - 1,) S_j = sum of C_{i,j} over the origins used for f[j]
    n_obs: np.ndarray  # (n_d - 1,) origins behind f[j] (the full pair set)
    n_pos: np.ndarray  # (n_d - 1,) origins behind sigma2[j] (those with C_{i,j} > 0)
    origin_periods: list[dt.date]
    dev_grain_months: int
    sigma_rule: str
    units: str | None = None
    loss_field: str | None = None
    #: what a zero cumulative was taken to be; see :func:`fit_mack_grid`
    zero_cells: str = "observed"

    def __post_init__(self) -> None:
        # checked here too, so a fit built by hand or decoded from a tampered
        # payload cannot carry a setting every reader would take as "observed"
        _require_zero_cells(self.zero_cells)

    # -- point estimates -------------------------------------------------------

    @property
    def n_w(self) -> int:
        return self.cum.shape[0]

    @property
    def n_d(self) -> int:
        return self.cum.shape[1]

    @property
    def latest(self) -> np.ndarray:
        """(n_w,) each origin's cumulative loss on the latest diagonal."""
        return self.cum[np.arange(self.n_w), self.latest_dev]

    @property
    def full(self) -> np.ndarray:
        """(n_w, n_d) observed triangle completed by the chain-ladder projection."""
        out = self.cum.copy()
        for i in range(self.n_w):
            for j in range(int(self.latest_dev[i]), self.n_d - 1):
                out[i, j + 1] = out[i, j] * self.f[j]
        return out

    @property
    def ultimate(self) -> np.ndarray:
        """(n_w,) projected ultimate = the completed triangle's last column."""
        return self.full[:, -1]

    @property
    def reserve(self) -> np.ndarray:
        """(n_w,) IBNR = ultimate - latest. Zero for a fully developed origin."""
        return self.ultimate - self.latest

    @property
    def zero_links(self) -> int:
        """How many observed link ratios ``zero_cells="missing"`` left out.

        A link ratio is left out under that rule when either of its two cells is
        zero. Always 0 under ``zero_cells="observed"``, which leaves nothing out
        (a zero there is data, and only sigma skips a pair starting from it).
        """
        if self.zero_cells != "missing":
            return 0
        pair = self.obs_mask[:, :-1] & self.obs_mask[:, 1:]
        zero = (self.cum[:, :-1] == 0) | (self.cum[:, 1:] == 0)
        return int((pair & zero).sum())

    @property
    def _zero_latest(self) -> np.ndarray:
        """(n_w,) bool: open origins whose latest cumulative is exactly zero, under
        ``zero_cells="missing"`` only (always all False under ``"observed"``)."""
        if self.zero_cells != "missing":
            return np.zeros(self.n_w, dtype=bool)
        return (self.latest_dev < self.n_d - 1) & (self.latest == 0)

    # -- preconditions ---------------------------------------------------------

    def require_positive_open_diagonals(self) -> None:
        """Every OPEN origin's latest-diagonal cell must be strictly positive.

        The factor estimator cannot enforce this and never could. Its ``c0``
        cells are exactly the cells with an observed successor, and on a run-off
        staircase an open origin's diagonal cell has none - so the diagonal is
        the one cell class no factor-side guard ever sees. Every variance formula
        downstream then divides by it: ``msep_runoff``'s process term
        (``ratio_j / C-hat_{i,j}`` starting at ``j = latest_dev[i]``) and
        Merz-Wuthrich's ``Phi_i = ratio_k / C_{i,k} + ...``.

        numpy divides silently, so without this the failure is invisible: a zero
        diagonal returns NaN msep for that origin AND a NaN total, a negative one
        returns a finite NEGATIVE msep whose square root is then NaN, and
        ``simulate_ultimates`` returns an exactly degenerate zero column because
        ``var = sigma2 * state`` is non-positive and every draw comes back at its
        mean. Nothing raises; the numbers are just wrong.

        Deliberately NOT called on the point path. The chain-ladder ultimate is
        a product of factors off that cell and needs no positivity at all, and
        the gallery's skill benchmark (``scripts/compare_gallery.py``) wants the
        ultimate even for a cohort whose variance is undefined. ``fit_mack``
        therefore still succeeds; only ``msep_runoff`` / ``simulate_ultimates`` /
        the CDR refuse.

        A CLOSED origin (already at the last dev column) is exempt: it has no
        remaining step, so nothing divides by its diagonal.

        Under ``zero_cells="missing"`` a latest cell of exactly zero is accepted
        too. Every term of Mack's msep for that origin carries its latest amount
        as a factor, so the msep's limit as the latest amount goes to zero is 0,
        and :meth:`msep_runoff` returns that limit directly rather than dividing
        by the zero. A simulation from it is a point mass at 0, which is the same
        limit. A negative latest cell is still refused under either rule.
        """
        open_ = self.latest_dev < self.n_d - 1
        diag = self.cum[np.arange(self.n_w), self.latest_dev]
        bad = np.nonzero(open_ & ~(diag > 0) & ~self._zero_latest)[0]
        if bad.size:
            step = self.dev_grain_months
            negative = (diag[bad] < 0).any()
            raise Refusal(
                "negative_cumulative" if negative else "variance_not_estimable",
                "non-positive cumulative on the latest diagonal of open origin(s) {cells}. "
                "Mack's conditional variance is proportional to that cell, so every msep "
                "rolling forward from it is undefined. The point estimate does not depend "
                "on it and is still available as .ultimate / .reserve",
                option="cells" if negative else "zero_cells",
                cells=[
                    RefusedCell(
                        None,
                        self.origin_periods[i],
                        (int(self.latest_dev[i]) + 1) * step,
                        float(diag[i]),
                    )
                    for i in bad
                ],
            )

    # -- run-off (total) uncertainty -------------------------------------------

    def msep_runoff(self) -> dict[str, np.ndarray | float]:
        """Mack's conditional MSEP of the FULL run-off reserve.

        Mack (1993) formula (3), per accident year ``i``:

            msep_i = C-hat_{i,J}^2 * sum_{j=k_i}^{J-1} (sigma_j^2 / f_j^2)
                                     * (1 / C-hat_{i,j} + 1 / S_j)

        where ``k_i = latest_dev[i]`` is the dev index of ``i``'s diagonal cell
        and ``J = n_d - 1``. The ``1/C-hat`` term is process risk, the ``1/S_j``
        term estimation risk; both are returned separately because
        ``cl.MackChainladder`` exposes them separately and the tie-out checks
        each. For the aggregate, Mack's second formula adds the estimation-risk
        covariance between accident years, which share the same estimated
        factors:

            msep_total = sum_i msep_i
                       + 2 * sum_{i<k} C-hat_{i,J} C-hat_{k,J}
                             * sum_{j=k_i}^{J-1} (sigma_j^2 / f_j^2) / S_j

        (the inner sum runs over the OLDER year's dev range, which is the
        intersection of the two ranges). Process risk carries no cross term:
        accident years are independent under Mack's assumptions.

        Returns ``msep`` / ``process`` / ``parameter`` per origin (variances,
        not standard errors) plus the scalars ``msep_total``,
        ``process_total``, ``parameter_total``.

        Under ``zero_cells="missing"`` an open origin whose latest cumulative is
        zero gets msep 0, its process and parameter parts 0, and adds nothing to
        the aggregate cross term. That is the limit of the formulas above as the
        latest amount goes to zero, since ``C-hat_{i,J}`` and every
        ``C-hat_{i,j}`` are that amount times a product of factors: the process
        part is proportional to it and the other two to its square. It is set
        directly because the process part's ``1 / C-hat_{i,j}`` would otherwise
        divide by the zero.
        """
        self.require_positive_open_diagonals()  # 1/C-hat_{i,j} below starts there
        full = self.full
        ratio = np.divide(  # (n_d - 1,) sigma_j^2 / f_j^2, the recurring weight
            self.sigma2, self.f**2, out=np.zeros_like(self.sigma2), where=self.f != 0
        )
        process = np.zeros(self.n_w)
        parameter = np.zeros(self.n_w)
        at_zero = self._zero_latest
        for i in range(self.n_w):
            if at_zero[i]:
                continue  # the limit as the latest amount goes to 0: every term is 0
            for j in range(int(self.latest_dev[i]), self.n_d - 1):
                process[i] += ratio[j] / full[i, j]
                parameter[i] += ratio[j] / self.s[j]
        ult2 = self.ultimate**2
        process *= ult2
        parameter *= ult2
        msep = process + parameter

        # aggregate estimation risk: every pair of accident years shares the
        # factors estimated on the dev steps they both still have to run through
        cross = 0.0
        for i in range(self.n_w):
            tail_ult = self.ultimate[i + 1 :].sum()
            if tail_ult == 0.0:
                continue
            shared = sum(ratio[j] / self.s[j] for j in range(int(self.latest_dev[i]), self.n_d - 1))
            cross += 2.0 * self.ultimate[i] * tail_ult * shared
        return {
            "msep": msep,
            "process": process,
            "parameter": parameter,
            "msep_total": float(msep.sum() + cross),
            "process_total": float(process.sum()),
            "parameter_total": float(parameter.sum() + cross),
        }

    def summary(self) -> pd.DataFrame:
        """One row per origin: latest, ultimate, IBNR and the run-off standard
        error, plus a ``total`` row. Mirrors ``cl.MackChainladder.summary_``."""
        import pandas as pd

        risk = self.msep_runoff()
        out = pd.DataFrame(
            {
                "origin": self.origin_periods,
                "latest": self.latest,
                "ultimate": self.ultimate,
                "ibnr": self.reserve,
                "runoff_se": np.sqrt(risk["msep"]),
            }
        )
        total = {
            "origin": "total",
            "latest": self.latest.sum(),
            "ultimate": self.ultimate.sum(),
            "ibnr": self.reserve.sum(),
            "runoff_se": np.sqrt(risk["msep_total"]),
        }
        return pd.concat([out, pd.DataFrame([total])], ignore_index=True)

    # -- serialization ---------------------------------------------------------

    def to_arrow(self, *, compression: str | None = None) -> bytes:
        """Arrow IPC bytes. Carries ``n_obs`` and ``n_pos`` separately, because
        they are different counts and the positivity contract is the difference."""
        from ibnr.kernels import codec

        return codec.to_arrow(self, compression=compression)

    @classmethod
    def from_arrow(cls, data: bytes) -> MackFit:
        """Decode a fit written by :meth:`to_arrow`, refusing any other kind -
        including the ``MackFitPanel`` it may well have come out of."""
        from ibnr.kernels import codec

        return codec.from_arrow(data, expect="MackFit")


def fit_mack(
    triangle: Triangle,
    *,
    loss_field: str = "paid_loss",
    as_of: dt.date | str | None = None,
    sigma_rule: str = "mack",
    zero_cells: str = "observed",
) -> MackFit:
    """Fit the distribution-free chain ladder on a single-cohort Triangle.

    ``as_of`` slices the backtest diagonal first (the training window); the
    triangle must hold exactly one segment combination. ``sigma_rule`` selects
    how the last development step's variance is estimated - see
    ``_estimate_factors``. ``zero_cells`` is as in :func:`fit_mack_grid`.
    """
    from ibnr.kernels.contract import cohort_grid

    train = triangle.as_of(as_of) if as_of is not None else triangle
    return fit_mack_grid(
        cohort_grid(train, loss_field=loss_field), sigma_rule=sigma_rule, zero_cells=zero_cells
    )


@dataclass(frozen=True)
class MackFitPanel:
    """Batch of per-cohort :class:`MackFit`\\ s from :func:`fit_mack_many`.

    ``fits`` is keyed by the cohort's segment-value tuple, in ``by`` order
    (``()`` for a segment-less triangle). ``errors`` holds the message for each
    cohort that was refused, and ``reasons`` its ``ibnr.errors.Refusal``
    reason code (such as ``"not_run_off"``), both populated under
    ``on_error="skip"`` only.
    """

    fits: dict[tuple, MackFit]
    errors: dict[tuple, str]
    by: tuple[str, ...]
    reasons: dict[tuple, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.fits)

    def __getitem__(self, key) -> MackFit:
        return self.fits[key if isinstance(key, tuple) else (key,)]

    def summary(self) -> pd.DataFrame:
        """One row per cohort: latest, ultimate, IBNR (point quantities only -
        per-cohort variance is ``self[key].msep_runoff()``, kept off this path
        because it needs the positive-open-diagonal guard cohort by cohort)."""
        import pandas as pd

        rows = [
            {
                **dict(zip(self.by, key, strict=True)),
                "latest": fit.latest.sum(),
                "ultimate": fit.ultimate.sum(),
                "ibnr": fit.reserve.sum(),
            }
            for key, fit in self.fits.items()
        ]
        return pd.DataFrame(rows)

    # -- serialization ---------------------------------------------------------

    def to_arrow(self, *, compression: str | None = None) -> bytes:
        """Arrow IPC bytes: one nested envelope per fitted cohort, plus
        ``errors`` - which under ``on_error="skip"`` is a result, not a failure."""
        from ibnr.kernels import codec

        return codec.to_arrow(self, compression=compression)

    @classmethod
    def from_arrow(cls, data: bytes) -> MackFitPanel:
        """Decode a panel written by :meth:`to_arrow`, refusing any other kind -
        including the single ``MackFit`` a caller may have meant to send."""
        from ibnr.kernels import codec

        return codec.from_arrow(data, expect="MackFitPanel")


def fit_mack_many(
    triangle: Triangle,
    *,
    loss_field: str = "paid_loss",
    as_of: dt.date | str | None = None,
    sigma_rule: str = "mack",
    on_error: str = "raise",
    zero_cells: str = "observed",
) -> MackFitPanel:
    """Fit the distribution-free chain ladder on every cohort in one pass.

    Semantically identical to looping ``fit_mack`` over
    ``triangle.filter(...)`` per segment combination - same grids, same
    estimators, same guards - but the triangle is sliced and materialized
    ONCE, and cohorts are gridded from the shared frame. The naive loop pays
    one engine round-trip per cohort (~5-10 ms each on duckdb), which is the
    dominant cost when fitting hundreds of Schedule P cohorts; this is the
    batch-fit API that closes it.

    ``on_error="raise"`` (default) fails fast naming the offending cohort;
    ``"skip"`` records the refusal's message in ``MackFitPanel.errors``, and its
    reason code in ``MackFitPanel.reasons``, and keeps going - real
    multi-company panels (e.g. clrd) routinely contain cohorts that are not
    run-off staircases or have zero-volume steps. Only an
    ``ibnr.errors.Refusal`` is skipped: any other exception is a defect, not a
    cohort the data rules out, and is raised whatever ``on_error`` says.

    ``zero_cells`` is as in :func:`fit_mack_grid` and applies to every cohort.
    """
    from ibnr.kernels.contract import cohort_grid_frame
    from ibnr.triangle.core import GRAIN_MONTHS

    if on_error not in ("raise", "skip"):
        raise Refusal(
            "invalid_option",
            "on_error must be 'raise' or 'skip', got {given}",
            option="on_error",
            given=on_error,
        )
    # checked before any cohort, so that on_error="skip" cannot turn a bad
    # setting into one identical error recorded against every cohort
    _require_zero_cells(zero_cells)
    _require_sigma_rule(sigma_rule)
    if triangle.meta.measure != "cumulative":
        raise Refusal(
            "invalid_option",
            "fit_mack_many requires a cumulative triangle",
            option="triangle",
        )
    train = triangle.as_of(as_of) if as_of is not None else triangle
    df = train.select_fields(loss_field).execute()
    if df.empty:
        raise Refusal(
            "invalid_option",
            "no rows for loss field {given}",
            option="loss_field",
            given=loss_field,
        )
    by = tuple(triangle.segments)
    step = GRAIN_MONTHS[triangle.meta.dev_grain]

    groups = df.groupby(list(by), dropna=False, sort=True) if by else [((), df)]
    fits: dict[tuple, MackFit] = {}
    errors: dict[tuple, str] = {}
    reasons: dict[tuple, str] = {}
    for key, group in groups:
        key = key if isinstance(key, tuple) else (key,)
        cohort = f"cohort {dict(zip(by, key, strict=True))}: "
        try:
            grid = cohort_grid_frame(
                group,
                dev_grain_months=step,
                units=triangle.meta.units,
                loss_field=loss_field,
                # the identity half a bare frame cannot derive: this loop's own
                # group key IS the cohort, and the measure was checked above
                segment=dict(zip(by, key, strict=True)),
                measure=triangle.meta.measure,
            )
            fits[key] = fit_mack_grid(grid, sigma_rule=sigma_rule, zero_cells=zero_cells)
        except Refusal as refusal:
            if on_error == "raise":
                # the same refusal, its message led by the cohort it came from
                raise refusal._replace(template=cohort + refusal.template) from refusal
            errors[key] = str(refusal)
            reasons[key] = refusal.reason
        except ValueError as exc:
            # not a refusal, so a defect: raised under either setting, named by cohort
            raise ValueError(f"{cohort}{exc}") from exc
    return MackFitPanel(fits=fits, errors=errors, by=by, reasons=reasons)


def fit_mack_grid(
    grid: dict[str, Any], *, sigma_rule: str = "mack", zero_cells: str = "observed"
) -> MackFit:
    """Fit Mack's distribution-free chain ladder from a grid dict (the array entry point).

    ``grid`` is the dense one-cohort dict that ``kernels.cohort_grid_frame``
    builds from a plain pandas frame (``origin_period``, ``dev_lag``, ``value``);
    its page lists the keys read here. It holds cumulative losses in ``cum``
    (origins by development steps, NaN where unobserved), with ``obs_mask``,
    ``latest_dev``, ``origin_periods`` and ``dev_grain_months`` beside it.
    :func:`fit_mack` is the same fit starting from a Triangle.

    The grid is checked before it is used, exactly as
    ``kernels.fit_conventional_grid`` checks it: the keys and arrays must agree,
    the measure must be ``"cumulative"``, origin periods must be the first day
    of their period and one development step apart, the cells must form a
    run-off triangle, and every still-developing origin must be observed up to
    the same date.

    This always estimates Mack's variance parameters as well as the factors, so
    it refuses some triangles the point estimate alone would not need to refuse:
    for example a development step with several origin pairs of which fewer
    than two start from a positive amount, which leaves that step's variance
    with nothing to be estimated from. For a point estimate on its own, use
    ``kernels.fit_conventional_grid``.

    ``sigma_rule`` picks how the variance is filled in for a development step
    with too few pairs to estimate it directly (usually the last step):

    - ``"mack"`` (the default here): Mack's 1993 rule, the smallest of the last
      two estimated variances and the next value their ratio points to.
    - ``"log_linear"``: extend a straight line through the logarithms of the
      earlier standard deviations.

    chainladder-python's ``MackChainladder`` uses the log-linear rule by
    default, so pass ``sigma_rule="log_linear"`` to compare standard errors
    with it. The factors and ultimates do not depend on this choice.

    ``zero_cells`` says what a cumulative of exactly zero is:

    - ``"observed"`` (the default here): data. The factor at each step uses
      every origin observing both ends of it, zeros included, as R's
      ``MackChainLadder`` does; sigma uses the pairs that start from a positive
      amount, since its residual divides by that amount.
    - ``"missing"``: chainladder-python's rule, which stores a zero cell as
      missing. A link ratio is used only when neither of its two cells is zero,
      so the link into a zero and the link out of it both drop, from the factor,
      its volume ``s``, ``n_obs`` and sigma alike. An origin whose latest
      cumulative is zero keeps it as its latest amount: its ultimate is 0 and
      its standard error is 0, the limit of Mack's formula (chainladder-python
      leaves that origin's ultimate missing instead). ``MackFit.zero_links``
      counts the link ratios the rule left out. The one-year claims development
      result refuses a fit where this rule left anything out, because the
      Merz-Wuthrich formulas have not been checked under it.

    On a triangle with no zero cumulative the two give the identical fit.
    """
    _require_sigma_rule(sigma_rule)
    _require_zero_cells(zero_cells)
    origins, _ = check_grid(grid)
    cum, mask = grid["cum"], grid["obs_mask"]
    n_d = grid["n_d"]
    if n_d < 2:
        raise Refusal(
            "variance_not_estimable",
            "Mack's chain ladder needs at least two development ages; this triangle has one",
            option="cells",
        )
    f, sigma2, s, n_obs, n_pos = _estimate_factors(
        cum,
        mask,
        sigma_rule=sigma_rule,
        zero_cells=zero_cells,
        lag_months=grid["dev_grain_months"],
        origins=origins,
    )
    return MackFit(
        cum=cum,
        obs_mask=mask,
        latest_dev=grid["latest_dev"],
        f=f,
        sigma2=sigma2,
        s=s,
        n_obs=n_obs,
        n_pos=n_pos,
        origin_periods=origins,
        dev_grain_months=grid["dev_grain_months"],
        sigma_rule=sigma_rule,
        units=grid.get("units"),
        loss_field=grid.get("loss_field"),
        zero_cells=zero_cells,
    )


def _require_zero_cells(zero_cells: str) -> None:
    if zero_cells not in ZERO_CELLS:
        raise Refusal(
            "invalid_option",
            "zero_cells must be 'observed' or 'missing', got {given}. 'observed' keeps "
            "a zero cumulative as data; 'missing' leaves out every link ratio with a zero at "
            "either end, as chainladder-python does",
            option="zero_cells",
            given=zero_cells,
        )


def _require_sigma_rule(sigma_rule: str) -> None:
    if sigma_rule not in SIGMA_RULES:
        raise Refusal(
            "invalid_option",
            f"sigma_rule must be one of {SIGMA_RULES}, got {{given}}",
            option="sigma_rule",
            given=sigma_rule,
        )


def _estimate_factors(
    cum: np.ndarray,
    mask: np.ndarray,
    *,
    sigma_rule: str,
    zero_cells: str = "observed",
    lag_months: int | None = None,
    origins: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Volume-weighted factors and Mack's variance parameters, step by step.

    WHICH CELLS MUST BE POSITIVE, and why it is not "all of them". The factor
    ``f_j = sum(C_{i,j+1}) / S_j`` divides once, by the column total, so it needs
    only ``S_j > 0``; an origin sitting at zero contributes nothing to the
    denominator and its successor to the numerator, which is exactly the
    column-total estimator every textbook writes down. Mack's sigma divides per
    origin, so it is estimable only on the pair origins with ``C_{i,j} > 0``.
    Hence three separate, separately-named errors rather than one blanket check:

    - ``C_{i,j} < 0``  -> hard error. A negative weight makes sigma_j^2 itself
      negative, which makes the msep negative and its square root NaN, all
      silently. There is no reading of Mack's model under which this is data.
    - ``S_j <= 0``     -> hard error. 0/0 factor.
    - fewer than 2 positive origins at a step that HAS several pairs -> hard
      error. This is not the last step's missing-degrees-of-freedom case and
      must not be extrapolated into: ``_tail_sigma2`` would happily return 0.0
      at ``j = 0`` and declare the most volatile development step noiseless.

    The last step (and any step with a single observation) genuinely has no
    residual degrees of freedom. Two conventions are offered, because the two
    reference implementations disagree and the difference is visible in the CDR:

    ``mack``        Mack's own rule from section 3 of the paper,
                    sigma_J^2 = min(sigma_{J-1}^4 / sigma_{J-2}^2,
                                    min(sigma_{J-2}^2, sigma_{J-1}^2)) -
                    what ``cl.Development(sigma_interpolation='mack')`` does.
    ``log_linear``  regress log(sigma_j) on j over the estimable steps and
                    extrapolate - the default of chainladder-python and of R's
                    ``MackChainLadder(est.sigma = "log-linear")``, and therefore
                    the convention behind most published raa numbers. Falls back
                    to Mack's rule when the regression is not identified (fewer
                    than two positive sigmas) or would give a non-positive value.

    Under ``zero_cells="missing"`` every pair with a zero at either end is left
    out of the step before anything is estimated, so ``f``, ``s``, ``n_obs`` and
    sigma all run over the same kept pairs, and ``n_pos == n_obs``. The negative
    cell check still runs over every observed pair first, so leaving a pair out
    can never hide a negative cumulative.

    ``origins`` (the grid's origin periods) and ``lag_months`` let a refusal
    name its cells and development ages; every refusal here is an
    ``ibnr.errors.Refusal``.

    Returns (f, sigma2, s, n_obs, n_pos), each of length ``n_d - 1``.
    """
    n_d = cum.shape[1]
    f = np.zeros(n_d - 1)
    sigma2 = np.full(n_d - 1, np.nan)
    s = np.zeros(n_d - 1)
    n_obs = np.zeros(n_d - 1, dtype=int)
    n_pos = np.zeros(n_d - 1, dtype=int)
    months = 1 if lag_months is None else int(lag_months)

    def cells(rows, j):
        return [
            RefusedCell(
                None,
                None if origins is None else origins[i],
                (j + 1) * months,
                float(cum[i, j]),
            )
            for i in rows
        ]

    for j in range(n_d - 1):
        link = [((j + 1) * months, (j + 2) * months)]
        pair = mask[:, j] & mask[:, j + 1]
        c0 = cum[pair, j]
        if c0.size == 0:
            raise Refusal(
                "no_link_ratio",
                "no origin has a link ratio {links}; the triangle cannot support a "
                "chain-ladder factor there",
                links=link,
            )
        if (c0 < 0).any():
            raise Refusal(
                "negative_cumulative",
                "negative cumulative loss in {cells}; Mack's variance is proportional to the "
                "cumulative a link ratio starts from, so a negative one drives the variance "
                "itself negative and every standard error built on it with it",
                option="cells",
                cells=cells(np.nonzero(pair)[0][c0 < 0], j),
            )
        if zero_cells == "missing":
            pair = pair & (cum[:, j] != 0) & (cum[:, j + 1] != 0)
            if not pair.any():
                raise Refusal(
                    "no_link_ratio",
                    "every origin with a link ratio {links} has a zero cumulative at one end "
                    "or the other, and zero_cells='missing' leaves those link ratios out, so "
                    "no factor can be estimated there. zero_cells='observed' keeps the zeros "
                    "as data",
                    option="zero_cells",
                    links=link,
                )
        c0, c1 = cum[pair, j], cum[pair, j + 1]
        s[j] = c0.sum()
        if s[j] <= 0:
            raise Refusal(
                "no_link_ratio",
                "every link ratio {links} starts from zero, so the volume-weighted factor "
                "there is 0/0",
                option="zero_cells",
                links=link,
            )
        n_obs[j] = c0.size
        f[j] = c1.sum() / s[j]
        # sigma's weighted residual is (c1 - f*c0)^2 / c0, so only the strictly
        # positive origins can carry it - a subset of the factor's. df is that
        # subset's count minus one, NOT n_obs - 1: dividing a shorter sum by a
        # longer df would shrink sigma for exactly the cohorts this admits.
        pos = c0 > 0
        n_pos[j] = int(pos.sum())
        if n_obs[j] > 1 and n_pos[j] < 2:
            raise Refusal(
                "variance_not_estimable",
                f"only {n_pos[j]} of the {n_obs[j]} link ratios {{links}} starts from a "
                "positive amount, so Mack's sigma there cannot be estimated. The cells at "
                "zero are {cells}. This is not the last age's missing-degrees-of-freedom "
                "case and is not extrapolated into: at an early age that would silently "
                "declare the most volatile part of the development noiseless",
                option="zero_cells",
                links=link,
                cells=cells(np.nonzero(pair)[0][~pos], j),
            )
        if n_pos[j] > 1:
            p0, p1 = c0[pos], c1[pos]
            sigma2[j] = float((p0 * (p1 / p0 - f[j]) ** 2).sum() / (n_pos[j] - 1))
    missing = np.nonzero(np.isnan(sigma2))[0]
    if zero_cells == "missing" and (missing < n_d - 2).any():
        # only this rule can leave a step BEFORE the last with one link ratio
        _fill_sigma_gaps(sigma2, missing, rule=sigma_rule, lag_months=months)
    else:
        for j in missing:
            sigma2[j] = _tail_sigma2(sigma2, j, rule=sigma_rule)
    return f, sigma2, s, n_obs, n_pos


def _fill_sigma_gaps(
    sigma2: np.ndarray, missing: np.ndarray, *, rule: str, lag_months: int
) -> None:
    """Fill sigma at steps left with one link ratio when one of them is not the last.

    Only ``zero_cells="missing"`` gets here: leaving out the link ratios that
    touch a zero can leave a single link ratio at an early or middle step, where
    sigma has no degrees of freedom. ``_tail_sigma2`` is written for the LAST
    step and extrapolates from the steps before it only, so at the first step it
    would answer 0.0 and call the most volatile step noiseless. This follows
    chainladder-python instead, and fills only from sigmas the data estimated,
    never from a value filled in here:

    - ``log_linear``: one regression of log(sigma_j) on j over every step with
      a positive estimate, before and after the gap, fills every step without
      an estimate. (chainladder-python also stores an estimated sigma of
      exactly 0 as missing and refills it; here it stays 0.)
    - ``mack``: Mack's rule from the two steps just before, which both need an
      estimate of their own.

    Anything else is refused by name rather than filled with a guess, as an
    ``ibnr.errors.Refusal`` with the reason ``variance_not_estimable``.
    """

    def links(steps) -> list[tuple[int, int]]:
        return [((int(j) + 1) * lag_months, (int(j) + 2) * lag_months) for j in steps]

    known = sigma2.copy()
    if rule == "log_linear":
        # A sigma of exactly 0 (every link ratio at that step equal) is an
        # estimate and stays as it is, but it has no logarithm to regress on.
        estimated = np.flatnonzero(np.isfinite(known) & (known > 0))
        if estimated.size < 2:
            raise Refusal(
                "variance_not_estimable",
                "the link ratios {links} kept at most one ratio each once zero_cells='missing' "
                "left out those with a zero cell, so Mack's sigma there has to be filled in "
                "from the other ages, and the log-linear rule needs at least two other ages "
                "with a positive sigma to do it. zero_cells='observed' keeps the zeros as data",
                option="sigma_rule",
                options=("sigma_rule", "zero_cells"),
                links=links(missing),
            )
        slope, intercept = np.polyfit(estimated.astype(float), np.log(np.sqrt(known[estimated])), 1)
        for j in missing:
            sigma2[j] = float(np.exp(intercept + slope * j) ** 2)
        return
    for j in missing:
        if j < 2 or not (np.isfinite(known[j - 1]) and np.isfinite(known[j - 2])):
            raise Refusal(
                "variance_not_estimable",
                "the link ratios {links} kept at most one ratio once zero_cells='missing' left "
                "out those with a zero cell, and Mack's rule fills that sigma from the two "
                "ages just before it, which do not both have an estimate. "
                "sigma_rule='log_linear' fills it from every estimated age instead; "
                "zero_cells='observed' keeps the zeros as data",
                option="sigma_rule",
                options=("sigma_rule", "zero_cells"),
                links=links([j]),
            )
        last, prev = float(known[j - 1]), float(known[j - 2])
        ratio = last**2 / prev if prev > 0 else last
        sigma2[j] = float(min(ratio, last, prev))


def simulate_ultimates(
    fit: MackFit,
    *,
    n_draws: int = 10_000,
    seed: int | np.random.SeedSequence | None = None,
    process: str = "gamma",
    parameter_risk: bool = True,
) -> PredictiveDistribution:
    """Simulate FULL run-off ultimates from a fitted Mack model.

    Mack's model is distribution-free, so a predictive *distribution* needs one
    assumption beyond it: the shape of the step-to-step shock. ``process``
    picks it from ``PROCESS_LAWS``; every choice matches Mack's two conditional
    moments and they differ only in tail shape and support. This is the
    bootstrap wrapper CLAUDE.md decision 4 requires of a deterministic baseline
    before it may enter the gallery, and it is the run-off counterpart of the
    one-year re-reserving in ``kernels/cdr.py``.

    Parameter risk is drawn ONCE PER DRAW and shared across accident years -
    that shared factor draw is what makes the accident years correlated, and
    dropping it (``parameter_risk=False``) leaves pure, independent process
    risk. The ``total`` column is the row-sum of the same draws, so the
    diversification is in the samples rather than assumed.

    ``seed`` is anything ``np.random.default_rng`` accepts. The ``mack`` gallery
    entry hands a per-cohort ``SeedSequence`` through here; a plain integer keeps
    the byte-exact meaning it has always had.
    """
    # a non-positive diagonal would give var = sigma2 * state <= 0, which
    # draw_step returns at its mean - a silent point mass, not an error
    fit.require_positive_open_diagonals()
    rng = np.random.default_rng(seed)
    n_w, n_d = fit.n_w, fit.n_d
    f_true = _factor_draws(fit, n_draws=n_draws, rng=rng, parameter_risk=parameter_risk)

    ult = np.empty((n_draws, n_w))
    for i in range(n_w):
        state = np.full(n_draws, fit.cum[i, fit.latest_dev[i]])
        for j in range(int(fit.latest_dev[i]), n_d - 1):
            mean = f_true[:, j] * state
            var = fit.sigma2[j] * state
            state = draw_step(rng, mean, np.maximum(var, 0.0), law=process)
        ult[:, i] = state

    import pandas as pd

    from ibnr.kernels.predictive import PredictiveDistribution

    targets = pd.DataFrame(
        {
            "label": [str(o) for o in fit.origin_periods],
            "origin_period": fit.origin_periods,
        }
    )
    return PredictiveDistribution(samples=ult, targets=targets, units=fit.units).with_total("total")


def _factor_draws(
    fit: MackFit, *, n_draws: int, rng: np.random.Generator, parameter_risk: bool
) -> np.ndarray:
    """``(n_draws, n_d - 1)`` "true" development factors, one row per draw.

    With ``parameter_risk``, drawn from Mack's estimation-error distribution
    ``f_j ~ N(f_j-hat, sigma_j^2 / S_j)`` - ONCE per draw and shared across
    every cell/origin that draw touches, which is exactly what correlates them.
    A normal draw can cross zero on a thin, volatile step; a negative "true"
    factor would make the simulated step meaningless, so it is floored at
    1e-12. Rare enough to be a footnote, loud enough to document.

    All THREE simulation paths consume this one function - ``simulate_ultimates``
    (run-off), :func:`draw_next_cells` (held-out CRPS) and
    ``kernels.cdr.simulate_one_year_cdr`` (one-year re-reserving) - so "parameter
    risk" cannot come to mean three subtly different things.
    """
    f_true = np.tile(fit.f, (n_draws, 1))
    if parameter_risk:
        se = np.sqrt(np.where(fit.s > 0, fit.sigma2 / fit.s, 0.0))
        f_true = np.maximum(f_true + rng.standard_normal((n_draws, fit.n_d - 1)) * se, 1e-12)
    return f_true


def _next_step_draws(
    fit: MackFit,
    step0: np.ndarray,
    prev: np.ndarray,
    *,
    n_draws: int,
    rng: np.random.Generator,
    process: str,
    parameter_risk: bool,
) -> np.ndarray:
    """One development step ahead of ``prev``: ``(n_draws, n_cells)`` draws with
    Mack's conditional moments ``E = f[j] * prev``, ``Var = sigma2[j] * prev``.

    ``step0`` is the 0-based step index per cell (``f[step0]`` carries the
    step). The single core behind BOTH :func:`draw_next_cells` (the held-out
    CRPS draws) and ``kernels.cdr.simulate_one_year_cdr``'s simulated next
    diagonal, shared so the two cannot drift: they are the same two steps -
    draw the true factors, then the step shock.
    """
    f_true = _factor_draws(fit, n_draws=n_draws, rng=rng, parameter_risk=parameter_risk)
    mean = f_true[:, step0] * prev  # (n_draws, n_cells)
    var = np.broadcast_to(fit.sigma2[step0] * prev, mean.shape)
    return draw_step(rng, mean, var, law=process)


def draw_next_cells(
    fit: MackFit,
    cells: CellIndex,
    *,
    rng: np.random.Generator,
    n_draws: int = 10_000,
    process: str = "gamma",
    parameter_risk: bool = True,
) -> np.ndarray:
    """``(n_draws, n_cells)`` one-step-ahead draws of CUMULATIVE loss at cells.

    The per-cell counterpart of ``kernels.cdr.simulate_one_year_cdr``'s step 2
    (and it shares that code, see :func:`_next_step_draws`): each cell's draw
    has Mack's conditional moments off its own training predecessor,

        E   = f[d - 2] * prev_value
        Var = sigma2[d - 2] * prev_value

    ``d - 2`` because ``cells.d`` is the 1-BASED dev index of the drawn cell
    while ``f``/``sigma2`` are 0-based per step (``f[j]`` carries dev index
    ``j + 1 -> j + 2``); the step ENDING at cell ``d`` is ``d - 2``. An
    off-by-one here reads a neighbouring factor and produces entirely plausible
    draws, which is why both bounds are checked loudly below and pinned by test.

    ``parameter_risk`` draws the true factors once per draw, SHARED across the
    cells (see :func:`_factor_draws`) - two cells on the same development step
    are positively correlated through it, exactly as the accident years are in
    the CDR simulation. ``process`` picks the step shock among
    ``PROCESS_LAWS``; ``gamma`` (the default) needs a positive conditional mean
    and :meth:`MackFit.require_positive_open_diagonals` is the guard that makes
    it well-posed. A cell whose step has ``sigma2 == 0`` (e.g. the extrapolated
    last step of a 2-column triangle) comes back as a POINT MASS at its mean,
    silently - ``draw_step``'s documented degenerate case.

    One step only, matching what ``kernels.holdout.next_diagonal`` scores:
    ``prev_value`` is training data, so no rollout and no leakage.
    """
    if process not in PROCESS_LAWS:
        raise _refuse_process(process)
    if n_draws < 1:
        raise _refuse_n_draws(n_draws)
    # Var = sigma2 * prev is non-positive off a non-positive diagonal, and
    # draw_step then returns the mean exactly - an invisible point mass rather
    # than an error. Same guard, same reason as every other variance path.
    fit.require_positive_open_diagonals()
    d = np.asarray(cells.d, dtype=int)
    if (d < 2).any():
        raise ValueError(
            f"{int((d < 2).sum())} cell(s) sit at dev index 1, which no development step "
            "ends at - there is no factor f[d-2] to draw with. next_diagonal() excludes "
            "these as new_origin, so reaching one means the cells were built by hand"
        )
    if (d - 2 >= len(fit.f)).any():
        raise ValueError(
            f"dev index {int(d.max())} is beyond the fitted steps (n_d={fit.n_d}, so the "
            f"deepest drawable dev index is {fit.n_d}); next_diagonal() excludes these as "
            "dev_beyond_trained"
        )
    prev = np.asarray(cells.prev_value, dtype=float)
    if np.isnan(prev).any():
        raise ValueError(
            f"{int(np.isnan(prev).sum())} cell(s) have no training predecessor "
            "(prev_value is NaN), so the one-step moments are undefined; next_diagonal() "
            "excludes these as no_predecessor"
        )
    return _next_step_draws(
        fit, d - 2, prev, n_draws=n_draws, rng=rng, process=process, parameter_risk=parameter_risk
    )


def _refuse_process(process) -> Refusal:
    return Refusal(
        "invalid_option",
        f"process must be one of {PROCESS_LAWS}, got {{given}}",
        option="process",
        given=process,
    )


def _refuse_n_draws(n_draws) -> Refusal:
    return Refusal(
        "invalid_option",
        "n_draws must be positive, got {given}",
        option="n_draws",
        given=n_draws,
    )


def draw_step(
    rng: np.random.Generator, mean: np.ndarray, var: np.ndarray, *, law: str
) -> np.ndarray:
    """One development step's shock, matching ``mean`` and ``var`` elementwise.

    Shared by the run-off simulation above and the one-year re-reserving in
    ``kernels/cdr.py`` so both carry the identical noise assumption. Degenerate
    cells (zero variance - e.g. a step whose sigma was estimated as zero) come
    back at their mean exactly, under every law.

    ``normal`` is unfloored on purpose: Mack's model constrains two moments and
    nothing else, so flooring would bias the comparison against the (also
    linear, also unfloored) analytic CDR formula. ``gamma`` and ``lognormal``
    keep the simulated cumulative positive and need a positive mean.
    """
    if law not in PROCESS_LAWS:
        raise _refuse_process(law)
    out = np.array(mean, dtype=float, copy=True)
    live = var > 0
    if not live.any():
        return out
    m, v = np.asarray(mean)[live], np.asarray(var)[live]
    if law == "normal":
        out[live] = m + np.sqrt(v) * rng.standard_normal(m.shape)
        return out
    if (m <= 0).any():
        raise Refusal(
            "negative_fitted_mean",
            f"{law} process noise needs positive conditional means; "
            f"{int((m <= 0).sum())} are zero or below",
            option="process",
            given=law,
        )
    if law == "gamma":
        out[live] = rng.gamma(shape=m**2 / v, scale=v / m)
    else:  # lognormal, moment-matched
        s2 = np.log1p(v / m**2)
        out[live] = rng.lognormal(mean=np.log(m) - 0.5 * s2, sigma=np.sqrt(s2))
    return out


def _tail_sigma2(sigma2: np.ndarray, j: int, *, rule: str) -> float:
    """Variance for a development step with no residual degrees of freedom."""
    known = np.array([v for v in sigma2[:j] if np.isfinite(v)])
    if known.size == 0:
        # Nothing to extrapolate from (a 2-column triangle): the step is
        # deterministic as far as the data can tell.
        return 0.0
    if rule == "log_linear" and known.size >= 2 and (known > 0).all():
        idx = np.array([k for k in range(j) if np.isfinite(sigma2[k])], dtype=float)
        slope, intercept = np.polyfit(idx, np.log(np.sqrt(known)), 1)
        # Deliberately uncapped: chainladder's loglinear_interpolation and R's
        # est.sigma="log-linear" both take the raw extrapolation, and the
        # tie-out tests pin us to it. Rising sigmas therefore extrapolate UP -
        # which is why Mack's own (always shrinking) rule is the default here.
        extrapolated = float(np.exp(intercept + slope * j) ** 2)
        if np.isfinite(extrapolated) and extrapolated > 0:
            return extrapolated
    if known.size == 1:
        return float(known[-1])
    last, prev = float(known[-1]), float(known[-2])
    ratio = last**2 / prev if prev > 0 else last
    return float(min(ratio, last, prev))
