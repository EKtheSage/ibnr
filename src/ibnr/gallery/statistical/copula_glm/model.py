"""Copula-linked lognormal regressions across lines of business, after
Shi & Frees (2011), *Dependent loss reserving using copulas* (ASTIN 41(2)).

Role in the gallery: the second frequentist stochastic-dependence baseline
(alongside ``sur``). Where SUR couples lines through an additive Gaussian error
correlation on cumulatives, this entry models each line's *incremental* loss
ratio with a lognormal regression and couples the lines with a Gaussian copula,
cell by cell. Shi & Frees' preferred lognormal marginal, per line ``k``:

    log( incr_{k,w,d} / premium_{k,w} ) = x'_{w,d} beta_k + sigma_k eps_{k,w,d}

with ``x`` = intercept + origin dummies (w >= 2) + dev dummies (d >= 2), or the
2-parameter Hoerl curve ``[1, origin dummies, ln(d), d]`` for thin triangles.
Within each (origin, dev) cell ``(eps_1, ..., eps_K)`` is a Gaussian copula with
correlation ``R``; cells are independent (Shi & Frees' baseline structure), so
on the standardized-residual scale the copula is just a correlation of normal
scores.

Why ``loss_field="paid_loss"`` is the default (and matters): the lognormal
marginal is defined only on positive increments, and *incremental reported*
losses routinely go negative at late lags as case reserves are released. Paid
increments stay non-negative. ``nonpositive="error"`` (default) refuses to fit
on negative increments rather than silently biasing the marginals.

Cross-refs: card.md (model card, estimation, bootstrap cost, limitations);
``kernels.multiline`` (shared data contract + target layout); Shi & Frees (2011).
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from ibnr.gallery.entry import GalleryEntry
from ibnr.gallery.registry import register
from ibnr.kernels.multiline import (
    assemble_predictive,
    flatten_with_totals,
    multiline_data,
    multiline_targets,
    realized_multiline,
)
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.triangle.core import Triangle

EIG_FLOOR = 1e-10


@register
class CopulaGLM(GalleryEntry):
    """Shi & Frees (2011) dependent loss reserving: lognormal marginals per line,
    Gaussian copula across lines within a cell, parametric-bootstrap parameter
    risk. Fitted state below is everything ``predict()``/``_bootstrap()`` reuse.
    """

    name = "copula_glm"
    family = "statistical"

    def __init__(self) -> None:
        self.contract_: dict | None = None
        self.beta_: np.ndarray | None = None  # (n_lob, p)
        self.sigma_: np.ndarray | None = None  # (n_lob,)
        self.corr_: np.ndarray | None = None  # (n_lob, n_lob)
        self._loss_field: str | None = None
        self._dev_effect: str = "factor"
        self._obs_w: np.ndarray | None = None  # 0-based origin index per obs cell
        self._obs_d: np.ndarray | None = None  # 0-based dev index per obs cell
        self._x: np.ndarray | None = None  # (n_obs, p) shared design
        self._pinv: np.ndarray | None = None  # (p, n_obs) precomputed lstsq
        self._latest_cum: np.ndarray | None = None  # (n_lob, n_w)
        self._latest_dev: np.ndarray | None = None  # (n_w,), 1-based

    def fit(
        self,
        triangle: Triangle,
        *,
        loss_field: str = "paid_loss",
        premium_field: str = "earned_premium",
        as_of: dt.date | str | None = None,
        dev_effect: str = "factor",  # "factor" | "hoerl"
        nonpositive: str = "error",  # "error" | "drop"
    ) -> CopulaGLM:
        """Fit per-line lognormal regressions on incremental loss ratios plus
        a Gaussian copula across lines, on the as_of-sliced upper triangle.

        ``dev_effect``: "factor" (a free dummy per dev step, Shi & Frees' own
        19-parameter budget on a 10x10 square) or "hoerl" (a 2-parameter curve
        for thin triangles). ``nonpositive``: "error" refuses to fit on a
        non-positive increment; "drop" censors those cells (documented as biased).
        """
        if dev_effect not in ("factor", "hoerl"):
            raise ValueError(f"dev_effect must be 'factor' or 'hoerl', got {dev_effect!r}")
        if nonpositive not in ("error", "drop"):
            raise ValueError(f"nonpositive must be 'error' or 'drop', got {nonpositive!r}")
        train = triangle.as_of(as_of) if as_of is not None else triangle
        # Canonical prep also pulls premium (the marginal's exposure denominator).
        self.contract_ = multiline_data(train, loss_field=loss_field, premium_field=premium_field)
        self._loss_field = loss_field
        self._dev_effect = dev_effect
        c = self.contract_
        n_lob, n_w, n_d = c["n_lob"], c["n_w"], c["n_d"]
        cum, mask = c["cum"], c["obs_mask"]  # cum: (K, n_w, n_d)

        # Model is on INCREMENTAL losses; de-cumulate along the dev axis. First
        # dev step is its own increment; later steps are successive differences.
        incr = np.full_like(cum, np.nan)  # (K, n_w, n_d)
        incr[:, :, 0] = cum[:, :, 0]
        incr[:, :, 1:] = cum[:, :, 1:] - cum[:, :, :-1]
        # A cell is usable only where observed AND its increment is defined (the
        # predecessor cumulative was also observed - gaps yield NaN differences).
        usable = mask & ~np.isnan(incr)  # (K, n_w, n_d)
        # Copula is estimated cell-by-cell across lines, so a cell counts only if
        # usable in EVERY line; collapse over the lob axis and rebroadcast.
        usable = np.broadcast_to(usable.all(axis=0), usable.shape).copy()

        # Lognormal marginal is undefined on non-positive increments (the crux of
        # the paid-vs-reported choice - see the module docstring).
        nonpos = usable & ~(incr > 0)
        if nonpos.any():
            cells = int(nonpos.any(axis=0).sum())  # count of offending (w, d) cells
            if nonpositive == "error":
                raise ValueError(
                    f"{cells} cells have a non-positive increment in some line; "
                    "lognormal marginals need positive increments - use paid losses, "
                    'or nonpositive="drop" (biased: it censors the left tail)'
                )
            usable &= ~nonpos.any(axis=0)  # drop the cell in every line, keep alignment

        # Masks are identical across lines now, so line 0's mask lists the cells.
        cell = usable[0]  # (n_w, n_d) shared usable-cell pattern
        obs_w, obs_d = np.nonzero(cell)  # each (n_obs,) 0-based origin/dev index
        n_obs = len(obs_w)
        # Response is the log incremental loss RATIO (increment / earned premium),
        # so exposure scales out and beta lives on the loss-ratio scale.
        ratios = incr[:, obs_w, obs_d] / c["premium"][:, obs_w]  # (K, n_obs)
        y = np.log(ratios)  # (K, n_obs)

        # Shared design across lines: (n_obs, p). Marginals differ only in beta_k.
        x = self._design(obs_w, obs_d, n_w, n_d, dev_effect)
        p = x.shape[1]
        if n_obs <= p:
            raise ValueError(
                f"{n_obs} usable cells for {p} marginal parameters; "
                'not enough data - try dev_effect="hoerl" or a coarser model'
            )
        self._check_identified(obs_w, obs_d, n_w, n_d, dev_effect)

        # Marginals: OLS on logs is exact ML for a lognormal regression - no GLM
        # IRLS needed. One pseudo-inverse solves all K lines at once.
        pinv = np.linalg.pinv(x)  # (p, n_obs)
        beta = (pinv @ y.T).T  # (K, p) per-line coefficients
        resid = y - beta @ x.T  # (K, n_obs) log-scale residuals
        sigma = np.sqrt((resid**2).sum(axis=1) / (n_obs - p))  # (K,) marginal sd, df=n-p
        # Copula = Pearson correlation of standardized log-residuals (the normal
        # scores); PD-repair so predict()'s Cholesky exists. Trivial for K=1.
        std_resid = resid / sigma[:, None]  # (K, n_obs)
        self.corr_ = _nearest_pd(np.corrcoef(std_resid)) if n_lob > 1 else np.ones((1, 1))

        # Anchor for prediction: each origin's latest observed cumulative and the
        # 1-based dev step it sits at (future increments start at latest_dev[w]).
        latest_dev = np.zeros(n_w, dtype=int)  # (n_w,) 1-based latest observed step
        latest_cum = np.zeros((n_lob, n_w))  # (K, n_w) cumulative on that diagonal
        for w in range(n_w):
            devs = np.nonzero(mask[0, w])[0]
            if devs.size == 0:
                raise ValueError(f"origin {c['origin_periods'][w]} has no observations")
            latest_dev[w] = int(devs[-1]) + 1
            latest_cum[:, w] = cum[:, w, devs[-1]]

        self.beta_, self.sigma_ = beta, sigma
        # Cache design + pseudo-inverse so the bootstrap refits are one matmul each.
        self._obs_w, self._obs_d, self._x, self._pinv = obs_w, obs_d, x, pinv
        self._latest_cum, self._latest_dev = latest_cum, latest_dev
        return self

    def predict(
        self,
        *,
        n_draws: int = 10_000,
        seed: int | None = None,
        param_uncertainty: str = "bootstrap",  # "bootstrap" | "plugin"
        n_boot: int = 200,
    ) -> PredictiveDistribution:
        """Simulate ultimates: correlated lognormal future increments cell by
        cell (dependence across lines within a cell, independence across
        cells, as in Shi & Frees), anchored on the latest observed diagonal.

        ``bootstrap`` refits the marginals+copula on ``n_boot`` triangles
        simulated from the fitted model and spreads the draws over the
        replicates - parameter risk included. ``plugin`` uses point estimates.
        """
        if self.beta_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        if param_uncertainty not in ("bootstrap", "plugin"):
            raise ValueError(
                f"param_uncertainty must be 'bootstrap' or 'plugin', got {param_uncertainty!r}"
            )
        c = self.contract_
        n_lob, n_w, n_d = c["n_lob"], c["n_w"], c["n_d"]
        rng = np.random.default_rng(seed)

        # Assemble per-draw parameters. Bootstrap: draw n_boot refits, then assign
        # each of the n_draws predictive draws a random replicate (parameter risk
        # enters as scatter across replicates). Plugin: reuse the point estimates.
        if param_uncertainty == "bootstrap":
            beta_r, sigma_r, chol_r = self._bootstrap(rng, n_boot)  # (n_boot, ...)
            rep = rng.integers(0, n_boot, size=n_draws)  # (n_draws,) replicate index
            beta_d, sigma_d, chol_d = beta_r[rep], sigma_r[rep], chol_r[rep]
        else:
            beta_d = np.broadcast_to(self.beta_, (n_draws, *self.beta_.shape))  # (n_draws, K, p)
            sigma_d = np.broadcast_to(self.sigma_, (n_draws, n_lob))  # (n_draws, K)
            chol = np.linalg.cholesky(self.corr_)  # (K, K) copula Cholesky
            chol_d = np.broadcast_to(chol, (n_draws, n_lob, n_lob))

        # Start every draw at the latest observed diagonal; future increments are
        # added on. (n_draws, K, n_w).
        ults = np.tile(self._latest_cum, (n_draws, 1, 1))
        # Enumerate the lower-triangle cells to project: dev steps beyond each
        # origin's latest observed step. Cells are simulated independently (Shi &
        # Frees' cell-independence), dependence acts only across lines within a cell.
        future_w, future_d = [], []
        for w in range(n_w):
            for d in range(self._latest_dev[w], n_d):
                future_w.append(w)
                future_d.append(d)
        if future_w:
            xf = self._design(np.array(future_w), np.array(future_d), n_w, n_d, self._dev_effect)
            mu = np.einsum("nkp,cp->nkc", beta_d, xf)  # (n_draws, K, n_cells) log-ratio mean
            # Gaussian copula: correlate the K normal scores within each cell via
            # the copula Cholesky, independent across draws and cells.
            z = rng.standard_normal((n_draws, len(future_w), n_lob))  # (n_draws, n_cells, K)
            eta = np.einsum("nkl,ncl->nkc", chol_d, z)  # (n_draws, K, n_cells) correlated
            # Lognormal increment = premium * exp(mu + sigma * eta); back onto the
            # loss (not loss-ratio) scale via each cell's origin premium.
            incr = c["premium"][None, :, future_w] * np.exp(mu + sigma_d[:, :, None] * eta)
            # Scatter-add each cell's increment onto its origin's running ultimate.
            np.add.at(ults, (slice(None), slice(None), np.array(future_w)), incr)

        targets = multiline_targets(c["lobs"], c["origin_periods"], premium=c["premium"])
        # Totals are derived as row-sums of the same draws (coherent diversification).
        return assemble_predictive(ults, targets, units=c["units"])

    def realized_ultimates(self, full_triangle: Triangle) -> np.ndarray:
        """Outcomes aligned to predict()'s targets, from the full triangle at
        the final dev lag."""
        if self.contract_ is None:
            raise RuntimeError("call fit() first")
        c = self.contract_
        realized = realized_multiline(
            full_triangle,
            loss_field=self._loss_field,
            dev_lag=c["n_d"] * c["dev_grain_months"],
            lobs=c["lobs"],
            origins=c["origin_periods"],
        )
        return flatten_with_totals(realized)

    # -- internals ---------------------------------------------------------------

    def _bootstrap(self, rng, n_boot: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parametric bootstrap: simulate the observed cells from the fitted
        model, refit marginals and copula per replicate.

        Each replicate regenerates log loss ratios at the *observed* design from
        the fitted marginals + copula, then re-runs the same OLS-on-logs +
        correlation estimation. Because the design is fixed and the marginal is
        OLS, a refit is one ``p x n_obs`` pseudo-inverse multiply - milliseconds
        for n_boot=200. Returns per-replicate (beta, sigma, copula Cholesky).
        """
        n_lob = self.beta_.shape[0]
        n_obs, p = self._x.shape
        chol = np.linalg.cholesky(self.corr_)  # (K, K) copula factor for simulation
        mu = self.beta_ @ self._x.T  # (K, n_obs) fitted log-ratio means

        beta_r = np.empty((n_boot, n_lob, p))  # (n_boot, K, p)
        sigma_r = np.empty((n_boot, n_lob))  # (n_boot, K)
        chol_r = np.empty((n_boot, n_lob, n_lob))  # (n_boot, K, K)
        for b in range(n_boot):
            # Simulate y* from the fitted model: correlated normal scores per cell
            # scaled by each line's sigma, added to the fitted mean.
            eta = chol @ rng.standard_normal((n_lob, n_obs))  # (K, n_obs) correlated
            y_star = mu + self.sigma_[:, None] * eta  # (K, n_obs) synthetic log ratios
            # Refit marginals (cached pinv) and the copula on the synthetic data.
            beta_b = (self._pinv @ y_star.T).T  # (K, p)
            resid = y_star - beta_b @ self._x.T  # (K, n_obs)
            sigma_b = np.sqrt((resid**2).sum(axis=1) / (n_obs - p))  # (K,)
            std = resid / sigma_b[:, None]  # (K, n_obs) standardized
            corr_b = _nearest_pd(np.corrcoef(std)) if n_lob > 1 else np.ones((1, 1))
            beta_r[b], sigma_r[b] = beta_b, sigma_b
            chol_r[b] = np.linalg.cholesky(corr_b)
        return beta_r, sigma_r, chol_r

    @staticmethod
    def _design(
        obs_w: np.ndarray, obs_d: np.ndarray, n_w: int, n_d: int, dev_effect: str
    ) -> np.ndarray:
        """Shared marginal design: intercept + origin dummies (w >= 2) + either
        dev dummies (d >= 2) or the 2-parameter Hoerl curve [ln(d), d].

        The w=0 origin and d=0 dev are the reference levels folded into the
        intercept (dummies start at index 1), so the design stays full rank.
        Returns (n_obs, p): p = 1 + (n_w-1) + (n_d-1) for factor, 1 + (n_w-1) + 2
        for hoerl.
        """
        n = len(obs_w)
        cols = [np.ones(n)]  # intercept column
        for w in range(1, n_w):
            cols.append((obs_w == w).astype(float))  # origin dummy, w=0 is reference
        if dev_effect == "factor":
            for d in range(1, n_d):
                cols.append((obs_d == d).astype(float))  # dev dummy, d=0 is reference
        else:
            # Hoerl curve: log-linear-in-dev shape, 2 params instead of n_d-1.
            dev = obs_d + 1.0  # 1-based dev step
            cols.append(np.log(dev))
            cols.append(dev)
        return np.column_stack(cols)  # (n_obs, p)

    @staticmethod
    def _check_identified(obs_w, obs_d, n_w: int, n_d: int, dev_effect: str) -> None:
        """Guard the dummy design: every origin needs a usable cell, and (factor
        case) every dev step too, or its coefficient is unidentified. Fail with a
        pointer to the coarser Hoerl parameterization rather than a rank error."""
        missing_w = [w + 1 for w in range(n_w) if not (obs_w == w).any()]
        if missing_w:
            raise ValueError(f"origins {missing_w} have no usable cells; effects unidentified")
        if dev_effect == "factor":
            missing_d = [d + 1 for d in range(n_d) if not (obs_d == d).any()]
            if missing_d:
                raise ValueError(
                    f"dev steps {missing_d} have no usable cells; "
                    'factor effects unidentified - try dev_effect="hoerl"'
                )


def _nearest_pd(mat: np.ndarray) -> np.ndarray:
    """Symmetrize and floor eigenvalues so the Cholesky factor exists."""
    sym = (mat + mat.T) / 2.0
    vals, vecs = np.linalg.eigh(sym)
    if vals.min() >= EIG_FLOOR:
        return sym
    scale = max(np.abs(vals).max(), 1.0)
    return (vecs * np.maximum(vals, EIG_FLOOR * scale)) @ vecs.T
