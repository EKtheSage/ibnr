"""Cross-backend posterior parity (design decision 7).

Before any convergence/speed comparison is meaningful, the NumPyro and PyMC
ports must be shown to target the *same posterior* as the Stan reference - a
correctness gate, not a performance one. Two correct samplers of the same model
differ only by Monte-Carlo noise, so we compare each marginal in MCSE units:

- **mean agreement**: |mean_ref - mean_port| over the combined Monte-Carlo
  standard error of the *mean*.
- **spread agreement**: |sd_ref - sd_port| over the combined MCSE of the *sd*.

Both are z-scores that are ~N(0, 1) under the null (same posterior, independent
runs). Scaling the spread check by ``mcse_sd`` - rather than a fixed fractional
tolerance - is what makes parity robust on short chains: a poorly identified
parameter (e.g. the deepest-dev ``sig`` with one observation) has a large
``mcse_sd``, so its noisier SD estimate is tolerated automatically instead of
tripping a flat 15% band. A single generous ``z_tol`` (default 4) then gates
both, flagging genuine disagreement while tolerating noise across many params.

**KS** on the pooled marginal draws is reported for context but does not gate:
MCMC autocorrelation inflates it, so it is diagnostic, not pass/fail.

Stan is ground truth; each port is compared against it. ``compare_posteriors``
works on any dict of ``arviz.InferenceData`` (e.g. numpyro-vs-pymc when cmdstan
is unavailable, as in CI).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: default parameters checked for the Meyers cross-classified family - the
#: interpretable quantities predict() consumes, not the raw nuisance draws.
CCL_PARITY_VARS = ("logelr", "alpha", "beta", "rho", "sig")

#: CSR's counterpart: same cross-classified core, with the across-accident-year
#: correlation ``rho`` replaced by the settlement-rate trend ``gamma``. ``gamma``
#: is the one parameter a port could get structurally wrong while still looking
#: plausible (it enters multiplicatively, through ``beta[d] * speedup[w]``), so
#: it must be in the compared set rather than left to the deterministic
#: ``speedup`` it drives.
CSR_PARITY_VARS = ("logelr", "alpha", "beta", "gamma", "sig")


@dataclass
class ParityReport:
    """Result of comparing one or more ported posteriors to a reference."""

    reference: str
    table: pd.DataFrame  # one row per (backend, parameter element)
    z_tol: float

    @property
    def passed(self) -> bool:
        if self.table.empty:
            return False
        return bool(
            (self.table["z_mean"].abs() <= self.z_tol).all()
            and (self.table["z_sd"].abs() <= self.z_tol).all()
        )

    def failures(self) -> pd.DataFrame:
        """Rows that breach the mean or sd z-tolerance (empty when parity holds)."""
        bad = (self.table["z_mean"].abs() > self.z_tol) | (self.table["z_sd"].abs() > self.z_tol)
        return self.table[bad]

    def summary(self) -> pd.DataFrame:
        """Worst mean/sd z and KS per compared backend."""
        g = self.table.groupby("backend")
        return pd.DataFrame(
            {
                "max_abs_z_mean": g["z_mean"].apply(lambda s: s.abs().max()),
                "max_abs_z_sd": g["z_sd"].apply(lambda s: s.abs().max()),
                "max_ks": g["ks"].max(),
                "n_params": g.size(),
                "passed": g[["z_mean", "z_sd"]].apply(
                    lambda d: bool(
                        (d["z_mean"].abs() <= self.z_tol).all()
                        and (d["z_sd"].abs() <= self.z_tol).all()
                    )
                ),
            }
        )


def _summ(idata, var_names) -> pd.DataFrame:
    """Full-precision arviz summary for the requested variables.

    ``round_to="none"`` is load-bearing, not tidiness: ``az.summary`` rounds to
    3 decimals by DEFAULT, and every z-score here is a *difference of two
    summaries divided by their MCSE*. Rounding first quantizes the numerator to
    the same 1e-3 grid for every parameter, so a parameter whose MCSE is
    ~1e-4 can be handed a rounding artifact worth several MCSE - a parity
    failure invented by the formatter. It also makes the reported z-scores land
    on exact multiples of sqrt(2), which is how this was noticed.
    """
    import arviz as az

    present = [v for v in var_names if v in idata.posterior]
    return az.summary(idata, var_names=present, kind="all", round_to="none")


def _pooled_draws(idata, param_label: str) -> np.ndarray | None:
    """Flat draw vector for one summary-style element label (e.g. ``alpha[2]``).

    Returns None when the name/index cannot be resolved, so KS is simply skipped
    for that element.
    """
    if "[" not in param_label:
        name, idx = param_label, None
    else:
        name, rest = param_label.split("[", 1)
        idx = tuple(int(i) for i in rest.rstrip("]").split(","))
    if name not in idata.posterior:
        return None
    arr = np.asarray(idata.posterior[name].values)  # (chain, draw, *dims)
    flat = arr.reshape((arr.shape[0] * arr.shape[1], *arr.shape[2:]))
    if idx is None:
        return flat.reshape(flat.shape[0], -1)[:, 0] if flat.ndim > 1 else flat
    return flat[(slice(None), *idx)]


def compare_posteriors(
    idatas: dict[str, object],
    reference: str,
    *,
    var_names=CCL_PARITY_VARS,
    z_tol: float = 4.0,
    with_ks: bool = True,
) -> ParityReport:
    """Compare each non-reference posterior in ``idatas`` to ``idatas[reference]``.

    Returns a :class:`ParityReport`; ``.passed`` is True when every compared
    parameter element agrees with the reference in both mean and SD to within
    ``z_tol`` MCSE units.
    """
    from scipy import stats

    if reference not in idatas:
        raise KeyError(f"reference backend {reference!r} not in {sorted(idatas)}")
    ref_summ = _summ(idatas[reference], var_names)
    ref_idata = idatas[reference]

    rows = []
    for backend, idata in idatas.items():
        if backend == reference:
            continue
        port_summ = _summ(idata, var_names)
        shared = ref_summ.index.intersection(port_summ.index)
        for label in shared:
            r, p = ref_summ.loc[label], port_summ.loc[label]
            mcse_mean = np.hypot(r["mcse_mean"], p["mcse_mean"])
            mcse_sd = np.hypot(r["mcse_sd"], p["mcse_sd"])
            z_mean = (r["mean"] - p["mean"]) / mcse_mean if mcse_mean > 0 else 0.0
            z_sd = (r["sd"] - p["sd"]) / mcse_sd if mcse_sd > 0 else 0.0
            ks = np.nan
            if with_ks:
                a, b = _pooled_draws(ref_idata, label), _pooled_draws(idata, label)
                if a is not None and b is not None:
                    ks = float(stats.ks_2samp(a, b).statistic)
            rows.append(
                {
                    "backend": backend,
                    "param": label,
                    "mean_ref": r["mean"],
                    "mean_port": p["mean"],
                    "z_mean": z_mean,
                    "sd_ref": r["sd"],
                    "sd_port": p["sd"],
                    "z_sd": z_sd,
                    "sd_ratio": p["sd"] / r["sd"] if r["sd"] > 0 else np.nan,
                    "ks": ks,
                }
            )
    table = pd.DataFrame(rows)
    return ParityReport(reference=reference, table=table, z_tol=z_tol)
