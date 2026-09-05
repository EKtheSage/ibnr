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

**Every requested name is compared, on every backend.** A comparison that
quietly dropped a variable one side did not carry would be an ``.all()`` over
the rows that happened to survive, which is a pass nobody earned: the input
this comparison exists to judge is a port under development, and a missing or
misnamed variable is exactly how such a port is wrong. So a requested name
absent from any posterior (the reference included), an element shape that
differs from the reference's, and a non-finite draw are each refused by name
with a ``ValueError``.

**Point masses** are handled explicitly rather than by dividing by a Monte
Carlo error that is zero for the mean and undefined for the sd. CCL, CSR and
ODP all pin an identifiability anchor (``alpha[1] = 0``, ``beta[n_d] = 0`` in
``model.stan``, written as literal zeros in both ports), so exact constants are
ordinary input here. Two equal constants agree perfectly and score z = 0 for
both mean and sd. Two constants that differ disagree completely and score
``z_mean = inf``, which fails the comparison and appears in ``failures()`` and
``summary()`` rather than being refused: the caller asked for a verdict, and
that is the verdict. A constant on one side only contributes 0 to the combined
error, so the other side's error carries the comparison and a pin the port left
free is caught by the sd z-score. When the combined error of a marginal that is
not constant is not a positive finite number, there is nothing to certify from
and the comparison refuses.

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

#: England & Verrall ODP. A different family: no lognormal ``sig`` (the
#: dispersion ``phi`` is a plug-in constant, not a parameter), so the compared
#: set is just the log-link linear predictor's coefficients - the intercept and
#: the two zero-pinned effect vectors.
ODP_PARITY_VARS = ("c", "alpha", "beta")

#: Bayesian Clark growth curve. Only three parameters, and all three matter:
#: ``logelr`` sets the Cape Cod level while ``omega``/``theta`` are the curve's
#: shape and scale, which trade off along a ridge - so a port that got the
#: curve subtly wrong would show up here as a shifted (omega, theta) pair even
#: when the fitted development pattern looks similar.
CLARK_PARITY_VARS = ("logelr", "omega", "theta")

#: Hierarchical growth curve (Guszcza / Gesmann). The population intercepts, the
#: curve's shape and scale, the hierarchical spread and the residual scale - plus
#: ``ulr`` itself, the per-accident-year ultimate loss ratios, because they are
#: what ``predict()`` and the held-out scorer actually consume (the same reason
#: CCL compares ``alpha``). The raw ``z_ulr`` is deliberately excluded: it is the
#: non-centered nuisance whose only content is ``ulr``, exactly as CCL excludes
#: ``a_ig``. Comparing in MCSE units is what makes including ``ulr`` safe - a
#: late origin with two cells has a large ``mcse_sd``, so its noisier estimate is
#: tolerated automatically rather than tripping a fixed band.
GUSZCZA_PARITY_VARS = ("ulr_pop", "sd_ulr", "ulr", "omega", "theta", "sigma")

#: Hierarchical compartmental. Deliberately the POPULATION-level scalars plus
#: the two residual scales and the reserving-cycle correlation - the set both
#: variants expose under the same names, in all three backends. The per-accident
#: -year RLR/RRF and the rates are NOT compared: Model 1 carries one set per
#: accident year while Model 2 resolves them per cell, so they are not the same
#: quantity across variants. PyMC also bundles ``sd_ay``/``L_ay`` into a single
#: ``LKJCholeskyCov`` variable where Stan keeps them separate, so those raw
#: nuisance draws have no common name to compare on either - ``rho_ay`` is the
#: interpretable summary of exactly that block and stands in for it.
COMPARTMENTAL_PARITY_VARS = (
    "b_oRLR",
    "b_oRRF",
    "b_oker",
    "b_okp",
    "sigma_os",
    "sigma_paid",
    "rho_ay",
)


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

    return az.summary(idata, var_names=list(var_names), kind="all", round_to="none")


def _require_posterior_vars(idatas: dict[str, object], reference: str, var_names) -> None:
    """Refuse anything the comparison cannot judge, before any summary is read.

    Checks every requested name on every backend, the reference included, and
    raises ``ValueError`` naming the backend and the variable when: the name is
    absent; its element shape differs from the reference's; or any draw is not
    finite. Also refuses an empty request. Without this the missing element
    simply contributed no row and the report passed on the rest, and a NaN draw
    made every arviz summary field NaN, which the z-score arithmetic used to
    read as a zero discrepancy.
    """
    names = list(var_names)
    if not names:
        raise ValueError("parity needs at least one variable name; var_names is empty")

    ref_post = idatas[reference].posterior
    ref_shapes: dict[str, tuple[int, ...]] = {}
    for name in names:
        if name not in ref_post:
            raise ValueError(
                f"parity requested {tuple(names)} but reference posterior {reference!r} "
                f"has no {name!r} (it carries: {', '.join(map(str, ref_post.data_vars))})"
            )
        ref_shapes[name] = tuple(np.asarray(ref_post[name].values).shape[2:])

    for backend, idata in idatas.items():
        post = idata.posterior
        for name in names:
            if name not in post:
                raise ValueError(
                    f"parity requested {tuple(names)} but posterior {backend!r} has no "
                    f"{name!r} (it carries: {', '.join(map(str, post.data_vars))})"
                )
            draws = np.asarray(post[name].values)
            shape = tuple(draws.shape[2:])
            if shape != ref_shapes[name]:
                raise ValueError(
                    f"{name!r} has element shape {shape} on {backend!r} but "
                    f"{ref_shapes[name]} on reference {reference!r}; "
                    "parity compares the same quantity or nothing"
                )
            n_bad = int((~np.isfinite(draws)).sum())
            if n_bad:
                raise ValueError(
                    f"posterior {backend!r} has {n_bad} non-finite draw(s) in {name!r}; "
                    "parity needs finite draws"
                )


def _z_scores(r, p, ref_draws, port_draws, *, label, backend, reference) -> tuple[float, float]:
    """Mean and sd z-scores for one element, with point masses handled by hand.

    A constant marginal has no Monte Carlo error at all, in either its mean or
    its sd, so each constant side contributes 0.0 to both combined errors. That
    matters because arviz reports ``mcse_sd`` as NaN for a constant and
    ``np.hypot(nan, x)`` is NaN, which used to send the whole row down a branch
    that scored it 0.0 - the best possible score - whatever the two summaries
    said. Draws are known finite here (see ``_require_posterior_vars``).
    """
    const_r = float(np.ptp(ref_draws)) == 0.0
    const_p = float(np.ptp(port_draws)) == 0.0
    if const_r and const_p:
        # Both exact point masses: equal is perfect agreement, unequal is total
        # disagreement. inf needs no special handling downstream, since
        # ``inf <= z_tol`` is False.
        return (0.0 if r["mean"] == p["mean"] else np.inf), 0.0

    mcse_mean = np.hypot(0.0 if const_r else r["mcse_mean"], 0.0 if const_p else p["mcse_mean"])
    mcse_sd = np.hypot(0.0 if const_r else r["mcse_sd"], 0.0 if const_p else p["mcse_sd"])
    for what, combined in (("mean", mcse_mean), ("sd", mcse_sd)):
        if not np.isfinite(combined) or combined <= 0.0:
            raise ValueError(
                f"the combined Monte Carlo standard error of the {what} of {label!r} "
                f"({reference!r} vs {backend!r}) is {combined}, not a positive number; "
                "parity cannot be certified from it"
            )
    return float((r["mean"] - p["mean"]) / mcse_mean), float((r["sd"] - p["sd"]) / mcse_sd)


def _pooled_draws(idata, param_label: str) -> np.ndarray:
    """Flat draw vector for one summary-style element label (e.g. ``alpha[2]``).

    Raises when the name cannot be resolved. Every label reaching here comes
    from a summary of names ``_require_posterior_vars`` already checked, so a
    miss is a bug in this module rather than a reason to skip the element.
    """
    if "[" not in param_label:
        name, idx = param_label, None
    else:
        name, rest = param_label.split("[", 1)
        idx = tuple(int(i) for i in rest.rstrip("]").split(","))
    if name not in idata.posterior:
        raise ValueError(f"no posterior variable {name!r} behind summary label {param_label!r}")
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

    Every REQUESTED name is compared, with the reference's element shape, on
    every backend, or a ``ValueError`` says which backend and which name is the
    problem. Returns a :class:`ParityReport`; ``.passed`` is True when every
    requested parameter element agrees with the reference in both mean and SD to
    within ``z_tol`` MCSE units. Two constants that differ are a failure the
    report carries (``z_mean = inf``), not a refusal.
    """
    from scipy import stats

    if reference not in idatas:
        raise KeyError(f"reference backend {reference!r} not in {sorted(idatas)}")
    _require_posterior_vars(idatas, reference, var_names)
    ref_summ = _summ(idatas[reference], var_names)
    ref_idata = idatas[reference]

    rows = []
    for backend, idata in idatas.items():
        if backend == reference:
            continue
        port_summ = _summ(idata, var_names)
        if not port_summ.index.equals(ref_summ.index):
            # The names and element shapes already agree, so the only way to
            # get here is one side attaching coordinate labels the other does
            # not. No port does that today; this is the backstop that keeps a
            # future one from narrowing the comparison silently.
            differing = sorted(set(ref_summ.index) ^ set(port_summ.index))
            raise ValueError(
                f"posterior {backend!r} summarizes different elements than reference "
                f"{reference!r}; first difference: {differing[0] if differing else 'ordering'}"
            )
        for label in ref_summ.index:
            r, p = ref_summ.loc[label], port_summ.loc[label]
            a, b = _pooled_draws(ref_idata, label), _pooled_draws(idata, label)
            z_mean, z_sd = _z_scores(r, p, a, b, label=label, backend=backend, reference=reference)
            ks = float(stats.ks_2samp(a, b).statistic) if with_ks else np.nan
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
