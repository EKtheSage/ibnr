"""Sampler convergence diagnostics, computed once for the whole Bayesian gallery.

Every Bayesian entry reports the same five numbers off its fitted posterior -
max R-hat, min bulk/tail ESS, the divergence count and its fraction of the
draws - and :class:`~ibnr.kernels.harness.ConvergenceGates` decides sampler
escalation from three of them. So the diagnostics are an evaluation algorithm
and live here rather than six times over (CLAUDE.md design decision 5); an
entry supplies only the two things that are genuinely its own, the per-backend
default parameter list and the label its results row should carry.

**Every requested name is summarized, or nothing is.** This is the same rule
``kernels.parity`` follows and it is here for the same reason. Each entry used
to filter its default list down to the names the fit happened to carry, so a
posterior missing one of them was summarized over the remainder with nothing
raised: measured on a CCL-shaped posterior whose ``a_ig`` is stuck at a
different level in every chain, dropping ``a_ig`` from the fit takes max R-hat
from 2.84 to 1.00 and min bulk ESS from 5 to 1833, and the escalation check
then passes a fit it should have re-run at more expensive settings. Nothing in
the row says the number came from four parameters instead of five. A missing
name is therefore a ``ValueError`` naming the parameter, the backend and what
the posterior does carry - it is either a default list that has drifted from
the model or a typo in an explicit request, and both want a person.

**The defaults are per BACKEND, and the key is the backend argument the entry
was fitted with, never the label it reports.** Stan is ground truth and both
ports mirror its parameter names, so for five of the six entries the three
lists are the same names written down three times. ``compartmental`` is the
exception that makes the mapping necessary: Stan and NumPyro declare the
correlated accident-year block as a ``sd_ay`` / ``L_ay`` pair, while PyMC
bundles it into one ``LKJCholeskyCov`` and reports the scales under a name of
its own. The reported label is separately settable because it is not always
the backend: a PyMC graph run through a foreign NUTS reports ``pymc:numpyro``,
which is what a published row must say, and is not a key any mapping has.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def resolve_convergence_vars(
    defaults: Mapping[str, Sequence[str]],
    backend: str | None,
    var_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """The parameters to summarize: ``var_names`` if given, else the backend's default.

    Parameters
    ----------
    defaults
        The entry's per-backend default lists, keyed by the backend argument
        its ``fit()`` took.
    backend
        Which of those lists to use. Only read when ``var_names`` is None, so
        an explicit request works on a posterior from anywhere.
    var_names
        An explicit request, which overrides the defaults entirely.

    Raises
    ------
    ValueError
        When the request is empty, or when no default list is written down for
        ``backend``. Falling back to some other backend's list would summarize
        a set nobody chose for this posterior, which is the failure this
        module exists to stop one level down.
    """
    if var_names is not None:
        names = tuple(var_names)
        if not names:
            raise ValueError(
                "convergence needs at least one parameter name to summarize; "
                "var_names is empty. Pass None for the backend's own defaults"
            )
        return names
    if backend not in defaults:
        raise ValueError(
            f"no default convergence parameters are written down for backend {backend!r}; "
            f"this entry declares {sorted(defaults)}. A fitted entry records the backend "
            "its fit() ran, so an unknown one means the entry was not fitted through fit()"
        )
    names = tuple(defaults[backend])
    if not names:
        raise ValueError(
            f"the default convergence parameters for backend {backend!r} are empty, so "
            "there is nothing to certify convergence from"
        )
    return names


def require_posterior_vars(idata, names: Sequence[str], *, backend: str | None) -> None:
    """Refuse a requested parameter the posterior does not carry, by name.

    Raises ``ValueError`` naming the missing parameters, the backend and every
    variable the posterior does carry, so a typo and a drifted default list are
    told apart by reading the message.
    """
    post = idata.posterior
    missing = [name for name in names if name not in post]
    if not missing:
        return
    carried = ", ".join(map(str, post.data_vars)) or "nothing"
    raise ValueError(
        f"convergence asked for {tuple(names)} but the {backend!r} posterior has no "
        f"{', '.join(map(repr, missing))} (it carries: {carried}). Summarizing the rest "
        "would report R-hat and ESS over a strictly smaller set than was asked for, which "
        "reads as convergence nobody measured - so this is either a default list that has "
        "drifted from the model or a misspelled request, and both need a person"
    )


def require_summarized(summary, names: Sequence[str], *, backend: str | None) -> None:
    """Refuse a parameter that reached ``arviz`` and left no row behind.

    Presence in the posterior is not the whole check: a variable of length zero
    is carried and summarizes to nothing, so max R-hat would again come from a
    smaller set than was requested. Element labels are ``name`` for a scalar
    and ``name[i]`` / ``name[i, j]`` otherwise, so the match is exact-or-braced
    rather than a prefix, which would let ``sd_k`` claim ``sd_ker``'s rows.
    """
    labels = [str(label) for label in summary.index]
    missing = [
        name
        for name in names
        if not any(label == name or label.startswith(f"{name}[") for label in labels)
    ]
    if missing:
        raise ValueError(
            f"arviz summarized no rows for {', '.join(map(repr, missing))} on the "
            f"{backend!r} posterior, so those parameters would leave the diagnostics "
            "silently. An empty (zero-length) variable is the usual cause"
        )


def convergence_report(
    idata,
    *,
    backend: str | None,
    defaults: Mapping[str, Sequence[str]],
    var_names: Sequence[str] | None = None,
    reported_backend: str | None = None,
) -> dict:
    """Max R-hat, min bulk/tail ESS, divergences and runtime for one fitted posterior.

    ``reported_backend`` defaults to ``backend`` and is what the returned row
    carries; an entry whose PyMC graph may be run by a foreign NUTS passes the
    sampler's own label instead, so a published row says what produced it.
    """
    import arviz as az

    names = resolve_convergence_vars(defaults, backend, var_names)
    require_posterior_vars(idata, names, backend=backend)
    summ = az.summary(idata, var_names=list(names))
    require_summarized(summ, names, backend=backend)

    post = idata.posterior
    n_draws = int(post.sizes["chain"] * post.sizes["draw"])
    # Divergences are the diagnostic that matters for the centered
    # parameterizations in this family; not every backend/idata carries
    # sample_stats, and None (not 0) is what "we did not measure it" means -
    # ConvergenceGates reads a missing diagnostic as one no escalation fixes.
    diverging = None
    if "sample_stats" in idata and "diverging" in idata.sample_stats:
        diverging = int(np.asarray(idata.sample_stats["diverging"].values).sum())
    return {
        "backend": backend if reported_backend is None else reported_backend,
        "runtime_s": float(idata.attrs.get("runtime_s", np.nan)),
        "n_draws": n_draws,
        "max_rhat": float(summ["r_hat"].max()),
        "min_ess_bulk": float(summ["ess_bulk"].min()),
        "min_ess_tail": float(summ["ess_tail"].min()),
        "divergences": diverging,
        "divergence_frac": (None if diverging is None else diverging / n_draws),
    }
