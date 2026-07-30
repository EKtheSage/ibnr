"""``nuts_sampler`` is reachable through every Bayesian entry's ``fit()``.

``nuts_sampler`` selects which NUTS implementation runs the *same* PyMC graph:
``"pymc"`` (native PyTensor, the default and the parity reference) or
``"nutpie"`` / ``"numpyro"`` / ``"blackjax"``. Every ``model_pymc.py`` has
supported it since the ports were written, but until now no ENTRY exposed it -
so ``gallery.fit(..., backend="pymc")`` could only reach the native path. On
``compartmental`` that path costs ~0.6-0.9 s/iteration against NumPyro's
~0.007, i.e. it is effectively unrunnable, which made the gallery's pymc
backend a trap for that entry. ``guszcza_growth_curve`` is the second such
entry - at the ``adapt_delta = 0.999`` its source specifies, native PyTensor
took 860 s against NumPyro's 6.7 s on the identical graph.

Two design points these tests pin, both deliberate:

**The default stays ``"pymc"``.** Defaulting to a faster foreign sampler would
quietly turn the cross-backend convergence comparison (design decision 7, and
the whole point of milestone 5) into NumPyro measured against NumPyro. The
runtime tables in the cards would then be comparing one sampler with itself
over two graph representations. The fast path is a documented CHOICE, never a
silent substitution.

**The backend label records which NUTS actually ran** - ``"pymc:numpyro"``, not
``"pymc"`` - so a swapped sampler can never masquerade as a native run in a
results CSV or a model card.
"""

from __future__ import annotations

import datetime as dt
import inspect

import pandas as pd
import pytest

from ibnr import gallery
from ibnr.triangle import Triangle

#: Every Bayesian entry, DERIVED from the registry rather than written out.
#:
#: Every entry in the family has a PyMC port, and CLAUDE.md's milestone-5 rule is
#: that a new Bayesian entry is not done until it has both ports - so deriving
#: this list is what makes that rule enforceable rather than advisory. A
#: Stan-only entry joining the family fails these tests on its first CI run,
#: which is exactly what a hardcoded list did NOT do: `guszcza_growth_curve`
#: landed two days after milestone 5 closed and sat portless, absent from this
#: list and from scripts/parity_gallery.py, with nothing going red.
ENTRIES = sorted(name for name in gallery.list() if gallery.get(name).family == "bayesian")


def test_the_entry_list_is_the_whole_bayesian_family():
    """The list above must BE the family, and must not be able to collapse.

    Set equality stops it drifting; the floor stops a registry that failed to
    import from turning every parametrized test below into a silent pass on zero
    cases - the failure mode CI's own count gates exist to catch one level up.
    """
    from_registry = {name for name in gallery.list() if gallery.get(name).family == "bayesian"}
    assert set(ENTRIES) == from_registry
    assert len(ENTRIES) >= 6, f"the bayesian family cannot have shrunk: {sorted(ENTRIES)}"


#: strictly increasing share of ultimate paid by development year. Increments are
#: positive at every step, which ODP requires and the lognormal entries prefer.
_G = (0.30, 0.55, 0.75, 0.88, 0.95)


@pytest.fixture(scope="module")
def tiny_triangle() -> Triangle:
    """A 5x5 run-off staircase every Bayesian entry's data prep accepts.

    Carries all three fields the family reads under their default names
    (``paid_loss`` / ``reported_loss`` / ``earned_premium``) on identical cells,
    so one triangle drives the dispatch test for every entry in the family.
    Reported is
    paid grossed up, so compartmental's derived OS = reported - paid is positive.
    Nothing here is sampled - it only has to survive ``stan_data()``.
    """
    premium, loss_ratio, paid_share = 1000.0, 0.70, 0.80
    rows = []
    n = len(_G)
    for i in range(n):
        for j in range(n - i):  # upper triangle: the 5th origin has one cell
            paid = premium * loss_ratio * _G[j]
            for field, value in (
                ("paid_loss", paid),
                ("reported_loss", paid / paid_share),
                ("earned_premium", premium),
            ):
                rows.append(
                    {
                        "origin_period": dt.date(2010 + i, 1, 1),
                        "dev_lag": 12 * (j + 1),
                        "eval_date": dt.date(2010 + i + j, 12, 31),
                        "field": field,
                        "value": value,
                    }
                )
    return Triangle.from_long(pd.DataFrame(rows), measure="cumulative")


@pytest.mark.parametrize("name", ENTRIES)
def test_fit_exposes_nuts_sampler_defaulting_to_native(name):
    """Reachable through the gallery, and defaulting to native PyMC.

    The default is the assertion that matters: see the module docstring on why
    a faster default would hollow out the convergence comparison.
    """
    param = inspect.signature(gallery.get(name).fit).parameters.get("nuts_sampler")
    assert param is not None, f"{name}.fit() does not expose nuts_sampler"
    assert param.default == "pymc", f"{name} must default to the native sampler"


@pytest.mark.parametrize("name", ENTRIES)
def test_sample_pymc_accepts_it(name):
    """The entry's PyMC wrapper must take the argument.

    Necessary but nowhere near sufficient - this only inspects a signature. What
    the argument is worth is whether ``fit()`` DELIVERS it, which is the next
    test's job.
    """
    assert "nuts_sampler" in inspect.signature(gallery.get(name)._sample_pymc).parameters


@pytest.mark.parametrize("name", ENTRIES)
@pytest.mark.parametrize("requested", ["pymc", "numpyro", "nutpie"])
def test_fit_delivers_it_to_the_pymc_sampler(name, requested, tiny_triangle):
    """``fit()`` must hand the requested sampler to ``_sample_pymc``.

    This is the assertion with teeth, and the reason it exists: the parameter
    was originally exposed on ``fit()`` and validated there, but the dispatch
    built a kwargs dict carrying only the *stan* controls, so the value was
    dropped on the floor and ``_sample_pymc`` fell back to its own ``"pymc"``
    default. Every signature-level test above still passed. The failure mode is
    the nastiest kind - a fit that runs fine, produces a plausible posterior,
    and is labelled ``"pymc"`` while the caller believes they asked for
    something else.

    Monkeypatching the sampler keeps this in the fast suite: real data prep
    runs, real dispatch runs, and no NUTS of any flavour is started.
    """
    seen = {}

    def recorder(*args, **kwargs):
        seen.update(kwargs)
        return "not-an-idata", None

    entry = gallery.get(name)()
    entry._sample_pymc = recorder  # instance attribute wins the self._sample_pymc lookup
    entry.fit(tiny_triangle, backend="pymc", nuts_sampler=requested)

    assert seen.get("nuts_sampler") == requested, (
        f"{name}.fit(nuts_sampler={requested!r}) reached _sample_pymc as "
        f"{seen.get('nuts_sampler')!r}"
    )


@pytest.mark.parametrize("name", ENTRIES)
def test_non_pymc_backends_are_not_handed_a_nuts_sampler(name, tiny_triangle):
    """The mirror of the guard below: stan and numpyro must not merely reject a
    non-default value, they must never RECEIVE the argument at all - their
    samplers do not take one, so leaking it would be a TypeError at sample time.
    """
    seen = {}

    def recorder(*args, **kwargs):
        seen.update(kwargs)
        return "not-an-idata", None

    entry = gallery.get(name)()
    entry._sample_numpyro = recorder
    entry.fit(tiny_triangle, backend="numpyro")

    assert "nuts_sampler" not in seen


@pytest.mark.parametrize("name", ENTRIES)
def test_non_pymc_backends_reject_it(name):
    """``nuts_sampler`` is meaningless for Stan and NumPyro, so asking for one
    there is an error rather than a silently ignored argument - the same rule
    ``parallel_chains`` follows. Validated before any data prep, so it costs
    nothing to be wrong."""
    with pytest.raises(ValueError, match="pymc-backend control"):
        gallery.get(name)().fit(None, backend="numpyro", nuts_sampler="numpyro")


@pytest.mark.parity
@pytest.mark.slow
def test_swapped_sampler_is_labelled_distinctly():
    """The label must record which NUTS ran, so provenance survives into the
    results CSVs. A native run is ``pymc``; a swapped one is ``pymc:<name>``."""
    pytest.importorskip("pymc")
    pytest.importorskip("numpyro")
    from ibnr.gallery.bayesian.england_verrall_odp import model_pymc

    from .test_parity_odp import simulate_odp_contract

    data = simulate_odp_contract(n_w=5, seed=1)
    native = model_pymc.sample(data, chains=1, iter_warmup=200, iter_sampling=200, seed=2)
    swapped = model_pymc.sample(
        data, chains=1, iter_warmup=200, iter_sampling=200, seed=2, nuts_sampler="numpyro"
    )
    assert native.attrs["backend"] == "pymc"
    assert swapped.attrs["backend"] == "pymc:numpyro"
