"""``nuts_sampler`` is reachable through every Bayesian entry's ``fit()``.

``nuts_sampler`` selects which NUTS implementation runs the *same* PyMC graph:
``"pymc"`` (native PyTensor, the default and the parity reference) or
``"nutpie"`` / ``"numpyro"`` / ``"blackjax"``. Every ``model_pymc.py`` has
supported it since the ports were written, but until now no ENTRY exposed it -
so ``gallery.fit(..., backend="pymc")`` could only reach the native path. On
``compartmental`` that path costs ~0.6-0.9 s/iteration against NumPyro's
~0.007, i.e. it is effectively unrunnable, which made the gallery's pymc
backend a trap for that entry.

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

import inspect

import pytest

from ibnr import gallery

#: every entry with a PyMC port
ENTRIES = [
    "meyers_ccl",
    "meyers_csr",
    "england_verrall_odp",
    "clark_growth_curve",
    "compartmental",
]


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
    """The entry's PyMC wrapper must actually take the argument - exposing it on
    ``fit()`` without threading it through would fail only at sample time."""
    assert "nuts_sampler" in inspect.signature(gallery.get(name)._sample_pymc).parameters


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
