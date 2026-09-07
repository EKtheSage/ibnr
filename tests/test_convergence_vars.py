"""``convergence()`` summarizes every parameter it was asked for, or refuses.

Each Bayesian entry filtered its default parameter list down to the names the
fitted posterior happened to carry, and then reported max R-hat and min ESS
over whatever survived. Nothing said the set had shrunk, and
:class:`~ibnr.kernels.harness.ConvergenceGates` decides sampler escalation from
those two numbers - so a fit missing a parameter read as converged over a
strictly smaller set than requested and was never re-run. Measured on a
CCL-shaped posterior whose ``a_ig`` sits at a different level in every chain:
with ``a_ig`` present max R-hat is 2.84 and min bulk ESS 5, and the check fails
the fit; drop ``a_ig`` from the posterior and the same call answers 1.00 and
1833, and the check passes it. Same defect family as the parity leniency PR
#113 fixed one module over (issue #119).

The fix could not be a copy of #113, because the filter acted on each entry's
DEFAULT list and there was no written-down statement of what a backend should
carry to check against. So each entry now declares ``CONVERGENCE_VARS``, keyed
by the backend argument its ``fit()`` took, and
``kernels.diagnostics.convergence_report`` refuses a name the posterior lacks.

Two things these tests are careful about:

**The entry list comes from the registry**, not from a list typed out here.
A hand-written one fails in both directions - a template defect spreads down it
and a newly added entry is tested by nothing - which is how
``guszcza_growth_curve`` sat portless and unnoticed for two days after
milestone 5 closed.

**The declared lists are checked against the real model graphs**, not against
each other. NumPyro sites come from ``initialize_model``'s trace and PyMC's
from ``build_model``'s ``named_vars``: both build the actual model, neither
runs NUTS, so this stays in the fast suite while still being a statement about
the models rather than about this file. The one place the three backends
genuinely disagree is ``compartmental``, where PyMC bundles Stan's ``sd_ay`` and
``L_ay`` into a single ``LKJCholeskyCov`` and reports the scales as
``ay_chol_stds`` - the case that made a per-backend mapping necessary at all.
"""

from __future__ import annotations

import datetime as dt
import importlib
import inspect
import re

import numpy as np
import pandas as pd
import pytest

from ibnr import gallery
from ibnr.triangle import Triangle

pytest.importorskip("arviz")

#: Every Bayesian entry, DERIVED from the registry (see the module docstring).
ENTRIES = sorted(name for name in gallery.list() if gallery.get(name).family == "bayesian")


def _module(name: str):
    """The entry's own model module - where its declarations live."""
    return importlib.import_module(gallery.get(name).__module__)


def _declared(name: str) -> dict:
    """The entry's ``CONVERGENCE_VARS``, or ``{}`` when it declares none.

    Deliberately not an assertion: collection must not blow up on an entry that
    has not declared its lists yet, or the whole file reports one error and the
    per-entry picture is lost.
    """
    return getattr(_module(name), "CONVERGENCE_VARS", None) or {}


def _variants(declared: dict) -> list[str | None]:
    """``[None]`` for a flat per-backend mapping, else the variant keys.

    An undeclared entry gets ``[None]`` too, so it still contributes a case and
    fails the tests below rather than parametrizing them over nothing - the
    vacuous pass that is the same shape as the defect being fixed.
    """
    if declared and all(isinstance(v, dict) for v in declared.values()):
        return sorted(declared)
    return [None]


def _for(name: str, variant: str | None) -> dict[str, tuple[str, ...]]:
    """The per-backend mapping for one (entry, variant), asserted non-empty."""
    declared = _declared(name)
    per_backend = declared if variant is None else declared.get(variant, {})
    assert per_backend, (
        f"{name} declares no convergence parameters for variant {variant!r}, so "
        "convergence() has no written-down statement of what each backend carries"
    )
    return per_backend


#: (entry, variant) pairs - the axes a default list is allowed to depend on.
#: ``compartmental`` is the only entry with a second axis, and it is discovered
#: from the nesting of its own ``CONVERGENCE_VARS`` rather than named here.
CASES = [(name, variant) for name in ENTRIES for variant in _variants(_declared(name))]

_G = (0.30, 0.55, 0.75, 0.88, 0.95)


@pytest.fixture(scope="module")
def tiny_triangle() -> Triangle:
    """A 5x5 run-off staircase every Bayesian entry's data prep accepts.

    The same shape ``tests/test_nuts_sampler.py`` uses, and for the same
    reason: it carries all three fields the family reads under their default
    names on identical cells, so one triangle drives every entry. Nothing is
    sampled from it - it only has to survive the contract builders.
    """
    premium, loss_ratio, paid_share = 1000.0, 0.70, 0.80
    rows = []
    n = len(_G)
    for i in range(n):
        for j in range(n - i):
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


@pytest.fixture(scope="module")
def contracts(tiny_triangle) -> dict[tuple[str, str | None], dict]:
    """Each entry's own contract dict, captured out of a real ``fit()``.

    The sampler is replaced, so the data prep and the dispatch are the entry's
    real ones and no NUTS starts. Building the contract by hand here would test
    this file's idea of the data block rather than the entry's.
    """
    out = {}
    for name, variant in CASES:
        captured = {}

        def recorder(contract, _captured=captured, **kwargs):
            _captured["contract"] = contract
            return "not-an-idata", None

        entry = gallery.get(name)()
        entry._sample_numpyro = recorder
        kwargs = {} if variant is None else {"variant": variant}
        entry.fit(tiny_triangle, backend="numpyro", **kwargs)
        out[(name, variant)] = captured["contract"]
    return out


def _numpyro_model(module, variant: str | None):
    """The port's model function, discovered rather than named in a table."""
    candidates = sorted(
        key
        for key, value in vars(module).items()
        if key.endswith("_model")
        and inspect.isfunction(value)
        and value.__module__ == module.__name__
    )
    if variant is not None:
        want = f"{variant}_model"
        assert want in candidates, f"{module.__name__} has no {want}: {candidates}"
        return getattr(module, want)
    assert len(candidates) == 1, f"{module.__name__} has {candidates}, expected exactly one"
    return getattr(module, candidates[0])


def _stub(name: str, *, backend: str, variant: str | None, posterior: dict, attrs=None):
    """A fitted-enough entry: real class, synthetic posterior.

    ``convergence()`` reads only ``idata_``, ``backend_`` and (compartmental)
    ``variant_``, so this exercises the real method without a sampler. The draws
    are meaningless on purpose - what is under test is WHICH parameters reach
    the summary, not what they say.
    """
    import arviz as az

    entry = gallery.get(name)()
    entry.idata_ = az.from_dict(posterior=posterior)
    if attrs:
        entry.idata_.attrs.update(attrs)
    entry.backend_ = backend
    if variant is not None:
        entry.variant_ = variant
    return entry


def _draws(names, *, stuck=None, n_chain=4, n_draw=400, seed=0) -> dict:
    """One 2-D array per name; ``stuck`` gets a different level per chain, so
    its R-hat is enormous and its bulk ESS tiny."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in names:
        arr = rng.normal(0.0, 1.0, size=(n_chain, n_draw))
        if name == stuck:
            arr = arr * 0.01 + np.arange(n_chain)[:, None] * 5.0
        out[name] = arr
    return out


# -- what is written down ----------------------------------------------------


def test_the_entry_list_is_the_whole_bayesian_family():
    """The list above must BE the family, and must not be able to collapse.

    The floor stops a registry that failed to import from turning every
    parametrized test below into a silent pass on zero cases.
    """
    from_registry = {name for name in gallery.list() if gallery.get(name).family == "bayesian"}
    assert set(ENTRIES) == from_registry
    assert len(ENTRIES) >= 6, f"the bayesian family cannot have shrunk: {sorted(ENTRIES)}"


@pytest.mark.parametrize("name", ENTRIES)
def test_every_entry_writes_its_defaults_down(name):
    """``CONVERGENCE_VARS`` exists on every entry in the family.

    The whole fix rests on it: without a written-down statement of what a
    backend should carry there is nothing for a strict check to compare a
    posterior against, which is why the six presence filters could not simply
    be deleted when ``kernels.parity``'s equivalent was (issue #119).
    """
    assert _declared(name), (
        f"{name} declares no CONVERGENCE_VARS, so convergence() has no written-down "
        "statement of what each backend should carry"
    )


@pytest.mark.parametrize(("name", "variant"), CASES)
def test_every_backend_has_a_default_list(name, variant):
    """One list per backend the entry can dispatch to, non-empty and unique.

    Keyed on the entry's own ``BACKENDS``, so a fourth backend cannot be added
    without saying what it carries - which is the only way ``convergence()``
    could go back to guessing.
    """
    mod = _module(name)
    per_backend = _for(name, variant)
    assert set(per_backend) == set(mod.BACKENDS), (
        f"{name} declares convergence parameters for {sorted(per_backend)} "
        f"but dispatches to {sorted(mod.BACKENDS)}"
    )
    for backend, names in per_backend.items():
        assert names, f"{name}/{backend} declares an empty list"
        assert len(set(names)) == len(names), f"{name}/{backend} repeats a parameter: {names}"


@pytest.mark.parametrize(("name", "variant"), CASES)
def test_the_numpyro_defaults_are_real_sites(name, variant, contracts):
    """Every declared NumPyro name is a site in the port's actual model graph.

    ``initialize_model`` is what builds it: a plain ``handlers.trace`` cannot,
    because three of these models place ``ImproperUniform`` priors (Stan's
    half-Student-t construction, which NumPyro has no sampler for) and tracing
    one raises ``NotImplementedError``.
    """
    jax = pytest.importorskip("jax")
    pytest.importorskip("numpyro")
    from numpyro.infer.util import initialize_model

    module = importlib.import_module(f"{_module(name).__package__}.model_numpyro")
    model = _numpyro_model(module, variant)
    trace = initialize_model(
        jax.random.PRNGKey(0), model, model_args=(contracts[(name, variant)],)
    ).model_trace
    sites = {key for key, site in trace.items() if site["type"] in ("sample", "deterministic")}
    missing = sorted(set(_for(name, variant)["numpyro"]) - sites)
    assert not missing, f"{name}/{variant} declares {missing}, absent from the numpyro graph"


@pytest.mark.parametrize(("name", "variant"), CASES)
def test_the_pymc_defaults_are_real_variables(name, variant, contracts):
    """Every declared PyMC name is a variable in the port's actual model graph.

    This is the test that pins ``compartmental``'s ``ay_chol_stds``: it is not
    a name anyone could infer from ``model.stan``, and the whole per-backend
    mapping exists because of it.
    """
    pytest.importorskip("pymc")

    module = importlib.import_module(f"{_module(name).__package__}.model_pymc")
    kwargs = {} if variant is None else {"variant": variant}
    model = module.build_model(contracts[(name, variant)], **kwargs)
    missing = sorted(set(_for(name, variant)["pymc"]) - set(model.named_vars))
    assert not missing, f"{name}/{variant} declares {missing}, absent from the pymc graph"


@pytest.mark.parametrize(("name", "variant"), CASES)
def test_the_stan_defaults_are_declared_in_the_program(name, variant):
    """Every declared Stan name is declared in the entry's ``.stan`` source.

    A tripwire rather than a proof, and deliberately the weakest test here: it
    reads the program text because compiling one needs cmdstan, which the fast
    suite does not have. It catches the case worth catching - a parameter
    renamed in the Stan file and not in the mapping - and cannot see a name
    that only appears in a comment.
    """
    mod = _module(name)
    stan_file = mod.STAN_FILE if variant is None else mod.STAN_FILES[variant]
    source = stan_file.read_text(encoding="utf-8")
    # strip line comments so a name mentioned only in prose cannot vouch for itself
    code = re.sub(r"//[^\n]*", "", source)
    missing = [
        n for n in _for(name, variant)["stan"] if not re.search(rf"\b{re.escape(n)}\b", code)
    ]
    assert not missing, f"{name}/{variant} declares {missing}, absent from {stan_file.name}"


# -- the refusal -------------------------------------------------------------


@pytest.mark.parametrize(("name", "variant"), CASES)
def test_a_default_the_posterior_lost_is_refused_and_would_have_changed_the_answer(name, variant):
    """The defect, both halves, for EVERY declared parameter of every backend.

    First half: with all of them present and one stuck at a different level per
    chain, the diagnostics must see that one - max R-hat lands far above any
    escalation threshold. Run per parameter rather than on a representative
    one, because that is what makes this a statement about the whole declared
    list: a name that is listed but never reaches the summary (a default the
    entry resolves from the wrong variant, say) is invisible to a spot check
    and shows up here as a healthy 1.00.

    Second half: take that same parameter out of the posterior and the call
    must refuse it by name. It used to answer over the remaining parameters,
    and since the survivors are healthy the answer was a clean R-hat that
    ``ConvergenceGates`` accepts and never escalates.
    """
    for backend, names in _for(name, variant).items():
        for stuck in names:
            healthy = _stub(
                name,
                backend=backend,
                variant=variant,
                posterior=_draws(names, stuck=stuck, n_chain=2, n_draw=200),
            )
            assert healthy.convergence()["max_rhat"] > 1.5, (
                f"{name}/{variant}/{backend}: {stuck} is declared but did not reach the summary"
            )

            lost = _stub(
                name,
                backend=backend,
                variant=variant,
                posterior=_draws([n for n in names if n != stuck], n_chain=2, n_draw=200),
            )
            with pytest.raises(ValueError, match=re.escape(repr(stuck))):
                lost.convergence()


@pytest.mark.parametrize(("name", "variant"), CASES)
def test_an_explicit_request_for_a_missing_name_is_refused(name, variant):
    """A typo in ``var_names`` is refused rather than quietly dropped.

    The one test here that reads nothing out of ``CONVERGENCE_VARS`` - the
    posterior is two invented names - so it says the same thing before and
    after the fix, and it is the cleanest pre-fix red: the old call filtered
    the request down to the name that existed and returned an ordinary
    diagnostics dict over it, with no indication that half the request was
    dropped on the floor.
    """
    entry = _stub(
        name,
        backend="stan",
        variant=variant,
        posterior=_draws(["present_parameter"]),
    )
    with pytest.raises(ValueError, match="not_a_parameter"):
        entry.convergence(var_names=["present_parameter", "not_a_parameter"])


@pytest.mark.parametrize(("name", "variant"), CASES)
def test_a_posterior_carrying_the_declared_names_is_summarized(name, variant):
    """The happy path, per backend: every declared name summarizes cleanly.

    Including ``compartmental``'s PyMC list with its ``LKJCholeskyCov`` naming,
    which is the acceptance criterion the per-backend mapping exists to meet.
    """
    for backend, names in _for(name, variant).items():
        entry = _stub(name, backend=backend, variant=variant, posterior=_draws(names))
        conv = entry.convergence()
        assert conv["backend"] == backend
        assert conv["n_draws"] == 1600
        assert 0.99 <= conv["max_rhat"] <= 1.02
        assert conv["min_ess_bulk"] > 100


def test_compartmental_pymc_uses_the_lkj_bundle_and_stan_does_not():
    """The named example: PyMC has no ``sd_ay`` and must not be asked for one.

    Stan and NumPyro declare ``sd_ay`` beside ``L_ay``; PyMC's
    ``LKJCholeskyCov`` is both at once and reports the scales as
    ``ay_chol_stds``. A single shared list would therefore have had to drop the
    scales from one backend or the other - which is exactly what the presence
    filter did, silently, on every PyMC fit of this entry.
    """
    from ibnr.gallery.bayesian.compartmental.model import CONVERGENCE_VARS

    for variant in ("gaussian", "lognormal"):
        per_backend = CONVERGENCE_VARS[variant]
        assert "sd_ay" in per_backend["stan"]
        assert "sd_ay" in per_backend["numpyro"]
        assert "sd_ay" not in per_backend["pymc"]
        assert "ay_chol_stds" in per_backend["pymc"]
        # the two lists otherwise describe the same model
        assert set(per_backend["pymc"]) - {"ay_chol_stds"} == set(per_backend["stan"]) - {"sd_ay"}


def test_the_variant_selects_the_list():
    """compartmental's two variants are different models, not one with holes.

    Model 1 gives ker and kp no varying effects at all, so its ``sd_dev`` /
    ``sd_ker`` / ``sd_kp`` do not exist. The old single list named them anyway
    and leaned on the presence filter to drop them, which is precisely the
    behaviour that also dropped parameters that should have been there.
    """
    from ibnr.gallery.bayesian.compartmental.model import CONVERGENCE_VARS

    gaussian = set(CONVERGENCE_VARS["gaussian"]["stan"])
    lognormal = set(CONVERGENCE_VARS["lognormal"]["stan"])
    assert gaussian < lognormal
    assert lognormal - gaussian == {"sd_dev", "sd_ker", "sd_kp"}


# -- the shared kernel -------------------------------------------------------


def test_an_unknown_backend_has_no_defaults_to_fall_back_on():
    """No list for this backend is a refusal, never another backend's list.

    Silently using a neighbour's would summarize a set nobody chose for this
    posterior - the same failure one level up from the one being fixed.
    """
    from ibnr.kernels.diagnostics import resolve_convergence_vars

    with pytest.raises(ValueError, match="no default convergence parameters"):
        resolve_convergence_vars({"stan": ("a",)}, "numpyro")


def test_an_empty_request_is_refused():
    """Summarizing nothing would report a diagnostic over no parameters."""
    from ibnr.kernels.diagnostics import resolve_convergence_vars

    with pytest.raises(ValueError, match="at least one parameter"):
        resolve_convergence_vars({"stan": ("a",)}, "stan", [])
    with pytest.raises(ValueError, match="are empty"):
        resolve_convergence_vars({"stan": ()}, "stan")


def test_a_parameter_that_leaves_no_summary_row_is_refused():
    """Present in the posterior is not the same as summarized.

    A zero-length variable is carried by the posterior and contributes no rows,
    so a presence check alone would let max R-hat come from a smaller set again
    - the identical defect one step further in.
    """
    import arviz as az

    from ibnr.kernels.diagnostics import convergence_report

    idata = az.from_dict(posterior={"logelr": np.zeros((2, 50)), "r_alpha": np.zeros((2, 50, 0))})
    with pytest.raises(ValueError, match="summarized no rows"):
        convergence_report(idata, backend="stan", defaults={"stan": ("logelr", "r_alpha")})


def test_the_reported_label_can_differ_from_the_key():
    """A PyMC graph run by a foreign NUTS reports ``pymc:numpyro``.

    The label says what produced the row; the DEFAULTS are still keyed on the
    backend argument, because ``pymc:numpyro`` names a sampler, not a graph, and
    no mapping has it. ``guszcza_growth_curve`` is the entry that does this.
    """
    from ibnr.gallery.bayesian.guszcza_growth_curve.model import CONVERGENCE_VARS

    entry = _stub(
        "guszcza_growth_curve",
        backend="pymc",
        variant=None,
        posterior=_draws(CONVERGENCE_VARS["pymc"]),
        attrs={"backend": "pymc:numpyro"},
    )
    assert entry.convergence()["backend"] == "pymc:numpyro"


@pytest.mark.parametrize("name", ENTRIES)
def test_convergence_still_refuses_an_unfitted_entry(name):
    """The pre-existing contract, unchanged: no posterior is a RuntimeError,
    not whatever the new resolution would make of ``backend_ = None``."""
    with pytest.raises(RuntimeError, match="call fit"):
        gallery.get(name)().convergence()
