"""Cross-backend parity for compartmental (milestone 5).

The hardest entry in the gallery, and the only one with two ablatable variants,
so the fast tier here is bigger than elsewhere: it pins the three constructs
that a hierarchical brms/Stan model can silently lose in translation.

**1. The positivity of ``sd_ay`` is load-bearing, not cosmetic.** Stan declares
``vector<lower=0>[2] sd_ay``. If a port drops that constraint the model acquires
an exact sign symmetry - with ``u = diag(sd) L z`` and ``L[0,0] = 1``, the map
``(sd_0, z[0,:], L[1,0]) -> (-sd_0, -z[0,:], -L[1,0])`` leaves ``u_ay``, and
hence the entire likelihood, invariant. The posterior becomes bimodal and
symmetric in rho, so ``rho_ay`` averages to ~0 and the model silently reports
"no reserving cycle" - which is its headline actuarial output. Nothing crashes.
Same failure family as the ``a_ig`` bound in the Meyers family.

**2. LKJ(1) on a 2x2 Cholesky must give rho uniform on (-1, 1).** Worth
checking directly because ``lkj_corr_cholesky(1)`` contributes *literally zero*
to Stan's target - the entire geometry comes from the constraint transform's
Jacobian. A port that gets the density term right and the transform wrong looks
correct on inspection and is wrong in the trace.

**3. The closed-form ODE curves must agree in all THREE implementations** -
Stan's ``functions`` block, mirrored in numpy (``model.py``, used by
``predict()``) and in jnp/pytensor (the ports). Three copies of the same
algebra is exactly where a transcription slip hides.
"""

from __future__ import annotations

import numpy as np
import pytest

from ibnr.kernels.parity import COMPARTMENTAL_PARITY_VARS

VARIANTS = ["gaussian", "lognormal"]


def make_data(n_w: int = 5, variant: str = "gaussian", seed: int = 0) -> dict:
    """A compartmental Stan data block, self-contained (no mart, no Triangle).

    Rows are delta-stacked (outstanding then paid) over a run-off triangle, and
    the values are generated from the model's own closed-form curves at the
    prior medians, so the fitted model is correctly specified.
    """
    from ibnr.gallery.bayesian.compartmental.model import os_curve, paid_curve

    rng = np.random.default_rng(seed)
    prem = rng.uniform(8000, 20000, n_w)
    w, d, t, delta = [], [], [], []
    for wi in range(1, n_w + 1):
        for di in range(1, n_w - wi + 2):
            for dl in (0, 1):
                w.append(wi)
                d.append(di)
                t.append(float(di))
                delta.append(dl)
    w, d = np.array(w), np.array(d)
    t, delta = np.array(t, dtype=float), np.array(delta)
    ker, kp, rlr, rrf = 3.0, 1.0, 0.7, 0.8
    os_lr = os_curve(t, ker, kp, rlr)
    pd_lr = paid_curve(t, ker, kp, rlr, rrf)

    if variant == "gaussian":
        mu = prem[w - 1] * np.where(delta == 0, os_lr, pd_lr)
        return {
            "len_data": len(w),
            "n_w": n_w,
            "w": w,
            "t": t,
            "delta": delta,
            "loss": mu + rng.normal(0.0, 50.0, len(w)),
            "premium": prem,
        }
    prev = paid_curve(np.maximum(t - 1.0, 0.0), ker, kp, rlr, rrf)
    mu = np.where(delta == 0, os_lr, pd_lr - prev)
    return {
        "len_data": len(w),
        "n_w": n_w,
        "n_d": int(d.max()),
        "w": w,
        "d": d,
        "t": t,
        "delta": delta,
        "y": np.maximum(mu * np.exp(rng.normal(0.0, 0.2, len(w))), 1e-9),
        "devfreq": 1.0,
    }


# -- fast: the three curve implementations agree -----------------------------


def test_curves_agree_across_all_three_implementations():
    """numpy (model.py, used by predict()), jnp (NumPyro port) and pytensor
    (PyMC port) must compute the same closed-form ODE solution. Three copies of
    one piece of algebra is where a transcription slip hides."""
    pytest.importorskip("numpyro")
    pytest.importorskip("pymc")
    import pytensor.tensor as pt

    from ibnr.gallery.bayesian.compartmental import model as entry
    from ibnr.gallery.bayesian.compartmental import model_numpyro as npy
    from ibnr.gallery.bayesian.compartmental import model_pymc as pmc

    t = np.array([0.0, 0.5, 1.0, 3.0, 10.0])
    ker, kp, rlr, rrf = 3.0, 1.0, 0.7, 0.8
    for name, args in (("os", (ker, kp, rlr)), ("paid", (ker, kp, rlr, rrf))):
        ref = getattr(entry, f"{name}_curve")(t, *args)
        jx = np.asarray(getattr(npy, f"{name}_curve")(t, *args))
        pyt = np.asarray(getattr(pmc, f"{name}_curve")(pt.as_tensor(t), *args).eval())
        np.testing.assert_allclose(jx, ref, rtol=1e-5, atol=1e-7, err_msg=f"{name}: jnp vs numpy")
        np.testing.assert_allclose(pyt, ref, rtol=1e-9, err_msg=f"{name}: pytensor vs numpy")


def test_paid_curve_is_exactly_zero_at_age_zero():
    """The incremental-paid differencing in the lognormal variant replaces
    Stan's ``t > devfreq`` branch with a clamp at 0, which is only valid
    because ``paid_curve(0) == 0`` exactly."""
    from ibnr.gallery.bayesian.compartmental.model import paid_curve

    assert paid_curve(0.0, 3.0, 1.0, 0.7, 0.8) == 0.0


# -- fast: the priors the ports must reproduce -------------------------------


def test_numpyro_sd_ay_is_positively_constrained():
    """THE silent-failure guard. An unconstrained ``sd_ay`` leaves the
    likelihood invariant under a sign flip of (sd, z, L[1,0]), making the
    posterior bimodal in rho and driving ``rho_ay`` - the model's headline
    reserving-cycle output - to ~0 with nothing raising."""
    pytest.importorskip("numpyro")
    import jax
    from numpyro import handlers
    from numpyro.distributions import constraints

    from ibnr.gallery.bayesian.compartmental import model_numpyro

    data = make_data(variant="gaussian")
    fixed = handlers.substitute(model_numpyro.gaussian_model, {"sd_ay": np.array([0.2, 0.1])})
    tr = handlers.trace(handlers.seed(fixed, jax.random.PRNGKey(0))).get_trace(data)
    support = tr["sd_ay"]["fn"].support
    base = getattr(support, "base_constraint", support)
    assert isinstance(base, constraints._GreaterThan | constraints._Positive), (
        f"sd_ay must be positively constrained, got {base}"
    )


def test_pymc_sd_scales_are_positive():
    """Same guard on the PyMC side: LKJCholeskyCov's sd_dist is a HalfStudentT,
    so every drawn scale is positive by construction."""
    pytest.importorskip("pymc")
    import pymc as pm

    from ibnr.gallery.bayesian.compartmental import model_pymc

    model = model_pymc.build_gaussian(make_data(variant="gaussian"))
    sds = np.asarray(pm.draw(model["ay_chol_stds"], draws=5000, random_seed=0))
    assert (sds > 0).all(), "LKJCholeskyCov scales must be strictly positive"


@pytest.mark.parametrize("ppl", ["numpyro", "pymc"])
def test_lkj1_gives_uniform_correlation(ppl):
    """LKJ(1) on a 2x2 correlation must be uniform on (-1, 1) - sd = 1/sqrt(3).

    Checked directly because Stan's ``lkj_corr_cholesky(1)`` adds nothing to
    the target: all of the geometry lives in the constraint transform, so a
    port can look right and sample the wrong prior.
    """
    if ppl == "numpyro":
        pytest.importorskip("numpyro")
        import jax
        import numpyro.distributions as dist

        lkj = dist.LKJCholesky(2, concentration=1.0)
        ls = np.asarray(lkj.sample(jax.random.PRNGKey(1), (60_000,)))
        rho = np.einsum("nij,nkj->nik", ls, ls)[:, 0, 1]
    else:
        pytest.importorskip("pymc")
        import pymc as pm

        with pm.Model():
            c = pm.LKJCorr("c", n=2, eta=1.0, return_matrix=False)
        rho = np.asarray(pm.draw(c, draws=60_000, random_seed=1)).ravel()
    assert abs(rho.mean()) < 0.02, f"{ppl}: rho should be centred at 0"
    # uniform(-1, 1) has sd 1/sqrt(3) = 0.5774
    assert abs(rho.std() - 1 / np.sqrt(3)) < 0.02, f"{ppl}: rho should be uniform on (-1, 1)"


def test_half_student_t_matches_the_analytic_quantiles():
    """Stan's ``<lower=0>`` + ``student_t(10, 0, s)`` is a half-Student-t. The
    NumPyro port spells it as a constrained site plus a factor (its
    TruncatedDistribution needs a Student-t CDF that is unavailable here), so
    the resulting density is worth checking against the closed form."""
    pytest.importorskip("numpyro")
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import MCMC, NUTS
    from scipy import stats

    from ibnr.gallery.bayesian.compartmental.model_numpyro import _half_student_t

    def model():
        _half_student_t("sd", 10.0, jnp.asarray([0.2]), (1,))

    mcmc = MCMC(NUTS(model), num_warmup=800, num_samples=12_000, num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0))
    s = np.asarray(mcmc.get_samples()["sd"]).ravel()
    assert (s > 0).all()
    ref = stats.t(10, 0, 0.2)
    for q in (0.5, 0.9):
        # half-t quantile at q is the base-t quantile at (1 + q) / 2
        assert abs(np.quantile(s, q) - ref.ppf((1 + q) / 2)) < 0.02 * ref.ppf(0.95)
    del numpyro  # imported for the side effect of the model above


# -- fast: graphs build with Stan's parameterization --------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_numpyro_graph_matches_stan(variant):
    """Trace once with the improper sites substituted: the sampled set must be
    Stan's parameter block, ``u_ay`` must be (2, n_w), and rho must be a real
    correlation."""
    pytest.importorskip("numpyro")
    import jax
    from numpyro import handlers

    from ibnr.gallery.bayesian.compartmental import model_numpyro

    data = make_data(variant=variant)
    model = getattr(model_numpyro, f"{variant}_model")
    subs = {"sd_ay": np.array([0.2, 0.1])}
    if variant == "lognormal":
        subs |= {
            "sd_dev": np.array([0.7, 0.5]),
            "sd_ker": np.array([0.3, 0.3]),
            "sd_kp": np.array([0.3, 0.3]),
        }
    tr = handlers.trace(
        handlers.seed(handlers.substitute(model, subs), jax.random.PRNGKey(0))
    ).get_trace(data)
    assert np.asarray(tr["u_ay"]["value"]).shape == (2, data["n_w"])
    assert -1.0 <= float(np.asarray(tr["rho_ay"]["value"])) <= 1.0
    mu = np.asarray(tr["mu"]["value"])
    assert mu.shape == (data["len_data"],) and np.isfinite(mu).all()
    if variant == "lognormal":
        assert (mu > 0).all(), "a lognormal mean must be strictly positive"


@pytest.mark.parametrize("variant", VARIANTS)
def test_pymc_graph_matches_stan(variant):
    """The PyMC graph carries the same deterministics under the same names.

    Note PyMC bundles Stan's separate ``sd_ay`` and ``L_ay`` into one
    ``LKJCholeskyCov`` variable - a naming difference, not a model difference
    (the priors are verified equal above), which is why parity compares
    ``rho_ay`` rather than the raw block.
    """
    pytest.importorskip("pymc")
    from ibnr.gallery.bayesian.compartmental import model_pymc

    data = make_data(variant=variant)
    model = model_pymc.build_model(data, variant=variant)
    names = set(model.named_vars)
    assert {"u_ay", "rho_ay", "mu", "sigma_os", "sigma_paid"} <= names
    assert [v.name for v in model.observed_RVs] == ["obs"]
    assert np.isfinite(model.compile_logp()(model.initial_point()))


def test_pymc_chol_is_diag_pre_multiply():
    """``LKJCholeskyCov``'s ``chol`` must equal Stan's
    ``diag_pre_multiply(sd_ay, L_ay)`` - that identity is what lets the
    non-centered line be transcribed verbatim."""
    pytest.importorskip("pymc")
    import pymc as pm

    with pm.Model():
        chol, corr, sds = pm.LKJCholeskyCov(
            "x",
            n=2,
            eta=1.0,
            sd_dist=pm.HalfStudentT.dist(nu=10, sigma=np.array([0.2, 0.1])),
            compute_corr=True,
        )
        c, co, sd = pm.draw([chol, corr, sds], draws=200, random_seed=0)
    # chol == diag(sd) @ L_corr  =>  chol @ chol.T == diag(sd) corr diag(sd)
    for i in range(0, 200, 50):
        expected = np.diag(sd[i]) @ co[i] @ np.diag(sd[i])
        np.testing.assert_allclose(c[i] @ c[i].T, expected, rtol=1e-8, atol=1e-10)


def test_entry_rejects_unknown_backend_and_variant():
    from ibnr.gallery.bayesian.compartmental.model import Compartmental

    with pytest.raises(ValueError, match="backend must be one of"):
        Compartmental().fit(None, backend="jags")
    with pytest.raises(ValueError, match="variant must be one of"):
        Compartmental().fit(None, variant="poisson")


# -- slow: NumPyro and PyMC sample the same posterior ------------------------


@pytest.mark.parity
@pytest.mark.slow
@pytest.mark.parametrize("variant", VARIANTS)
def test_numpyro_pymc_parity(variant):
    """The parity gate, on both variants. Compared parameters are the
    population-level scalars both variants expose under identical names; see
    ``COMPARTMENTAL_PARITY_VARS`` for why the per-accident-year RLR/RRF and the
    raw (sd_ay, L_ay) block are deliberately excluded."""
    pytest.importorskip("numpyro")
    pytest.importorskip("pymc")
    from ibnr.gallery.bayesian.compartmental import model_numpyro, model_pymc
    from ibnr.kernels.parity import compare_posteriors

    data = make_data(n_w=6, variant=variant, seed=7)
    idn = model_numpyro.sample(
        data, variant=variant, chains=2, iter_warmup=1000, iter_sampling=2500, seed=11
    )
    idp = model_pymc.sample(
        data, variant=variant, chains=2, iter_warmup=1000, iter_sampling=2500, seed=11
    )
    report = compare_posteriors(
        {"numpyro": idn, "pymc": idp}, reference="numpyro", var_names=COMPARTMENTAL_PARITY_VARS
    )
    assert report.passed, f"{variant} parity failed:\n{report.failures()}"
