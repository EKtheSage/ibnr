"""``kernels/densities.py``: the catalogue, and the change of measure.

The measure tests are the point of this file. Ethan's milestone-6 call is a
single global ELPD ranking, so every entry's density has to be carried to one
scale, and a wrong carry is invisible to any comparison of values: the result is
still smooth, still unimodal, still ranks points identically, and is simply not
a density. Normalization is the only thing that sees it, so every measure is
integrated to 1 here - and ``test_normalization_catches_a_missing_jacobian``
proves those integrals fail when the carry is removed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from scipy.special import gammaln

from ibnr.kernels.densities import (
    MEASURES,
    check_normalization,
    lognormal_lpdf,
    normal_lpdf,
    odp_lpdf,
    to_amount_scale,
)

# -- the catalogue ------------------------------------------------------------


def test_normal_matches_scipy():
    y = np.linspace(-4.0, 6.0, 41)
    assert np.allclose(normal_lpdf(y, 1.5, 2.0), stats.norm.logpdf(y, 1.5, 2.0))


def test_lognormal_matches_scipy_and_is_already_on_the_amount_scale():
    """``lognormal_lpdf`` carries its own ``-log y``, so it needs no conversion.

    The distinction against ``normal_lpdf(log y, ...)`` is exactly the bug this
    module exists to prevent, so it is asserted rather than assumed.
    """
    y = np.linspace(0.05, 20.0, 60)
    mu_log, sigma = 0.7, 0.5
    assert np.allclose(
        lognormal_lpdf(y, mu_log, sigma), stats.lognorm.logpdf(y, sigma, scale=np.exp(mu_log))
    )
    assert np.allclose(
        lognormal_lpdf(y, mu_log, sigma), normal_lpdf(np.log(y), mu_log, sigma) - np.log(y)
    )
    check_normalization(lambda v: lognormal_lpdf(v, mu_log, sigma), lo=1e-9, hi=200.0)


def test_odp_is_the_scaled_poisson_identity():
    """Milestone 5 pinned the port against ``Poisson(mu/phi).log_prob(x/phi)``.
    That identity is what makes the quasi-likelihood proper; it is pinned here
    too so the numpy copy cannot drift from the Stan/PPL ones."""
    phi, mu = 2.5, 40.0
    x = np.arange(0, 60) * phi
    assert np.allclose(odp_lpdf(x, mu, phi), stats.poisson.logpmf(x / phi, mu / phi))


def test_odp_keeps_the_parameter_free_lgamma_term():
    """Dropping ``-lgamma(x/phi + 1)`` cannot move a posterior, which is exactly
    why it is easy to drop - and it shifts every log_lik by a constant, breaking
    ELPD comparability with Stan, which does not drop it either."""
    phi, mu = 2.0, 30.0
    x = np.array([0.0, 4.0, 20.0, 50.0])
    without = (x / phi) * np.log(mu / phi) - mu / phi

    assert np.allclose(odp_lpdf(x, mu, phi), without - gammaln(x / phi + 1.0))
    # and the difference is real, not a rounding-level distinction
    assert np.max(np.abs(odp_lpdf(x, mu, phi) - without)) > 1.0
    # dropping it is exactly a per-observation shift, so it survives any
    # comparison of shapes and only shows up in an absolute ELPD
    assert not np.allclose(odp_lpdf(x, mu, phi), without)


@pytest.mark.parametrize("bad", [{"sigma": 0.0}, {"sigma": -1.0}])
def test_normal_rejects_nonpositive_sigma(bad):
    with pytest.raises(ValueError, match="sigma must be positive"):
        normal_lpdf(1.0, 0.0, bad["sigma"])


def test_odp_rejects_nonpositive_mu():
    with pytest.raises(ValueError, match="positive mu"):
        odp_lpdf(10.0, 0.0, 2.0)


# -- the change of measure ----------------------------------------------------


def test_amount_measure_is_a_no_op():
    lp = np.array([-3.0, -2.5, -9.0])
    assert np.allclose(to_amount_scale(lp, measure="amount"), lp)


def test_log_amount_carry_normalizes_on_the_amount_scale():
    """meyers_ccl / meyers_csr: ``normal_lpdf(log C | mu, sig)`` is a density on
    ``log C``. Carried by ``- log C`` it must integrate to 1 **in C**."""
    mu, sigma = 8.0, 0.35

    def on_amount(c):
        return to_amount_scale(normal_lpdf(np.log(c), mu, sigma), measure="log_amount", value=c)

    check_normalization(on_amount, lo=1e-6, hi=5.0e5)
    # and it must equal the lognormal, which is the same statement said twice
    grid = np.linspace(100.0, 2.0e4, 50)
    assert np.allclose(on_amount(grid), lognormal_lpdf(grid, mu, sigma))


def test_loss_ratio_carry_normalizes_on_the_amount_scale():
    """compartmental (lognormal): the density is of a loss RATIO, so carrying it
    to amounts is ``- log premium``."""
    premium, mu_log, sigma = 1000.0, -0.4, 0.3

    def on_amount(y):
        ratio = y / premium
        return to_amount_scale(
            lognormal_lpdf(ratio, mu_log, sigma), measure="loss_ratio", premium=premium
        )

    check_normalization(on_amount, lo=1e-6, hi=5.0e4)


def test_odp_lattice_carry_normalizes_as_a_spacing_weighted_sum():
    """england_verrall_odp / clark: a probability MASS on spacing ``phi``.

    After ``- log phi`` the values are a density, so the mass check is
    ``sum(exp(.) * phi)`` - which is the same as summing the original pmf, and
    that is the sense in which the conversion is a convention rather than a
    theorem.
    """
    phi, mu = 3.0, 90.0

    def on_amount(x):
        return to_amount_scale(odp_lpdf(x, mu, phi), measure="odp_lattice", phi=phi)

    check_normalization(on_amount, lo=0.0, hi=phi * 400, lattice=phi)
    # the raw pmf already sums to 1; the carry is exactly the spacing division
    x = np.arange(0, 400) * phi
    assert np.isclose(np.sum(np.exp(odp_lpdf(x, mu, phi))), 1.0, atol=1e-6)
    assert np.allclose(on_amount(x), odp_lpdf(x, mu, phi) - np.log(phi))


def test_normalization_catches_a_missing_jacobian():
    """The test that makes every other measure test meaningful.

    Without the carry the numbers are still finite, smooth and monotone in the
    right places - and integrate to something that is not 1. If this assertion
    ever stops holding, `check_normalization` has stopped being a check.
    """
    mu, sigma = 8.0, 0.35

    def unconverted(c):  # the bug: a log-scale density used as an amount density
        return normal_lpdf(np.log(c), mu, sigma)

    with pytest.raises(AssertionError, match="integrates to"):
        check_normalization(unconverted, lo=1e-6, hi=5.0e5)

    def sign_flipped(c):  # the other bug: carried the wrong way
        return normal_lpdf(np.log(c), mu, sigma) + np.log(c)

    with pytest.raises(AssertionError, match="integrates to"):
        check_normalization(sign_flipped, lo=1e-6, hi=5.0e5)


def test_increment_and_cumulative_are_the_same_measure():
    """``X = C - C_prev`` with ``C_prev`` on the training diagonal, so it is data
    and the Jacobian is 1. This is what lets an entry modelling cumulative loss
    and one modelling increments be summed into a single ELPD at all."""
    mu, sigma, c_prev = 8.0, 0.35, 2400.0
    c = np.linspace(2500.0, 9000.0, 40)

    as_cumulative = to_amount_scale(
        normal_lpdf(np.log(c), mu, sigma), measure="log_amount", value=c
    )

    def on_increment(x):
        cum = x + c_prev
        return to_amount_scale(normal_lpdf(np.log(cum), mu, sigma), measure="log_amount", value=cum)

    assert np.allclose(on_increment(c - c_prev), as_cumulative)
    # the support in increment space is (-c_prev, inf), because the density is a
    # lognormal in C over (0, inf) and the shift is exactly c_prev. Starting the
    # integral at 0 would drop the mass where C < C_prev and report ~0.73 - a
    # bound error, not a Jacobian error, and worth not confusing for one.
    check_normalization(on_increment, lo=-c_prev + 1e-6, hi=5.0e5 - c_prev)


# -- the covariate contract ---------------------------------------------------


def test_every_catalogued_measure_is_covered_by_a_normalization_test():
    """A measure added without a normalization test is the failure mode this
    module cannot otherwise detect, so the catalogue is pinned."""
    assert set(MEASURES) == {"amount", "log_amount", "loss_ratio", "odp_lattice"}


def test_missing_covariate_is_an_error():
    with pytest.raises(ValueError, match="needs 'value'"):
        to_amount_scale([-1.0], measure="log_amount")
    with pytest.raises(ValueError, match="needs 'phi'"):
        to_amount_scale([-1.0], measure="odp_lattice")


def test_an_unused_covariate_is_refused_rather_than_ignored():
    """A covariate accepted and ignored is a wire that looks connected and is
    not - and here it would leave the density silently unconverted while still
    returning plausible numbers."""
    with pytest.raises(ValueError, match="does not use 'phi'"):
        to_amount_scale([-1.0], measure="log_amount", value=100.0, phi=2.0)
    with pytest.raises(ValueError, match="needs no covariate"):
        to_amount_scale([-1.0], measure="amount", premium=1000.0)


def test_unknown_measure_names_the_catalogue():
    with pytest.raises(ValueError, match="measure must be one of"):
        to_amount_scale([-1.0], measure="log_loss_ratio")


def test_nonpositive_covariate_is_an_error_not_a_nan():
    """A zero premium or a zero loss would give ``-inf``/NaN silently; on the
    mart both occur (316k zero cells), so it has to be an error at the door."""
    with pytest.raises(ValueError, match="must be positive"):
        to_amount_scale([-1.0], measure="loss_ratio", premium=0.0)
