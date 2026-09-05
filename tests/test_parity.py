"""What ``kernels.parity`` refuses, and what it is allowed to call agreement.

The six ``test_parity_<model>.py`` files fit two real backends and assert
``report.passed``. They cannot see the two ways a comparison used to be
lenient, because both of them make a *healthy* pair of fits look exactly the
same as before:

* a requested variable that one side does not carry was quietly dropped from
  the comparison, so the report was an ``.all()`` over the rows that happened
  to survive, and
* a marginal that is an exact point mass has no Monte Carlo error, so the old
  code divided by zero-or-NaN and wrote down ``z = 0``, the best possible
  score, whatever the two constants were.

These tests are fast and use hand-built posteriors, because both behaviours
need input a correct port never produces. The control is
``test_equal_point_masses_pass``: CCL, CSR and ODP all pin an identifiability
anchor (``alpha[1] = 0``, ``beta[n_d] = 0``) as a literal constant in all three
backends, so two equal point masses must keep passing at z = 0.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("arviz")

import arviz as az  # noqa: E402

from ibnr.kernels import compare_posteriors  # noqa: E402


def _idata(**arrays):
    """An ``InferenceData`` whose posterior carries exactly these arrays."""
    return az.from_dict(posterior={k: np.asarray(v) for k, v in arrays.items()})


def _normal(rng, loc=0.0, scale=1.0, shape=(4, 500)):
    return rng.normal(loc, scale, size=shape)


def _compare(ref, port, var_names, **kw):
    return compare_posteriors(
        {"stan": ref, "numpyro": port}, reference="stan", var_names=var_names, **kw
    )


def test_requested_name_missing_from_the_port_is_refused_by_name():
    rng = np.random.default_rng(0)
    ref = _idata(alpha=_normal(rng), beta=_normal(rng, 2.0))
    port = _idata(alpha=_normal(rng))
    with pytest.raises(ValueError, match=r"beta"):
        _compare(ref, port, ("alpha", "beta"))


def test_requested_name_missing_from_the_reference_is_refused_by_name():
    rng = np.random.default_rng(1)
    ref = _idata(alpha=_normal(rng))
    port = _idata(alpha=_normal(rng), beta=_normal(rng, 5.0))
    with pytest.raises(ValueError, match=r"beta"):
        _compare(ref, port, ("alpha", "beta"))


def test_requested_name_missing_everywhere_is_refused_by_name():
    """The old failure here was xarray's, and it named a dimension, not the
    variable the caller asked for."""
    rng = np.random.default_rng(2)
    ref = _idata(alpha=_normal(rng))
    port = _idata(alpha=_normal(rng))
    with pytest.raises(ValueError, match=r"gamma"):
        _compare(ref, port, ("alpha", "gamma"))


def test_element_shape_mismatch_under_a_shared_name_is_refused_by_name():
    """A port whose vector is shorter than the reference's is a different
    model, not a comparison over the elements they share."""
    rng = np.random.default_rng(3)
    ref = _idata(alpha=_normal(rng, shape=(4, 500, 3)))
    port = _idata(alpha=_normal(rng, shape=(4, 500, 2)))
    with pytest.raises(ValueError, match=r"element shape.*alpha|alpha.*element shape"):
        _compare(ref, port, ("alpha",))


@pytest.mark.parametrize("side", ["stan", "numpyro"])
def test_non_finite_draws_are_refused_by_name(side):
    """One NaN draw makes every arviz summary field NaN, which the old code
    read as a zero discrepancy. Either side can carry it, so both are checked -
    the reference is not exempt from any of these refusals."""
    rng = np.random.default_rng(7)
    bad = _normal(rng, 50.0)
    bad[0, 0] = np.nan
    good = _normal(rng)
    ref = _idata(alpha=bad if side == "stan" else good)
    port = _idata(alpha=bad if side == "numpyro" else good)
    with pytest.raises(ValueError, match=rf"{side}.*non-finite draw.*alpha"):
        _compare(ref, port, ("alpha",))


def test_coordinate_labels_that_differ_between_backends_are_refused():
    """Same name, same element shape, different labels: one side attached
    coordinates. No port does that today, so this is the backstop behind the
    name and shape checks, and it must refuse rather than compare whatever the
    two label sets happen to share, which here is nothing."""
    rng = np.random.default_rng(11)
    ref = _idata(alpha=_normal(rng, shape=(4, 500, 3)))
    port = az.from_dict(
        posterior={"alpha": _normal(rng, shape=(4, 500, 3))},
        coords={"lag": ["a", "b", "c"]},
        dims={"alpha": ["lag"]},
    )
    with pytest.raises(ValueError, match=r"different elements"):
        _compare(ref, port, ("alpha",))


def test_empty_var_names_is_refused_by_name():
    rng = np.random.default_rng(8)
    ref = _idata(alpha=_normal(rng))
    port = _idata(alpha=_normal(rng))
    with pytest.raises(ValueError, match=r"var_names"):
        _compare(ref, port, ())


@pytest.mark.parametrize("with_ks", [True, False])
def test_unequal_point_masses_are_a_parity_failure(with_ks):
    """Two constants that differ disagree completely, so the row fails and the
    report still comes back for the caller to write down."""
    rng = np.random.default_rng(4)
    ref = _idata(alpha=np.full((4, 500), 1.0), free=_normal(rng))
    port = _idata(alpha=np.full((4, 500), 100.0), free=_normal(rng))
    report = _compare(ref, port, ("alpha", "free"), with_ks=with_ks)
    assert not report.passed, report.table
    assert "alpha" in set(report.failures()["param"]), report.table
    assert report.summary().loc["numpyro", "n_params"] == 2, report.table


@pytest.mark.parametrize("with_ks", [True, False])
def test_equal_point_masses_pass(with_ks):
    """CONTROL. The identifiability anchors are exact, equal constants in all
    three backends of CCL, CSR and ODP, so they must keep passing at z = 0. A
    fix that refused every zero Monte Carlo error would fail every published
    parity run here."""
    rng = np.random.default_rng(5)
    ref = _idata(alpha=np.zeros((4, 500)), free=_normal(rng))
    port = _idata(alpha=np.zeros((4, 500)), free=_normal(rng))
    report = _compare(ref, port, ("alpha", "free"), with_ks=with_ks)
    assert report.passed, report.table
    row = report.table.set_index("param").loc["alpha"]
    assert row["z_mean"] == 0.0 and row["z_sd"] == 0.0


@pytest.mark.parametrize("with_ks", [True, False])
def test_a_pin_the_port_left_free_is_a_parity_failure(with_ks):
    """The reference pins alpha at 0; the port samples it around 0 with spread
    0.3. Same mean, different spread. A point mass has zero Monte Carlo error
    for its spread too, so the combined error is the port's alone and the
    spread z-score is large. The old code got NaN from ``hypot(nan, x)`` and
    wrote 0.0."""
    rng = np.random.default_rng(6)
    ref = _idata(alpha=np.zeros((4, 500)), free=_normal(rng))
    port = _idata(alpha=_normal(rng, 0.0, 0.3), free=_normal(rng))
    report = _compare(ref, port, ("alpha", "free"), with_ks=with_ks)
    row = report.table.set_index("param").loc["alpha"]
    assert not report.passed, report.table
    assert abs(row["z_sd"]) > report.z_tol, report.table


def test_undefined_mcse_on_a_varying_marginal_is_refused():
    """Three draws per chain is below what arviz needs to estimate an effective
    sample size, so both Monte Carlo errors come back NaN on a marginal that is
    not constant. There is nothing to certify from, so refuse rather than score
    it a perfect zero."""
    rng = np.random.default_rng(9)
    ref = _idata(alpha=_normal(rng, shape=(2, 3)))
    port = _idata(alpha=_normal(rng, 50.0, shape=(2, 3)))
    with pytest.raises(ValueError, match=r"alpha"):
        _compare(ref, port, ("alpha",))


def test_every_requested_element_is_a_row():
    """A three-element vector and a scalar are four rows, so a narrowing
    cannot return quietly."""
    rng = np.random.default_rng(10)
    ref = _idata(alpha=_normal(rng, shape=(4, 500, 3)), beta=_normal(rng))
    port = _idata(alpha=_normal(rng, shape=(4, 500, 3)), beta=_normal(rng))
    report = _compare(ref, port, ("alpha", "beta"))
    assert list(report.table["param"]) == ["alpha[0]", "alpha[1]", "alpha[2]", "beta"]
    assert report.summary().loc["numpyro", "n_params"] == 4
