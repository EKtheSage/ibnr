"""Pieces of the Guszcza ports that are not PPL-specific.

``model_numpyro`` and ``model_pymc`` each write their own ``growth_curve``,
because that one IS PPL-specific (``jax.numpy`` against ``pytensor.tensor``) and
because a reader should be able to hold each port beside ``model.stan`` and see
the same algebra. Everything here is plain numpy or a constant, so duplicating
it would only create two things to keep in step - the failure mode where a fix
lands in one port and silently misses the other.

Still inside the model directory, so ``gallery.scaffold()`` stays a directory
copy (design decision 6).
"""

from __future__ import annotations

import numpy as np

#: Stan's integer curve codes (``model.stan``'s ``curve``), mirrored so neither
#: port can silently fit the other curve.
LOGLOGISTIC, WEIBULL = 1, 2

#: ``model.stan``'s own control list, from the magesblog post's ``brm(control=)``
#: - ``adapt_delta = 0.999`` and this tree depth. Held explicitly rather than
#: left to NumPyro's and PyMC's default of 10: at that step size this posterior
#: needs trajectories longer than 2**10 leapfrogs, so a port running the default
#: would differ from the Stan reference by a SAMPLER setting rather than by the
#: model - exactly what parity is meant to hold constant.
MAX_TREE_DEPTH = 15


def check_data(t: np.ndarray, y: np.ndarray) -> None:
    """Refuse the two data-side violations that would otherwise fail silently.

    ``t > 0`` is what lets both ports' ``growth_curve`` omit the zero-age branch
    the Clark entries need: ``model.stan`` omits it too, because ``t = d *
    dev_grain_months / 12`` with ``d >= 1``. Stan's ``data`` block nonetheless
    declares ``vector<lower=0>[len_data] t``, which is LOOSER than that
    invariant, and at ``t = 0`` the loglogistic's ``(theta/t)^omega`` is ``inf``
    with a NaN derivative - the value survives and the gradient does not, so the
    failure would surface as NUTS dying on the first leapfrog with no message
    rather than as a bad number.

    ``y > 0`` is the lognormal's support, guaranteed upstream by the shared
    contract's non-positive-loss refusal.

    Both arguments are data, so this runs once when the model is traced or the
    graph is built, on numpy arrays, and never enters the computation graph.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if not np.all(t > 0):
        raise ValueError(
            f"{int(np.sum(t <= 0))} of {t.size} development ages are non-positive; "
            "growth_curve mirrors model.stan, which has no zero-age branch, and its "
            "gradient is NaN at t = 0"
        )
    if not np.all(y > 0):
        raise ValueError(
            f"{int(np.sum(y <= 0))} of {y.size} loss ratios are non-positive, "
            "outside the lognormal's support"
        )
