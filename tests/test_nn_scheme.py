"""gallery.nn._scheme: the shared NN training scheme. Pure numpy - no torch.

``_scheme.py`` is deliberately torch-free (every NN entry imports it, and
``ibnr.gallery`` must import without the ``[nn]`` extra), so its own tests run
in the core CI leg rather than behind ``importorskip``. The leakage properties
of :func:`splits` and the pinning rule of :func:`norm_stats` are pinned in
``test_nn_transformer.py``, through the entry that first grew them; what lives
here is the per-channel form of ``norm_stats`` - the one that lets a feature
channel's statistics come from the cells where THAT channel has a value.

The bit-identity test is the load-bearing one. The per-channel form has to be
reachable without moving the number a target-only mask produces, because that
number is what every published NN result was standardized against.
"""

from __future__ import annotations

import numpy as np
import pytest

from ibnr.gallery.nn._scheme import norm_stats


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, obs, x_obs): 4 cohorts, 2 channels, 1 origin, 2 devs.

    Channel 1 is padding at the last cohort - the contract's zero for "no usable
    value here" - while the TARGET is observed at that cell. That is the whole
    disagreement between the two mask forms, and it is chosen so both means are
    round numbers a reader can check: 1.5 against 2.0 at dev 0, 3.75 against 5.0
    at dev 1.
    """
    x = np.zeros((4, 2, 1, 2))
    x[:, 0, 0, 0] = [10.0, 11.0, 12.0, 13.0]
    x[:, 0, 0, 1] = [20.0, 21.0, 22.0, 23.0]
    x[:, 1, 0, 0] = [1.0, 2.0, 3.0, 0.0]
    x[:, 1, 0, 1] = [4.0, 5.0, 6.0, 0.0]
    obs = np.ones((4, 1, 2), dtype=bool)
    x_obs = np.ones((4, 2, 1, 2), dtype=bool)
    x_obs[3, 1] = False
    return x, obs, x_obs


def test_per_channel_masks_use_each_channels_own_cells():
    """A feature channel's mean must come from ITS observed cells.

    Under the target-only mask the contract's padding zero is averaged in as
    though it were a feature value of zero, which drags the channel's mean toward
    zero and inflates its spread - a distortion that grows with how often the
    feature is missing where the target is not.
    """
    x, obs, x_obs = _grid()
    mean_t, std_t, pin_t = norm_stats(x, obs, obs)
    mean_c, std_c, pin_c = norm_stats(x, x_obs, x_obs)
    assert mean_t[1, 0] == pytest.approx(1.5)  # mean(1, 2, 3, 0) - the padding counted
    assert mean_c[1, 0] == pytest.approx(2.0)  # mean(1, 2, 3)
    assert std_c[1, 0] == pytest.approx(np.std([1.0, 2.0, 3.0]))
    assert std_t[1, 0] > std_c[1, 0]
    # the target channel reads identically either way - x_obs[:, 0] IS obs_mask
    np.testing.assert_array_equal(mean_t[0], mean_c[0])
    np.testing.assert_array_equal(std_t[0], std_c[0])
    assert not pin_t.any() and not pin_c.any()


def test_the_pinned_fallback_also_uses_the_channels_own_cells():
    """The pinned branch reads the SECOND mask, and it needs the same treatment.

    A pinned dev's mean is what an unstandardized prediction returns there, so
    letting the target's observedness choose the cells hands a feature channel a
    fallback averaged over cells where its own value is padding.
    """
    x, obs, x_obs = _grid()
    cells = obs.copy()
    cells[:, :, 1] = False  # no context at the deepest dev -> pinned
    cells_c = x_obs.copy()
    cells_c[:, :, :, 1] = False
    mean_t, std_t, pin_t = norm_stats(x, cells, obs)
    mean_c, std_c, pin_c = norm_stats(x, cells_c, x_obs)
    np.testing.assert_array_equal(pin_t, pin_c)
    assert pin_t[:, 1].all() and not pin_t[:, 0].any()
    np.testing.assert_allclose(std_t[:, 1], 1.0)
    assert mean_t[1, 1] == pytest.approx(3.75)  # mean(4, 5, 6, 0)
    assert mean_c[1, 1] == pytest.approx(5.0)  # mean(4, 5, 6)


def test_a_repeated_mask_reproduces_the_target_only_form_bit_for_bit():
    """Handing every channel the target's mask must be the old function exactly.

    Not ``allclose``: the entries standardize against these numbers and the
    published NN results were produced by the three-dimensional call, so the new
    argument shape has to be a pure widening. Compared as raw bytes.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(5, 3, 4, 4))
    cal = np.arange(4)[:, None] + np.arange(4)[None, :] + 1
    obs = np.broadcast_to(cal <= 4, (5, 4, 4))
    cells = obs & (cal <= 3)  # a validation diagonal held out of the context
    legacy = norm_stats(x, cells, obs)
    repeated = norm_stats(
        x, np.broadcast_to(cells[:, None], x.shape), np.broadcast_to(obs[:, None], x.shape)
    )
    for a, b in zip(legacy, repeated, strict=True):
        assert a.tobytes() == b.tobytes()
    # and the fixture is not vacuous: the deepest dev is pinned
    assert legacy[2][:, 3].all()


def test_a_mask_with_the_wrong_channel_count_is_refused():
    """Silently broadcasting or truncating a mismatched channel axis would
    standardize channels against each other's cells."""
    x = np.zeros((2, 3, 1, 1))
    bad = np.ones((2, 2, 1, 1), dtype=bool)
    with pytest.raises(ValueError, match="2 channels but x has 3"):
        norm_stats(x, bad, bad)


def test_a_mask_of_the_wrong_rank_is_refused():
    x = np.zeros((2, 3, 1, 1))
    flat = np.ones((2, 1), dtype=bool)
    with pytest.raises(ValueError, match="dimensions"):
        norm_stats(x, flat, flat)
