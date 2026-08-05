"""The exposure the pooled NN mixin scores with is the CONTRACT's, not the
caller's. Skips without torch.

``gallery/nn/_heldout.py::PooledMDNHeldout`` is shared by every single-line NN
entry, and both of its native hooks turn a loss RATIO into an amount or back:
the density standardizes ``increment / premium`` with the per-dev statistics
``_scheme.norm_stats`` estimated over exactly that ratio, and the draws scale a
sampled ratio up by the same exposure. Both read ``cells.premium`` - whatever
the caller happened to attach to the holdout frame - where the fit was
conditioned on ``nn_data``'s per-origin premium. Perturbing only that column
therefore moved the reported density AND the forecast, silently, with entirely
plausible numbers: measured pre-fix on this fixture, a x1.5 premium took
``nn_transformer``'s mean held-out log density from -3.78 to -28.70 nats/cell
and multiplied every incremental draw by exactly 1.5 - all four entries, no
error on either path.

The rule the rest of the gallery already follows (``guszcza_growth_curve/
scorer.py`` and ``compartmental/scorer.py``, which state it): the ratio math
uses the CONTRACT's premium and the caller's is VERIFIED against it. Verified,
not ignored - ``ScoresHeldout.log_lik_at``'s measure carry legitimately divides
by the CELLS' premium, so the two sources must agree or the carried density
stops integrating to 1, a wrong Jacobian that nothing downstream can see.

Both hooks get their own test because they are two independent reads of the
same wrong number: an entry can be ELPD-eligible, CRPS-eligible or both, and
half a fix would leave whichever axis was not looked at.

Parametrized over the entries the REGISTRY says use the mixin, with the case
table asserted to BE that set: an entry cloned from one of these cannot ship
without a row here (the lesson of
``test_fit_atomicity.py::test_every_registered_entry_is_covered``).
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ibnr import gallery  # noqa: E402
from ibnr.gallery.nn._heldout import PooledMDNHeldout  # noqa: E402
from ibnr.gallery.nn.deeptriangle.config import DeepTriangleConfig  # noqa: E402
from ibnr.gallery.nn.mdn.config import MDNConfig  # noqa: E402
from ibnr.gallery.nn.resnet.config import ResNetConfig  # noqa: E402
from ibnr.gallery.nn.transformer.config import TransformerConfig  # noqa: E402
from ibnr.kernels.holdout import next_diagonal  # noqa: E402

from .conftest import make_multiline_triangle  # noqa: E402

START = 2000
AS_OF = "2005-12-31"  # diagonal 6 of a 6x6 square starting in 2000
SEG0 = {"company_code": "0001", "line_of_business": "lob_0"}
#: dev steps whose per-dev normalizer is pinned at this as_of; a pinned cell is
#: refused before any premium is read, so the density's panel excludes it
PINNED_DEVS = (6,)

#: everything the four entries' configs share, deliberately under-powered: two
#: ensemble members (the density's draw axis needs >= 2) and three epochs.
#: Nothing here asserts predictive quality - only which premium is used.
COMMON = dict(
    dropout=0.0,
    n_components=2,
    lob_embedding_dim=4,
    batch_size=8,
    max_epochs=3,
    patience=5,
    ensemble_size=2,
    n_draws=50,  # rollout draws (predict)
    heldout_n_draws=50,  # held-out diagonal draws (predict_at); 10,000 by default
)

#: entry name -> its tiny fit config. Add a row when an entry joins the mixin;
#: ``test_every_mixin_entry_is_covered`` is what forces that.
CONFIGS = {
    "nn_transformer": TransformerConfig(d_model=16, n_layers=1, n_heads=2, ffn_dim=32, **COMMON),
    "mdn": MDNConfig(hidden_dim=32, n_layers=1, embedding_dim=4, **COMMON),
    "resnet": ResNetConfig(channels=16, n_blocks=2, n_groups=4, **COMMON),
    "deeptriangle": DeepTriangleConfig(hidden_dim=16, company_embedding_dim=4, **COMMON),
}

#: how far the perturbed premium is off. Large enough that a pre-fix run
#: produces obviously different numbers rather than a rounding difference -
#: the point being that it produced numbers at all.
WRONG_FACTOR = 1.5


def _matrices() -> np.ndarray:
    """(2, 6, 6) cumulative squares with a decaying incremental pattern - the
    ``test_nn_heldout.py`` fixture, shared by the pooled fit triangle and the
    per-cohort cells triangle so the cells describe the data trained on."""
    rng = np.random.default_rng(0)
    n_w = n_d = 6
    dev_level = np.exp(np.linspace(-0.8, -3.0, n_d))
    incr = 1000.0 * dev_level[None, None, :] * rng.lognormal(0.0, 0.1, size=(2, n_w, n_d))
    return np.cumsum(incr, axis=2)


@pytest.fixture(scope="module", params=sorted(CONFIGS))
def fitted(request):
    """One TINY pooled fit per mixin entry, plus lob_0's next-diagonal cells.

    duckdb only, unlike the per-entry held-out files: the guard under test is
    plain numpy over a ``CellIndex``, so a second backend would re-run four
    fits to re-check an engine that never sees the premium comparison.
    """
    name = request.param
    cum = _matrices()
    prem = np.full(6, 1000.0)
    lobs = {"lob_0": cum[0], "lob_1": cum[1]}
    pooled = make_multiline_triangle(
        "duckdb", lobs, premium_by_lob=dict.fromkeys(lobs, prem), start_year=START
    )
    # feature_fields=() uniformly: deeptriangle defaults to an auxiliary
    # reported channel the other three do not have, and the premium a cell is
    # scored with is orthogonal to how many channels the network reads.
    entry = gallery.get(name)().fit(
        pooled,
        loss_field="paid_loss",
        feature_fields=(),
        as_of=AS_OF,
        config=CONFIGS[name],
        seed=0,
    )
    one = make_multiline_triangle(
        "duckdb", {"lob_0": cum[0]}, premium_by_lob={"lob_0": prem}, start_year=START
    )
    cells = next_diagonal(one, as_of=AS_OF, fields="paid_loss", premium_field="earned_premium")
    return SimpleNamespace(name=name, entry=entry, cells=cells, live=_unpinned(cells))


def _unpinned(cells):
    """The next-diagonal cells at unpinned dev steps - the density's panel."""
    mask = cells.frame["dev_lag"].isin([12 * d for d in PINNED_DEVS])
    return replace(cells, frame=cells.frame[~mask].reset_index(drop=True))


def _rescaled_premium(cells, factor: float):
    """The same cells with ONLY the premium column moved - the exact edit the
    reviewer made, and the one that must stop changing any answer."""
    frame = cells.frame.copy()
    frame["premium"] = frame["premium"] * factor
    return replace(cells, frame=frame)


def _without_premium(cells):
    """The same cells with no premium column at all, so ``index_into`` falls
    back to the fitted contract's per-origin premium."""
    return replace(cells, frame=cells.frame.drop(columns=["premium"]), premium_field=None)


def test_fixture_carries_a_premium_the_test_can_move(fitted):
    """The cells actually carry their own premium column, and both panels are
    non-empty. Without this, every refusal below could pass vacuously on cells
    that never had a caller-supplied premium to disagree with."""
    assert "premium" in fitted.cells.frame.columns
    assert fitted.cells.n_cells == 5
    assert fitted.live.n_cells == 4
    np.testing.assert_allclose(fitted.cells.frame["premium"].to_numpy(), 1000.0)


def test_log_lik_refuses_a_premium_that_disagrees_with_the_contract(fitted):
    """The density path (``_heldout_log_lik``).

    Pre-fix this raised nothing: the standardized ``z`` divided by the cells'
    premium, so a x1.5 column returned a finite, smooth, plausible density
    built on a scale the per-dev normalizer never saw - and the base class's
    ``- log premium`` carry then divided by that same wrong number, so not even
    the normalization check could see it.
    """
    entry = fitted.entry
    honest = entry.log_lik_at(fitted.live, field="paid_loss")
    assert np.isfinite(honest).all()

    with pytest.raises(ValueError) as excinfo:
        entry.log_lik_at(_rescaled_premium(fitted.live, WRONG_FACTOR), field="paid_loss")
    message = str(excinfo.value)
    # the message must name BOTH sources, or a reader cannot tell which two
    # numbers disagree and which of them the entry acted on
    assert "cell" in message and "contract" in message


def test_predict_at_refuses_a_premium_that_disagrees_with_the_contract(fitted):
    """The draws path (``_heldout_draws``), the same wrong read a second time.

    Pre-fix a x1.5 premium scaled every incremental draw by 1.5 - the forecast
    changed by 50% with no error - and the base class then added the honest
    training anchor on top, so the result stayed in a believable range.
    """
    entry = fitted.entry
    honest = entry.predict_at(fitted.cells, field="paid_loss", seed=7)
    assert honest.shape == (CONFIGS[fitted.name].heldout_n_draws, fitted.cells.n_cells)
    assert np.isfinite(honest).all()

    with pytest.raises(ValueError) as excinfo:
        entry.predict_at(_rescaled_premium(fitted.cells, WRONG_FACTOR), field="paid_loss", seed=7)
    message = str(excinfo.value)
    assert "cell" in message and "contract" in message


def test_the_honest_path_does_not_read_the_cells_premium_at_all(fitted):
    """Drop the premium column entirely and both answers must be BIT-identical.

    The positive half of the fix: with the caller's number gone, ``index_into``
    falls back to the contract's, so identical output is what proves the ratio
    math was already using the contract's number rather than agreeing by
    coincidence. It is also the guard on the guard - a fix that refused a
    missing premium instead of a disagreeing one would fail right here, and
    that path is live (``next_diagonal`` attaches no premium unless asked).
    """
    entry = fitted.entry
    np.testing.assert_array_equal(
        entry.log_lik_at(_without_premium(fitted.live), field="paid_loss"),
        entry.log_lik_at(fitted.live, field="paid_loss"),
    )
    np.testing.assert_array_equal(
        entry.predict_at(_without_premium(fitted.cells), field="paid_loss", seed=7),
        entry.predict_at(fitted.cells, field="paid_loss", seed=7),
    )


def test_every_mixin_entry_is_covered():
    """``CONFIGS`` must BE the set of registered entries using the mixin.

    A hand-written list is exactly how ``deeptriangle``/``mdn``/``resnet``
    shipped a bug that had already been fixed elsewhere: they were cloned from
    a template and no list mentioned them. Deriving the set from the registry
    means a new mixin entry fails here, naming itself, at the one moment its
    author is looking at this concern.
    """
    using_mixin = {
        name for name in gallery.list() if issubclass(gallery.get(name), PooledMDNHeldout)
    }
    assert set(CONFIGS) == using_mixin, (
        "entries using PooledMDNHeldout without a row in CONFIGS: "
        f"{sorted(using_mixin - set(CONFIGS))}; rows naming entries that no longer use it: "
        f"{sorted(set(CONFIGS) - using_mixin)}. Both hooks of the mixin divide by the "
        "fitted contract's premium and must refuse a cell whose own premium disagrees."
    )
