"""The context mask every NN entry hands its network during fit(). Skips
without torch.

Each NN entry's ``fit`` builds one line inside its ``train_loss`` closure:

    ctx = obs_t[idx] & (cal_t[None] <= cutoffs[:, None, None])

That single ``&`` is the entry-level no-leak boundary. Drop the right-hand
gate and the model conditions on cells it is being asked to predict AND on the
eval_date validation diagonal, so the training loss sees the future and the
early-stopping signal stops being held out - a fit that is silently worthless
while every number it produces still looks reasonable.

Nothing pinned that line before this file. The per-entry receptive-field /
attention no-leak tests pin the NETWORK invariant (a cell outside the context
mask contributes nothing to any output), which is a different claim: they
prove the network honors whatever mask it is given, not that the entry builds
the right one. Mutating the closure to ``ctx = obs_t[idx]`` left the whole
suite green.

So the test drives a real ``fit()`` through the PUBLIC entry point and records
what actually reached the network, via two spies:

* the entry's ``train_ensemble`` name is swapped for a wrapper that wraps the
  ``train_loss`` callback, capturing each batch's cohort ``idx`` and augmented
  ``cutoffs``. Taking the cutoffs here rather than off the forward signature is
  deliberate - not every entry's network is handed the cutoff (deeptriangle's
  is not), but every entry's ``train_loss`` receives it;
* the network method that receives the context mask is swapped for a wrapper
  that records the mask, tagged with the batch in flight.

Both halves of the training loop are checked, because the leak has two places
to hide: the augmented-cutoff context in ``train_loss``, and the fixed
validation-cutoff context in ``val_loss`` (passing ``obs_t`` there instead of
the context-eligible mask would let the validation diagonal condition on
itself). Verified by mutation - see ``CASES`` for how to add an entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ibnr.gallery.nn import _scheme  # noqa: E402
from ibnr.gallery.nn.transformer import model as tf_model  # noqa: E402
from ibnr.gallery.nn.transformer import network as tf_net  # noqa: E402
from ibnr.gallery.nn.transformer.config import TransformerConfig  # noqa: E402
from ibnr.gallery.nn.transformer_ml import model as ml_model  # noqa: E402
from ibnr.gallery.nn.transformer_ml import network as ml_net  # noqa: E402
from ibnr.gallery.nn.transformer_ml.config import TransformerMLConfig  # noqa: E402

from .conftest import make_multiline_triangle  # noqa: E402

START = 2000

# under-powered on purpose: one member, two epochs. Nothing here reads a
# prediction, only the masks the training loop builds.
TINY = dict(
    d_model=16,
    n_layers=1,
    n_heads=2,
    ffn_dim=32,
    dropout=0.0,
    n_components=2,
    batch_size=8,
    max_epochs=2,
    patience=5,
    ensemble_size=1,
    n_draws=10,
)


@dataclass(frozen=True)
class Case:
    """One NN entry's spy wiring.

    entry:       zero-arg constructor for the GalleryEntry.
    config:      the fit config (see TINY).
    module:      the entry module - where ``train_ensemble`` is name-bound, and
                 so where it must be monkeypatched.
    net_cls /
    net_method:  the network method that receives the context mask, whose
                 signature starts ``(self, x, context_mask, ...)``. Pick the
                 one place every head funnels through (``encode`` when the
                 entry has more than one head), or the mask gets recorded twice.
    """

    id: str
    entry: Any
    config: Any
    module: Any
    net_cls: type
    net_method: str


#: Add an entry here rather than copying this file. deeptriangle / mdn / resnet
#: (PRs #45 / #44 / #46) carry the identical ``train_loss`` closure and each
#: needs one row; their networks all take the mask as ``forward``'s second
#: argument.
CASES = [
    Case(
        id="nn_transformer",
        entry=tf_model.NNTransformer,
        config=TransformerConfig(lob_embedding_dim=4, **TINY),
        module=tf_model,
        net_cls=tf_net.TriangleTransformer,
        net_method="forward",
    ),
    Case(
        id="nn_transformer_ml",
        entry=ml_model.NNTransformerML,
        # both dependence heads run through encode(), so one case covers them
        config=TransformerMLConfig(dependence="ar", **TINY),
        module=ml_model,
        net_cls=ml_net.TriangleTransformerML,
        net_method="encode",
    ),
]


def synthetic_triangle(backend_name, n_w=6):
    """Two LOB full squares, one company - read as two cohorts by the
    single-line entry and as one two-line company by the multi-line one, so a
    single fixture serves both.

    A full square (not just the upper triangle) is what gives the test its
    teeth: with observed cells on every diagonal up to ``n_w + n_d - 1``, every
    augmented cutoff (drawn below the validation cutoff) leaves observed cells
    beyond it that the gate MUST exclude. On an upper triangle a late cutoff
    could exclude nothing and the assertion would hold vacuously.
    """
    rng = np.random.default_rng(0)
    n_d = n_w
    dev_level = np.exp(np.linspace(-0.8, -3.0, n_d))
    incr = 1000.0 * dev_level[None, None, :] * rng.lognormal(0.0, 0.1, size=(2, n_w, n_d))
    cum = np.cumsum(incr, axis=2)
    lobs = {f"lob_{k}": cum[k] for k in range(2)}
    prem = {f"lob_{k}": np.full(n_w, 1000.0) for k in range(2)}
    return make_multiline_triangle(backend_name, lobs, premium_by_lob=prem, start_year=START)


def record_fit(case: Case, monkeypatch, triangle):
    """Run a real ``fit()`` and return (entry, calls).

    Each call record is ``{"ctx": mask, "batch": {...} | None}``; ``batch`` is
    the ``(idx, cutoffs)`` of the training batch in flight, or None for a
    forward pass made outside ``train_loss`` - i.e. the validation pass.
    """
    calls: list[dict] = []
    live: dict[str, dict | None] = {"batch": None}
    real_train_ensemble = case.module.train_ensemble
    real_method = getattr(case.net_cls, case.net_method)

    def spy_train_ensemble(n_cohorts, *, train_loss, **kwargs):
        def wrapped(model, idx, cutoffs):
            live["batch"] = {
                "idx": idx.detach().cpu().numpy().copy(),
                "cutoffs": cutoffs.detach().cpu().numpy().copy(),
            }
            try:
                return train_loss(model, idx, cutoffs)
            finally:
                live["batch"] = None

        return real_train_ensemble(n_cohorts, train_loss=wrapped, **kwargs)

    def spy_method(self, x, context_mask, *rest, **kwargs):
        calls.append({"ctx": context_mask.detach().cpu().numpy().copy(), "batch": live["batch"]})
        return real_method(self, x, context_mask, *rest, **kwargs)

    monkeypatch.setattr(case.module, "train_ensemble", spy_train_ensemble)
    monkeypatch.setattr(case.net_cls, case.net_method, spy_method)
    entry = case.entry().fit(triangle, loss_field="paid_loss", config=case.config, seed=0)
    return entry, calls


def grids(entry, case):
    """(cal, obs, val_cutoff) from the entry's own fitted contract.

    ``obs`` is reshaped to (n_cohorts, S, W, D) so one set of assertions covers
    both layouts: S = 1 for a single-line entry, S = n_lines for the multi-line
    one, whose masks carry a line axis between cohort and origin. Squeezing to
    a common rank beats branching on the entry.
    """
    c = entry.contract_
    cal = np.asarray(c["cal_idx"])  # (W, D), 1-based calendar diagonal
    obs = np.asarray(c["obs_mask"])
    n_w, n_d = cal.shape
    obs = obs.reshape(obs.shape[0], -1, n_w, n_d)
    # the multi-line entry carves its split on "any line observed", which for a
    # single-line entry (S = 1) is the observation mask itself
    _, _, val_cutoff = _scheme.splits(obs.any(axis=1), cal, case.config.val_diagonals)
    return cal, obs, val_cutoff


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_training_context_is_gated_by_the_augmented_cutoff(case, backend_name, monkeypatch):
    """Every context mask the network sees during training is the observed
    cells AT OR BEFORE that batch's augmented cutoff - nothing later.

    Four assertions, in the order they matter:

    1. no cell past the cutoff is conditioned on. This is the leak the
       ``& (cal <= cutoff)`` gate exists to prevent: those cells are the
       prediction targets;
    2. no validation-diagonal cell enters any training context. Implied by (1)
       given cutoffs are drawn below the validation cutoff, asserted separately
       because it is the property that makes early stopping meaningful, and it
       would survive a broken cutoff draw undetected otherwise;
    3. the gate actually excluded something on every batch, so (1) cannot pass
       vacuously (see ``synthetic_triangle``);
    4. everything observed at or before the cutoff IS in context - the
       completeness half. A gate that excluded too much would leak nothing and
       train on nothing.
    """
    entry, calls = record_fit(case, monkeypatch, synthetic_triangle(backend_name))
    cal, obs, val_cutoff = grids(entry, case)
    n_w, n_d = cal.shape

    training = [rec for rec in calls if rec["batch"] is not None]
    assert training, (
        "no forward pass was recorded inside train_loss - the spy is wired to the wrong "
        f"method, so this test proves nothing about {case.id}"
    )

    for rec in training:
        idx, cutoffs = rec["batch"]["idx"], rec["batch"]["cutoffs"]
        ctx = rec["ctx"].reshape(len(idx), -1, n_w, n_d)
        gate = cal[None, None] <= cutoffs[:, None, None, None]  # (B, 1, W, D)
        observed = obs[idx]

        leaked = int((ctx & ~gate).sum())
        assert not leaked, (
            f"{case.id}: {leaked} context cell(s) sit past their batch's augmented cutoff "
            f"{cutoffs.tolist()} - the training loss is conditioning on the cells it is "
            "being asked to predict"
        )
        val_leaked = int((ctx & (cal > val_cutoff)[None, None]).sum())
        assert not val_leaked, (
            f"{case.id}: {val_leaked} context cell(s) sit on the held-out validation "
            f"diagonal(s) (calendar index > {val_cutoff}); early stopping would be "
            "scoring data the training context already saw"
        )
        assert (observed & ~gate).any(), (
            f"{case.id}: this batch had no observed cell past its cutoff, so the gate "
            "excluded nothing and the no-leak assertions above are vacuous - the fixture "
            "has stopped exercising the mask"
        )
        np.testing.assert_array_equal(
            ctx,
            observed & gate,
            err_msg=(
                f"{case.id}: the training context is not exactly the observed cells at or "
                "before the augmented cutoff"
            ),
        )


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_validation_context_stops_at_the_validation_cutoff(case, backend_name, monkeypatch):
    """The validation pass conditions on the training window only.

    The other place the same leak hides: ``val_loss`` conditions on every
    context-eligible cell and scores the trailing diagonal(s). Hand it the raw
    observation mask instead and the held-out diagonal conditions on itself,
    which drives validation NLL down, which is exactly the signal early
    stopping trusts - so the fit stops on a number it has rigged.
    """
    entry, calls = record_fit(case, monkeypatch, synthetic_triangle(backend_name))
    cal, obs, val_cutoff = grids(entry, case)
    n_w, n_d = cal.shape

    validation = [rec for rec in calls if rec["batch"] is None]
    assert validation, (
        f"no forward pass was recorded outside train_loss for {case.id}; the per-epoch "
        "validation pass is where early stopping gets its number and it must be checked"
    )

    eligible = obs & (cal <= val_cutoff)[None, None]  # every cohort, training window only
    for rec in validation:
        ctx = rec["ctx"].reshape(obs.shape[0], -1, n_w, n_d)
        leaked = int((ctx & (cal > val_cutoff)[None, None]).sum())
        assert not leaked, (
            f"{case.id}: the validation pass conditions on {leaked} cell(s) past the "
            f"validation cutoff {val_cutoff} - i.e. on its own scoring targets"
        )
        np.testing.assert_array_equal(
            ctx,
            eligible,
            err_msg=(
                f"{case.id}: the validation context is not exactly the context-eligible "
                "cells (observed, at or before the validation cutoff)"
            ),
        )
