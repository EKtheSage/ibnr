"""kernels.tuning: random-search HPO over NN configs.

What this file protects, in three layers:

1. **Distributions.** Every ``Dist`` respects its bounds, log-scaled ones are
   actually log-spaced (an arithmetic-uniform impostor has a very different
   median), int ones return real Python ints, and invalid bounds fail at
   construction, not at draw time.
2. **The search loop.** The reproducibility contract is the load-bearing part:
   trial ``i``'s parameters come from one ``default_rng(seed)`` stream sampled
   BEFORE the fit, so a failing trial cannot perturb its successors, and
   ``trial_seed = seed + 10_000 * i`` is pinned literally. Failed trials are
   recorded (score NaN + exception text), non-finite scores can never become
   ``best_score`` (a ``-inf`` "NLL" would be unbeatable forever), and the
   trials frame's column layout is part of the API.
3. **Gallery glue.** ``validation_score`` is the per-member early-stopping
   best (min over epochs, NOT the last epoch), averaged over members; and
   ``tune_entry`` actually DELIVERS each sampled param into the trial's config
   and each trial seed into ``gallery.fit`` - proven through the public entry
   point with a monkeypatched ``gallery.fit``, per the inert-parameter rule
   (a signature proves a wire exists, not that it is connected).

Everything up to the final test is torch-free; the integration test
importorskips torch and runs two real (tiny) transformer fits.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from ibnr.gallery.nn.transformer.config import TransformerConfig
from ibnr.kernels.tuning import (
    Choice,
    IntChoice,
    IntLogUniform,
    LogUniform,
    Uniform,
    sample,
    tune,
    tune_entry,
    validation_score,
)

from .conftest import make_multiline_triangle

# -- distributions -------------------------------------------------------------


def test_choice_returns_the_exact_objects():
    """Choice never routes values through a numpy array (which would coerce a
    mixed-type set to strings); every value is reachable."""
    rng = np.random.default_rng(0)
    values = ("adam", 3, 2.5)
    c = Choice(values)
    draws = [c.sample(rng) for _ in range(200)]
    assert set(draws) == set(values)
    assert any(type(v) is int for v in draws)
    assert any(type(v) is str for v in draws)


def test_uniform_respects_bounds():
    rng = np.random.default_rng(1)
    u = Uniform(-2.0, 3.0)
    draws = np.array([u.sample(rng) for _ in range(500)])
    assert draws.min() >= -2.0
    assert draws.max() < 3.0
    assert draws.std() > 0.5  # spread over the interval, not stuck at a point


def test_loguniform_bounds_and_log_spacing():
    """Uniform-in-log: the median sits at the geometric midpoint (1e-2 for
    [1e-4, 1]), nowhere near the 0.5 an arithmetic uniform would give."""
    rng = np.random.default_rng(2)
    lu = LogUniform(1e-4, 1.0)
    draws = np.array([lu.sample(rng) for _ in range(2000)])
    assert draws.min() >= 1e-4
    assert draws.max() < 1.0
    assert 3e-3 < np.median(draws) < 3e-2


def test_int_distributions_return_python_ints():
    rng = np.random.default_rng(3)
    ic = IntChoice((8, 16, 32))
    draws = [ic.sample(rng) for _ in range(100)]
    assert all(type(v) is int for v in draws)
    assert set(draws) == {8, 16, 32}

    ilu = IntLogUniform(1, 1000)
    idraws = [ilu.sample(rng) for _ in range(2000)]
    assert all(type(v) is int for v in idraws)
    assert min(idraws) >= 1
    assert max(idraws) <= 1000
    # log spacing: geometric midpoint ~32; an arithmetic uniform's median ~500
    assert np.median(idraws) < 150


def test_invalid_distributions_fail_at_construction():
    with pytest.raises(ValueError):
        Choice(())
    with pytest.raises(TypeError):
        Choice(3)
    with pytest.raises(ValueError):
        Uniform(2.0, 2.0)
    with pytest.raises(ValueError):
        LogUniform(0.0, 1.0)
    with pytest.raises(ValueError):
        LogUniform(0.5, 0.5)
    with pytest.raises(TypeError):
        IntChoice((8, 16.0))
    with pytest.raises(TypeError):
        IntChoice((True, 2))
    with pytest.raises(ValueError):
        IntLogUniform(0, 10)
    with pytest.raises(TypeError):
        IntLogUniform(1, "big")
    with pytest.raises(ValueError):
        IntLogUniform(10, 10)


def test_sample_is_deterministic_per_seed():
    space = {
        "lr": LogUniform(1e-5, 1e-2),
        "d_model": IntChoice((16, 32, 64)),
        "dropout": Uniform(0.0, 0.5),
    }
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    seq_a = [sample(space, rng_a) for _ in range(10)]
    seq_b = [sample(space, rng_b) for _ in range(10)]
    assert seq_a == seq_b
    seq_c = [sample(space, np.random.default_rng(8)) for _ in range(10)]
    assert seq_a != seq_c


def test_sample_rejects_a_non_dist():
    with pytest.raises(TypeError, match="not a Dist"):
        sample({"lr": 0.1}, np.random.default_rng(0))


# -- tune ----------------------------------------------------------------------


def test_tune_finds_the_quadratic_minimum():
    def objective(params, trial_seed):
        return (params["x"] - 0.7) ** 2

    result = tune(objective, {"x": Uniform(-2.0, 2.0)}, n_trials=50, seed=11)
    assert abs(result.best_params["x"] - 0.7) < 0.2
    assert result.best_score < 0.04
    assert result.best_score == pytest.approx(result.trials["score"].min())


def test_trial_seeds_follow_the_documented_convention():
    """trial_seed = seed + 10_000 * i, echoing the family's widely-spaced
    member seeds (fit spreads members by 1000 within each trial)."""
    seen = []

    def objective(params, trial_seed):
        seen.append(trial_seed)
        return 1.0

    result = tune(objective, {"x": Uniform(0.0, 1.0)}, n_trials=4, seed=123)
    assert seen == [123, 10_123, 20_123, 30_123]
    assert list(result.trials["seed"]) == seen


def test_tune_is_deterministic_and_failures_do_not_perturb_the_stream():
    """Same seed -> identical trials; and because params are drawn BEFORE each
    fit from the one stream, a raising objective sees the exact same parameter
    sequence as a clean one."""
    space = {"x": Uniform(-1.0, 1.0)}

    def clean(params, trial_seed):
        return params["x"] ** 2

    def flaky(params, trial_seed):
        if params["x"] > 0:
            raise ValueError("bad region")
        return params["x"] ** 2

    a = tune(clean, space, n_trials=30, seed=5)
    b = tune(clean, space, n_trials=30, seed=5)
    pd.testing.assert_frame_equal(
        a.trials.drop(columns="seconds"), b.trials.drop(columns="seconds")
    )
    assert a.best_params == b.best_params

    # the parameter sequence is a pure function of the seed: it matches raw
    # stream sampling, and the flaky run reproduces it draw for draw
    rng = np.random.default_rng(5)
    expected_x = [sample(space, rng)["x"] for _ in range(30)]
    assert list(a.trials["x"]) == expected_x

    c = tune(flaky, space, n_trials=30, seed=5)
    assert list(c.trials["x"]) == expected_x
    assert len(c.trials) == 30

    failed = c.trials[c.trials["error"].notna()]
    assert np.array_equal(c.trials["error"].notna().to_numpy(), (c.trials["x"] > 0).to_numpy())
    assert len(failed) > 0
    assert failed["score"].isna().all()
    assert failed["error"].str.contains("ValueError: bad region").all()
    assert c.best_params["x"] <= 0


def test_all_trials_failing_raises():
    def broken(params, trial_seed):
        raise RuntimeError("no fit")

    with pytest.raises(RuntimeError, match="all 3 trials failed.*no fit"):
        tune(broken, {"x": Uniform(0.0, 1.0)}, n_trials=3, seed=0)


def test_non_finite_scores_are_failures_not_winners():
    """A -inf "NLL" would be an unbeatable best and NaN silently poisons
    ranking - both are recorded as failed trials instead."""

    def objective(params, trial_seed):
        return -math.inf if params["x"] > 0 else params["x"] ** 2

    result = tune(objective, {"x": Uniform(-1.0, 1.0)}, n_trials=20, seed=3)
    assert result.best_params["x"] <= 0
    assert math.isfinite(result.best_score)
    bad = result.trials[result.trials["error"].notna()]
    assert len(bad) > 0
    assert bad["error"].str.contains("non-finite").all()

    def nan_objective(params, trial_seed):
        return math.nan

    with pytest.raises(RuntimeError, match="all 2 trials failed"):
        tune(nan_objective, {"x": Uniform(0.0, 1.0)}, n_trials=2, seed=0)


def test_trials_frame_layout_and_input_validation():
    def objective(params, trial_seed):
        return params["a"] + params["b"]

    space = {"a": Uniform(0.0, 1.0), "b": Uniform(0.0, 1.0)}
    result = tune(objective, space, n_trials=3, seed=9)
    assert list(result.trials.columns) == ["trial", "a", "b", "score", "seed", "seconds", "error"]
    assert list(result.trials["trial"]) == [0, 1, 2]
    assert result.trials["error"].isna().all()
    assert (result.trials["seconds"] >= 0).all()

    with pytest.raises(ValueError, match="reserved"):
        tune(objective, {"score": Uniform(0.0, 1.0)}, n_trials=1, seed=0)
    with pytest.raises(ValueError, match="n_trials"):
        tune(objective, space, n_trials=0, seed=0)


def test_keep_top_and_top():
    def objective(params, trial_seed):
        if params["x"] > 0.8:
            raise ValueError("boom")
        return params["x"]

    result = tune(objective, {"x": Uniform(0.0, 1.0)}, n_trials=12, seed=21, keep_top=4)
    n_ok = int(result.trials["error"].isna().sum())
    assert 4 <= n_ok < 12  # the fixture must exercise both failure and truncation

    t = result.top()
    assert len(t) == 4  # default report size = keep_top
    assert list(t["score"]) == sorted(t["score"])
    assert t["error"].isna().all()
    assert t.loc[0, "score"] == result.best_score
    assert len(result.top(2)) == 2
    assert len(result.top(100)) == n_ok  # never pads with failed trials


# -- gallery glue --------------------------------------------------------------


class _StubEntry:
    def __init__(self, history):
        self.history_ = history


def test_validation_score_is_the_mean_early_stopping_best():
    """Per member the MIN val over epochs (the early-stopping best, whose
    weights fit() restores - member 0's last epoch is deliberately worse than
    its best), averaged across members."""
    history = [
        [
            {"epoch": 0, "train": 5.0, "val": 4.0},
            {"epoch": 1, "train": 4.0, "val": 3.0},
            {"epoch": 2, "train": 3.5, "val": 3.4},
        ],
        [
            {"epoch": 0, "train": 6.0, "val": 5.0},
            {"epoch": 1, "train": 5.0, "val": 4.6},
        ],
    ]
    assert validation_score(_StubEntry(history)) == pytest.approx((3.0 + 4.6) / 2)


def test_validation_score_raises_clearly_when_history_is_missing():
    with pytest.raises(ValueError, match="empty or None"):
        validation_score(_StubEntry([]))
    with pytest.raises(ValueError, match="empty or None"):
        validation_score(_StubEntry(None))
    with pytest.raises(ValueError, match="member 1"):
        validation_score(_StubEntry([[{"epoch": 0, "train": 1.0, "val": 1.0}], []]))

    class NoHistory:
        pass

    with pytest.raises(TypeError, match="history_"):
        validation_score(NoHistory())


def test_tune_entry_delivers_params_and_seeds_through_gallery_fit(monkeypatch):
    """The wire is connected, not just present: each trial's config IS
    base_config with exactly the sampled params replaced, each fit gets the
    documented trial seed, fit_kwargs pass through verbatim, and the score is
    validation_score of what gallery.fit returned."""
    import ibnr.gallery as gallery

    calls = []

    def fake_fit(name, triangle, *, config, seed, **kwargs):
        calls.append(
            {"name": name, "triangle": triangle, "config": config, "seed": seed, "kwargs": kwargs}
        )
        # score = the trial's own dropout, so best_score pins score wiring too
        return _StubEntry([[{"epoch": 0, "train": 0.0, "val": config.dropout}]])

    monkeypatch.setattr(gallery, "fit", fake_fit)

    base = TransformerConfig(d_model=16, n_heads=2)
    space = {"dropout": Uniform(0.0, 0.5), "d_model": IntChoice((16, 32))}
    result = tune_entry(
        "nn_transformer",
        "THE_TRIANGLE",
        space,
        base_config=base,
        n_trials=3,
        seed=42,
        fit_kwargs={"loss_field": "paid_loss"},
    )

    assert [c["name"] for c in calls] == ["nn_transformer"] * 3
    assert [c["seed"] for c in calls] == [42, 10_042, 20_042]
    assert all(c["triangle"] == "THE_TRIANGLE" for c in calls)
    assert all(c["kwargs"] == {"loss_field": "paid_loss"} for c in calls)

    rng = np.random.default_rng(42)
    for c in calls:
        params = sample(space, rng)
        assert c["config"] == dataclasses.replace(base, **params)
        assert c["config"] != base  # the sampled params actually landed

    assert result.best_score == pytest.approx(min(c["config"].dropout for c in calls))


def test_tune_entry_rejects_bad_spaces_before_any_fit():
    base = TransformerConfig()
    with pytest.raises(ValueError, match="not fields of TransformerConfig"):
        tune_entry(
            "nn_transformer",
            None,
            {"lerning_rate": Uniform(0.0, 1.0)},
            base_config=base,
            n_trials=1,
            seed=0,
        )
    with pytest.raises(TypeError, match="dataclass instance"):
        tune_entry(
            "nn_transformer",
            None,
            {"lr": Uniform(0.0, 1.0)},
            base_config=TransformerConfig,
            n_trials=1,
            seed=0,
        )


# -- integration (torch) -------------------------------------------------------


def test_tune_entry_runs_real_transformer_fits():
    """Two real (tiny) nn_transformer fits through the public tune_entry path:
    both trials succeed, scores are finite validation NLLs, and the winning
    params come from the space."""
    pytest.importorskip("torch")

    rng = np.random.default_rng(0)
    n_w = n_d = 6
    dev_level = np.exp(np.linspace(-0.8, -3.0, n_d))
    incr = 1000.0 * dev_level[None, None, :] * rng.lognormal(0.0, 0.1, size=(2, n_w, n_d))
    cum = np.cumsum(incr, axis=2)
    lobs = {f"lob_{k}": cum[k] for k in range(2)}
    prem = {f"lob_{k}": np.full(n_w, 1000.0) for k in range(2)}
    triangle = make_multiline_triangle("duckdb", lobs, premium_by_lob=prem, start_year=2000)

    base = TransformerConfig(
        d_model=16,
        n_layers=1,
        n_heads=2,
        ffn_dim=32,
        dropout=0.0,
        n_components=2,
        lob_embedding_dim=4,
        batch_size=8,
        max_epochs=2,
        patience=3,
        ensemble_size=1,
        n_draws=20,
    )
    space = {"lr": LogUniform(1e-4, 1e-3), "dropout": Choice((0.0, 0.1))}
    result = tune_entry(
        "nn_transformer",
        triangle,
        space,
        base_config=base,
        n_trials=2,
        seed=0,
        fit_kwargs={"loss_field": "paid_loss"},
    )
    assert len(result.trials) == 2
    assert result.trials["error"].isna().all()
    assert np.isfinite(result.trials["score"]).all()
    assert set(result.best_params) == {"lr", "dropout"}
    assert 1e-4 <= result.best_params["lr"] < 1e-3
    assert result.best_params["dropout"] in (0.0, 0.1)
