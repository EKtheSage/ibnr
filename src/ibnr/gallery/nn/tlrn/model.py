"""The ``tlrn`` gallery entry. See card.md.

Reproduces the companion study's checkpoint protocol on ibnr's data contract:
one training cutoff per epoch, validation on held-out calendar diagonals, ten
independently seeded members of which the two that validated best are kept, and
a company-level predictive distribution built from the kept checkpoints'
historical errors rather than from a distributional head.

WHAT THE ENTRY OWNS AND WHAT IT BORROWS. The feature tensors come from
``kernels.nn_features``, the network and head from the two modules beside this
one, the training loop from ``gallery/nn/_training.py`` and the calibration from
``kernels.residual_calibration``. What is here is the protocol: which cutoffs
train, which validate, which are calibrated on, how the members are ranked, and
how a company total is assembled from per-cell predictions.

THE ROLES, NOT A LIST OF CHANNELS. ``fit`` takes ``incurred_field`` and
``case_field`` rather than a general ``feature_fields``, because the five extra
features each read their channel differently: one is an incurred emergence, one
a paid-to-incurred ratio, one an incurred loss ratio and one an outstanding
balance. A generic list of channels cannot say which is which, so the entry
names the roles and refuses one without the other.

WHAT THE DISTRIBUTION IS. Not a native predictive distribution. The kept
checkpoints are applied at earlier cutoffs, their company-level errors are
standardised by a floored scale, pooled by company size and centred, and
resampled around the final point. So it is a statement about how wrong this
model has been historically on companies of that size, and it is calibrated at
COMPANY level only - there are no per-cell draws and this entry joins no
per-cell held-out board. The calibration cutoffs deliberately overlap the
training targets at the earlier ones, which is the study's own choice and which
the card discloses.

Torch is imported inside ``fit`` and ``predict`` only, so ``ibnr.gallery``
imports and registers this entry without the ``[nn]`` extra.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ibnr.gallery.entry import GalleryEntry
from ibnr.gallery.nn._scheme import splits
from ibnr.gallery.nn._training import train_ensemble, warmup_cosine
from ibnr.gallery.nn.tlrn.blend import blend_weight
from ibnr.gallery.nn.tlrn.config import TLRNConfig
from ibnr.gallery.registry import register
from ibnr.kernels.contract import _as_date
from ibnr.kernels.multiline import flatten_with_totals, multiline_targets
from ibnr.kernels.nn_contract import cohort_identities, nn_company_data
from ibnr.kernels.nn_features import pooled_incremental_lr, tlrn_features
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.kernels.residual_calibration import calibrate, calibrated_draws, rolling_residuals
from ibnr.kernels.rng import cohort_stream
from ibnr.triangle.core import Triangle

#: the arrays a feature set carries as float tensors
_FLOAT_KEYS = ("feat", "target", "target_mask", "premium", "c_lk", "anchor_start", "p_lk")
#: and as long tensors
_LONG_KEYS = ("lk", "line_ix", "lag_ix")
#: what can train the members: the torch loop, or one JAX program for all of them
BACKENDS: tuple[str, ...] = ("torch", "jax")


def _refuse_overlap(train_masks, val_masks) -> None:
    """No cell may be scored by a training set and by a validation set alike.

    The member kept is the one that validated best, so a cell in both windows
    would have the selection choose between members on data every one of them
    was trained on. The reference implementation checks this the same way, on
    the masks rather than on the arithmetic that built them: the arithmetic is
    exactly what a change here would get wrong.
    """
    if not train_masks or not val_masks:
        return
    trained = np.sum(train_masks, axis=0) > 0
    validated = np.sum(val_masks, axis=0) > 0
    shared = int((trained & validated).sum())
    if shared:
        raise ValueError(
            f"{shared} cell(s) are scored as both a training target and a validation "
            "target. Validation picks which member to keep, so a shared cell makes that "
            "choice on data every member saw. The training window must end at the first "
            "validation diagonal"
        )


def _training_parts(
    cfg, n_l, n_d, n_feat, dev, support, train_tensors, val_tensors, *, n_w=None, head_init=None
):
    """``make_model``, ``forward``, ``train_loss``, ``val_loss`` and ``param_groups``.

    Module level rather than closures inside ``fit`` so that a worker process can
    rebuild exactly what ``fit`` trains with from the arrays alone: a function defined
    inside ``fit`` cannot be sent to a process started by ``spawn``, which is the only
    start method Windows has. ``train_tensors`` maps each training cutoff to its
    tensor set. ``n_w`` is the number of accident years per company and ``head_init``
    the premium head's starting log loss ratios, (n_l, n_d), a numpy array.
    """
    import torch

    from ibnr.gallery.nn.tlrn import head as tlrn_head
    from ibnr.gallery.nn.tlrn.network import TLRNNetwork

    start = None if head_init is None else torch.tensor(head_init, dtype=torch.float32)

    def make_model():
        return TLRNNetwork(
            cfg, n_lines=n_l, n_lag=n_d, n_feat=n_feat, n_origin=n_w, head_init=start
        ).to(dev)

    def forward(model, tset: dict, rows=None, *, blend: bool = True) -> dict:
        """One pass. Under ``member="mcl_blend"`` the result is the blend of the network
        with the multivariate chain ladder, ``pred_net`` and ``C_net`` being the
        network alone; ``blend=False`` returns the network alone as ``pred``."""
        take = (lambda t: t) if rows is None else (lambda t: t[rows])
        out = model(
            take(tset["feat"]),
            take(tset["c_lk"]),
            take(tset["p_lk"]),
            take(tset["lk"]),
            tset["line_ix"],
            tset["lag_ix"],
            take(tset["written"]),
            factor_support=support,
            fallback_logf=tset["fallback_logf"],
            fallback_lr=tset["fallback_lr"],
            anchor_logf=take(tset["anchor_logf"]),
            anchor_start=take(tset["anchor_start"]),
            visible=take(tset["visible"]),
        )
        if cfg.member != "mcl_blend" or not blend:
            return out
        mcl_pred, mcl_c = take(tset["mcl_pred"]), take(tset["mcl_C"])
        a = model.alpha
        return {
            **out,
            "pred_net": out["pred"],
            "C_net": out["C"],
            "pred": mcl_pred + a * (out["pred"] - mcl_pred),
            "C": mcl_c + a * (out["C"] - mcl_c),
        }

    def example_rows(idx):
        """The example rows a batch of units names: itself, or every year of each company."""
        if cfg.batch_unit != "company":
            return idx
        years = torch.arange(n_w, device=idx.device)
        return (idx.unsqueeze(1) * n_w + years).reshape(-1)

    def train_loss(model, idx, cutoffs):
        k = int(cutoffs[0])
        if not bool((cutoffs == cutoffs[0]).all()):
            raise ValueError(
                "tlrn needs one cutoff for the whole batch, because a batch is scored "
                "against one feature set: set cutoff_sampling='per_epoch'"
            )
        tset = train_tensors[k]
        idx = example_rows(idx)
        mask = tset["target_mask"][idx]
        if not float(mask.sum()):
            return None  # this epoch's cutoff left this batch nothing to score
        out = forward(model, tset, idx, blend=False)
        return tlrn_head.point_loss(
            out["pred"],
            tset["target"][idx],
            mask,
            tset["premium"][idx],
            n_l,
            n_d,
            w_pe=cfg.w_pe,
            w_mse=cfg.w_mse,
            mse_scale=cfg.mse_scale,
        )

    def val_loss(model) -> float:
        if cfg.member == "mcl_blend":
            return TLRN._blend_ay_line_ape(model, val_tensors, forward, n_l, n_d)
        return TLRN._ay_line_ape(model, val_tensors, forward, n_l, n_d)

    def param_groups(m):
        owned = set(m.head_param_names)
        return [
            {"params": m.head_parameters(), "lr": cfg.lr_phi},
            {"params": [p for n, p in m.named_parameters() if n not in owned]},
        ]

    return make_model, forward, train_loss, val_loss, param_groups


#: what a worker process built once from its payload, reused for every member it trains
_WORKER: dict[str, Any] = {}


def _start_worker(payload_paths: list[str], threads: int) -> None:
    """Prepare a worker process: remember where each protocol's payload is.

    A payload is read from a file the caller wrote rather than sent with the worker:
    a worker that dies while starting never reads what it was sent, and on Windows the
    caller then waits forever writing several megabytes into that worker's pipe. It
    carries numpy arrays rather than tensors, so nothing here depends on how torch
    shares memory between processes; the tensors are made in the worker the same way
    ``fit`` makes them. ``threads`` is the calling process's torch thread count,
    because the arithmetic, and so the trained weights, depend on it.

    One pool serves every protocol of a fit (the final one and, under
    ``calibration="retrain_per_valuation"``, each earlier valuation date), so a
    protocol's parts are built the first time a member of it reaches this worker and
    kept for the rest.
    """
    import torch

    torch.set_num_threads(threads)
    _WORKER.clear()
    _WORKER.update(paths=list(payload_paths), parts={}, payload={}, dev=torch.device("cpu"))


def _worker_protocol(index: int):
    """The training parts and payload of protocol ``index``, built once per worker."""
    import pickle

    import torch

    if index not in _WORKER["parts"]:
        with open(_WORKER["paths"][index], "rb") as handle:
            payload = pickle.load(handle)  # written by this module's own caller, just now
        dev = _WORKER["dev"]
        support = payload["support"]
        _WORKER["parts"][index] = _training_parts(
            payload["cfg"],
            payload["n_l"],
            payload["n_d"],
            payload["n_feat"],
            dev,
            None if support is None else torch.tensor(support, device=dev),
            {k: TLRN._to_tensors(s, torch, dev) for k, s in payload["train_sets"].items()},
            [TLRN._to_tensors(s, torch, dev) for s in payload["val_sets"]],
            n_w=payload["n_w"],
            head_init=payload["head_init"],
        )
        _WORKER["payload"][index] = payload
    return _WORKER["parts"][index], _WORKER["payload"][index]


def _train_member(job: tuple[int, int]) -> tuple[int, int, dict, list[dict], int]:
    """Train member ``job[1]`` of protocol ``job[0]``, exactly as the whole ensemble would.

    Returns the torch thread count it trained on as well, so the caller can check the
    worker used the count it was sent: on a small problem the weights come out the same
    either way, so nothing else would show a worker that ignored it.
    """
    import torch

    index, member = job
    (make_model, _, train_loss, val_loss, param_groups), payload = _worker_protocol(index)
    cfg = payload["cfg"]
    models, history = train_ensemble(
        payload["n_units"],
        config=cfg,
        seed=payload["seed"],
        make_model=make_model,
        train_loss=train_loss,
        val_loss=val_loss,
        min_cutoff=cfg.min_cutoff,
        val_cutoff=payload["train_end"],
        device=_WORKER["dev"],
        schedule=warmup_cosine(cfg.max_epochs, cfg.warmup),
        param_groups=param_groups,
        min_epochs=cfg.min_epochs,
        check_every=cfg.check_every,
        cutoff_sampling=cfg.cutoff_sampling,
        members=[member],
    )
    state = {k: v.detach().cpu().clone() for k, v in models[0].state_dict().items()}
    return index, member, state, history[0], torch.get_num_threads()


@register
class TLRN(GalleryEntry):
    """The transformer loss reserving network (see card.md).

    One fit spans every company in the training triangle; a cohort is a company
    and one training example is that company at one accident year. ``point``
    gives a company's deterministic ultimates in the multi-line layout and
    ``predict`` gives its total ultimate as historically calibrated draws.

    Neither held-out mixin. The distribution this entry has is at company level,
    so it cannot answer a per-cell density or a per-cell draw, and claiming the
    capability would put a company-level spread on cells that never carried one.
    """

    name = "tlrn"
    family = "nn"
    #: the dataclass ``fit(config=...)`` takes, reachable through
    #: ``gallery.get("tlrn").config_class`` without importing it by path
    config_class = TLRNConfig

    def __init__(self) -> None:
        self.contract_: dict | None = None
        self.config_: TLRNConfig | None = None
        self.models_: list | None = None  # the kept checkpoints
        self.history_: list[list[dict]] | None = None
        self.selection_: pd.DataFrame | None = None  # one row per TRAINED member
        self.factor_support_: np.ndarray | None = None
        self.feature_stats_: dict | None = None
        self.point_ultimates_: np.ndarray | None = None  # (n_c, L, n_w)
        self.point_reserves_: np.ndarray | None = None  # (n_c, L, n_w)
        self.point_cumulative_: np.ndarray | None = None  # (n_c, L, n_w, n_d)
        # every TRAINED member's reserves, in selection_ row order
        self.member_reserves_: np.ndarray | None = None  # (n_members, n_c, L, n_w)
        self.calibration_: Any = None
        self.company_size_: np.ndarray | None = None  # (n_c,) premium
        self.backtest_: pd.DataFrame | None = None
        self._loss_field: str | None = None
        self._device: str = "cpu"

    # -- fit ---------------------------------------------------------------------

    def fit(
        self,
        triangle: Triangle,
        *,
        loss_field: str = "paid_loss",
        incurred_field: str | None = None,
        case_field: str | None = None,
        premium_field: str = "earned_premium",
        as_of: dt.date | str | None = None,
        config: TLRNConfig | None = None,
        device: str | None = None,
        seed: int | None = None,
        show_progress: bool = False,
        processes: int = 1,
        backend: str = "torch",
    ) -> TLRN:
        """Pooled fit across every company in the triangle.

        ``incurred_field`` and ``case_field`` switch the 8-feature form to the
        13-feature one and must be given together: the extra features read the
        two channels by ROLE, and a channel list cannot say which is the
        incurred emergence and which the outstanding balance. The case reserve
        is an evaluation-date level, so it is declared as one to the contract.

        ``processes`` trains the members in that many worker processes instead
        of one after another. It changes where they train, never what they
        learn: member ``m`` is seeded ``seed + 1000 * m`` wherever it runs, and
        every worker uses the calling process's torch thread count, so the fit
        is the one ``processes=1`` gives. Set the thread count first - a small
        network trains fastest on a few threads, and ``processes`` times that
        count should not exceed the cores. CPU only.

        Under ``calibration="retrain_per_valuation"`` the members of every valuation
        date's protocol go to the one pool, so ``processes`` can usefully exceed the
        ensemble size. Measured on a 16-thread laptop with the accident-year variant,
        a worker is compute-bound (a step is about 25 ms on one thread and the time per
        epoch does not fall past four threads), so the machine delivers roughly 10 to 15
        member-epochs per second however the threads are split; one thread per worker
        and as many workers as physical cores is the setting to start from.

        ``backend`` says what trains the members, and is an execution choice like
        ``processes``. ``"torch"`` (the default) is the loop above. ``"jax"`` trains every
        member of every valuation date at once, as one compiled JAX program, which is what
        a TPU or a GPU is fast at; it needs the ``[jax]`` extra and is slower than torch on
        a CPU. It changes how the members train and nothing after: each member starts from
        the weights torch would give it and sees the same batches and cutoffs, and the
        selection, the point, the calibration and ``predict`` run on torch either way. Only
        the dropout draws differ, so with dropout on the members are different draws of
        the same procedure, not the same numbers. It trains in one process, so it refuses
        ``processes > 1`` rather than ignoring it. See ``jax_backend.py``.
        """
        import torch

        if isinstance(processes, bool) or not isinstance(processes, int) or processes < 1:
            raise ValueError(f"processes must be a positive int, got {processes!r}")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {list(BACKENDS)}, got {backend!r}")
        if backend == "jax" and processes > 1:
            raise ValueError(
                f"backend='jax' trains every member in one compiled program in this process, "
                f"so processes={processes} would have nothing to do. Use processes=1 with "
                "backend='jax', or backend='torch' to spread members over worker processes"
            )
        cfg = config or TLRNConfig()
        train = triangle.as_of(as_of) if as_of is not None else triangle
        # BUILD FIRST, ASSIGN AFTER TRAINING SUCCEEDED - fit() must be atomic, so
        # a failed refit leaves the previous fit intact rather than half of each.
        extra = (incurred_field, case_field) if incurred_field and case_field else ()
        contract = nn_company_data(
            train,
            loss_field=loss_field,
            feature_fields=extra,
            level_fields=(case_field,) if extra else (),
            premium_field=premium_field,
        )
        dev = torch.device(device or "cpu")
        if processes > 1 and dev.type != "cpu":
            raise ValueError(
                f"processes={processes} trains in separate worker processes, which cannot "
                f"share the {dev.type!r} device this fit was asked for; use processes=1 "
                "on an accelerator"
            )
        n_c, n_l, _, n_w, n_d = contract["x"].shape
        last_diagonal = n_w + n_d - 1

        # the eval_date-style split: the trailing cfg.val_diagonals diagonals
        # validate, everything before them trains
        obs_any = contract["obs_mask"].any(axis=1)
        _, _, train_end = splits(obs_any, contract["cal_idx"], cfg.val_diagonals)
        c_max = train_end + cfg.val_diagonals

        def features(cutoff: int, target_lo: int, target_hi: int, max_lag=None) -> dict:
            return tlrn_features(
                contract,
                cutoff=cutoff,
                target_lo=target_lo,
                target_hi=target_hi,
                incurred_field=incurred_field,
                case_field=case_field,
                with_mcl=cfg.member == "mcl_blend",
                max_lag=max_lag,
            )

        bad_cutoffs = [k for k in cfg.calibration_cutoffs if not 1 <= k < c_max]
        if bad_cutoffs:
            raise ValueError(
                f"calibration cutoff(s) {bad_cutoffs} are outside 1 to {c_max - 1}: a "
                "calibration forecast is scored against diagonals this triangle has "
                f"already observed, and the latest of those is {c_max}"
            )
        if cfg.calibration == "retrain_per_valuation":
            too_early = [
                k for k in cfg.calibration_cutoffs if k - cfg.val_diagonals - 1 < cfg.min_cutoff
            ]
            if too_early:
                raise ValueError(
                    f"calibration cutoff(s) {too_early} leave no training cutoff once "
                    f"{cfg.val_diagonals} diagonal(s) validate and training starts at "
                    f"min_cutoff {cfg.min_cutoff}: retraining at a valuation date needs "
                    "a whole fit's worth of earlier diagonals"
                )
        specs = [
            {
                "c_max": c_max,
                "target_hi": last_diagonal,
                "seed": seed,
                "show_progress": show_progress,
            }
        ]
        if cfg.calibration == "retrain_per_valuation":
            # one protocol per calibration cutoff, each from what was known there; a
            # valuation date's members must not be the final fit's members
            specs += [
                {
                    "c_max": v,
                    "target_hi": c_max,
                    "final_max_lag": v if cfg.scoring == "reached_cells" else None,
                    "seed": None if seed is None else seed + 10_000 * v,
                    "show_progress": False,
                }
                for v in cfg.calibration_cutoffs
            ]
        runs = self._run_protocols(
            contract,
            cfg,
            specs,
            features=features,
            dev=dev,
            processes=processes,
            torch=torch,
            backend=backend,
        )
        run = runs[0]
        models, kept, kept_history = run["models"], run["kept"], run["kept_history"]
        selection, final_set, forward = run["selection"], run["final_set"], run["forward"]
        final_tensors = run["final_tensors"]
        support_np, train_sets = run["support_np"], run["train_sets"]

        point = self._ensemble_pred(kept, final_tensors, forward)
        ultimates, reserves, cumulative = self._assemble_point(
            point, final_set, contract, kept, final_tensors, forward
        )
        # every trained member's own reserves, the dropped ones included: the kept
        # ensemble is their mean over the kept rows, and any other group of members
        # can be scored from them without refitting
        member_reserves = np.stack(
            [
                self._assemble_point(
                    self._ensemble_pred([m], final_tensors, forward),
                    final_set,
                    contract,
                    [m],
                    final_tensors,
                    forward,
                )[1]
                for m in models
            ]
        )

        size = np.array(
            [
                np.nansum(
                    np.where(contract["line_mask"][c][:, None], contract["premium"][c], np.nan)
                )
                for c in range(n_c)
            ]
        )
        if cfg.calibration == "retrain_per_valuation":
            at = self._retrained_totals(runs[1:], cfg.calibration_cutoffs, n_c)
        else:
            calibration_sets = {k: features(k, k + 1, c_max) for k in cfg.calibration_cutoffs}
            tensors = {id(f): self._to_tensors(f, torch, dev) for f in calibration_sets.values()}
            at = self._rescored_totals(kept, calibration_sets, tensors, forward, n_c)
        residuals, calibration = self._calibrate(at, contract, cfg, c_max, size)

        self.contract_ = contract
        self.config_ = cfg
        self._loss_field = loss_field
        self._device = str(dev)
        self.models_, self.history_ = kept, kept_history
        self.selection_ = selection
        self.factor_support_ = support_np
        self.feature_stats_ = {
            "n_feat": final_set["n_feat"],
            "feature_names": final_set["feature_names"],
            "train_cutoffs": tuple(train_sets),
            "validation_cutoffs": tuple(range(c_max - cfg.val_diagonals, c_max)),
            "final_cutoff": c_max,
            "n_dropped": {k: s["n_dropped"] for k, s in train_sets.items()},
            "fallback_logf": final_set["fallback_logf"],
        }
        self.point_ultimates_ = ultimates
        self.point_reserves_ = reserves
        self.point_cumulative_ = cumulative
        self.member_reserves_ = member_reserves
        self.calibration_ = calibration
        self.company_size_ = size
        self.backtest_ = residuals
        return self

    def _run_protocols(
        self, contract, cfg, specs, *, features, dev, processes, torch, backend="torch"
    ) -> list[dict]:
        """Run the protocol once per spec and return each run, in spec order.

        Every spec is prepared first, then ALL their members are trained, then each is
        finished. With ``processes > 1`` the members of every protocol go to one pool, so
        the workers stay busy across protocols: twenty members on sixteen processes would
        otherwise leave twelve idle while the last four of each protocol finish. What a
        member learns is unchanged, because it is seeded from its own protocol's seed and
        its own index wherever and whenever it runs.
        """
        preps = [
            self._prepare_protocol(
                contract, cfg, features=features, dev=dev, processes=processes, torch=torch, **spec
            )
            for spec in specs
        ]
        if backend == "jax":
            trained = self._train_with_jax(preps, cfg, torch)
        elif processes == 1:
            trained = [
                train_ensemble(
                    prep["n_units"],
                    config=cfg,
                    seed=prep["seed"],
                    make_model=prep["make_model"],
                    train_loss=prep["train_loss"],
                    val_loss=prep["val_loss"],
                    min_cutoff=cfg.min_cutoff,
                    val_cutoff=prep["train_end"],
                    device=dev,
                    show_progress=prep["show_progress"],
                    schedule=warmup_cosine(cfg.max_epochs, cfg.warmup),
                    param_groups=prep["param_groups"],
                    min_epochs=cfg.min_epochs,
                    check_every=cfg.check_every,
                    cutoff_sampling=cfg.cutoff_sampling,
                    keep=None,
                )
                for prep in preps
            ]
        else:
            trained = self._train_in_processes(preps, processes, torch)
        return [
            self._finish_protocol(prep, cfg, models, history)
            for prep, (models, history) in zip(preps, trained, strict=True)
        ]

    def _prepare_protocol(
        self,
        contract,
        cfg,
        *,
        c_max,
        target_hi,
        final_max_lag=None,
        seed,
        show_progress,
        features,
        dev,
        processes,
        torch,
    ) -> dict:
        """Build everything the protocol at diagonal ``c_max`` trains and is scored on.

        The whole protocol for ONE valuation date: the trailing ``val_diagonals``
        diagonals up to ``c_max`` validate, everything before them train, and the
        final feature set forecasts ``c_max + 1`` to ``target_hi``. ``fit`` runs it
        at the latest observed diagonal; ``calibration="retrain_per_valuation"``
        runs it again at each earlier valuation date, reading nothing past that date
        into any input, so its forecasts of the later diagonals are out of sample.
        """
        from ibnr.gallery.nn.tlrn import head as tlrn_head

        n_c, n_l, _, n_w, n_d = contract["x"].shape
        train_end = c_max - cfg.val_diagonals
        if train_end - 1 < cfg.min_cutoff:
            raise ValueError(
                f"the latest observed diagonal is {c_max} and {cfg.val_diagonals} of them "
                f"validate, which leaves training cutoffs {cfg.min_cutoff} to {train_end - 1} "
                "- an empty range. Lower min_cutoff or val_diagonals, or fit on a triangle "
                "with more elapsed diagonals"
            )
        # one training set per cutoff; each is the same triangle presented as a
        # complete "forecast the next diagonals" task from a different date
        train_sets = {k: features(k, k + 1, train_end) for k in range(cfg.min_cutoff, train_end)}
        # the validation sets score every diagonal still held out at their own
        # cutoff, so the selection is not made on one-step forecasts alone
        reached = cfg.scoring == "reached_cells"
        val_sets = [
            features(k, k + 1, c_max, max_lag=k if reached else None)
            for k in range(train_end, c_max)
        ]
        _refuse_overlap(
            [s["target_mask"] for s in train_sets.values()],
            [s["target_mask"] for s in val_sets],
        )
        final_set = features(c_max, c_max + 1, target_hi, max_lag=final_max_lag)
        n_feat = final_set["n_feat"]
        n_ex = final_set["n_ex"]
        every_set = [*train_sets.values(), *val_sets, final_set]
        tensor_of = {id(s): self._to_tensors(s, torch, dev) for s in every_set}

        support = None
        if cfg.tail_policy == "observed_cl":
            # which parameters the TRAINING targets could move; read off the
            # masks (and the starting lags), never off a validation or a test value
            support_np = tlrn_head.factor_support(
                [s["target_mask"] for s in train_sets.values()],
                [s["lk"] for s in train_sets.values()],
                n_l,
                n_d,
            )
            support = torch.tensor(support_np, device=dev)
        else:
            support_np = None

        # the premium head starts at the pooled incremental loss ratio known when
        # training starts, which is the last training diagonal and not a later one
        head_init = (
            np.log(pooled_incremental_lr(contract, train_end)) if cfg.head == "premium_lr" else None
        )
        # what train_ensemble batches: examples, or whole companies
        n_units = n_c if cfg.batch_unit == "company" else n_ex

        val_tensors = [tensor_of[id(s)] for s in val_sets]
        make_model, forward, train_loss, val_loss, param_groups = _training_parts(
            cfg,
            n_l,
            n_d,
            n_feat,
            dev,
            support,
            {k: tensor_of[id(s)] for k, s in train_sets.items()},
            val_tensors,
            n_w=n_w,
            head_init=head_init,
        )
        payload = None
        if processes > 1:
            payload = {
                "cfg": cfg,
                "n_l": n_l,
                "n_d": n_d,
                "n_feat": n_feat,
                "n_units": n_units,
                "n_w": n_w,
                "head_init": head_init,
                "train_end": train_end,
                "seed": seed,
                "support": support_np,
                "train_sets": train_sets,
                "val_sets": val_sets,
            }
        return {
            "n_units": n_units,
            "seed": seed,
            "show_progress": show_progress,
            "train_end": train_end,
            "make_model": make_model,
            "forward": forward,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "param_groups": param_groups,
            "payload": payload,
            "val_tensors": val_tensors,
            "val_sets": val_sets,
            "n_c": n_c,
            "support_np": support_np,
            "train_sets": train_sets,
            "final_set": final_set,
            "final_tensors": tensor_of[id(final_set)],
        }

    def _finish_protocol(self, prep: dict, cfg, models: list, history: list) -> dict:
        """Select the members of a trained protocol and keep the pieces a fit reads."""
        forward = prep["forward"]

        def company_ape(model) -> float:
            return self._company_ape(
                model, prep["val_tensors"], forward, prep["val_sets"], prep["n_c"]
            )

        # keep=None: every member was trained and reported, and the selection is
        # made here so the table can carry the members that were dropped
        selection = self._selection_table(models, history, cfg.n_kept, company_ape)
        if cfg.member == "mcl_blend":
            # the weight each member's best checkpoint put on the network
            selection["alpha"] = [float(m.alpha) for m in models]
        kept_ix = selection.index[selection["kept"]].tolist()
        return {
            "models": models,
            "kept": [models[m] for m in kept_ix],
            "kept_history": [history[m] for m in kept_ix],
            "selection": selection,
            "final_set": prep["final_set"],
            "final_tensors": prep["final_tensors"],
            "forward": forward,
            "support_np": prep["support_np"],
            "train_sets": prep["train_sets"],
        }

    # -- the gallery contract -------------------------------------------------------

    def cohorts(self) -> list[dict]:
        """One dict per COMPANY, in ``contract_["companies"]`` row order."""
        if self.contract_ is None:
            raise RuntimeError("call fit() first")
        return cohort_identities(self.contract_, key="companies")

    def point(self, segment: Mapping | None = None) -> pd.DataFrame:
        """Deterministic ultimates, the mean over the kept checkpoints.

        With ``segment``: one company in the multi-line layout - per (line,
        origin) rows, per-line totals, then the grand total - so the same frame
        a ``sur`` or ``mcl`` point can be read beside. Without: one row per
        (company, line, origin) and no totals, because a total across companies
        is not a quantity anyone books.
        """
        if self.point_ultimates_ is None:
            raise RuntimeError("call fit() first")
        ci = self.cohort_index(segment)
        c = self.contract_
        if ci is not None:
            present = np.nonzero(c["line_mask"][ci])[0]
            lobs = [c["lob_levels"][li] for li in present]
            targets = multiline_targets(
                lobs, c["origin_periods"], premium=c["premium"][ci, present]
            )
            return targets.assign(point=flatten_with_totals(self.point_ultimates_[ci, present]))
        rows, values = [], []
        for cj in range(len(c["companies"])):
            for li in np.nonzero(c["line_mask"][cj])[0]:
                for w, origin in enumerate(c["origin_periods"]):
                    rows.append(
                        {
                            **dict(c["companies"].iloc[cj]),
                            "line_of_business": c["lob_levels"][li],
                            "origin_period": origin,
                            "premium": c["premium"][cj, li, w],
                        }
                    )
                    values.append(self.point_ultimates_[cj, li, w])
        return pd.DataFrame(rows).assign(point=values)

    def predict(
        self,
        segment: Mapping | None = None,
        n_draws: int | None = None,
        seed: int | None = None,
    ) -> PredictiveDistribution:
        """The company's TOTAL ultimate as historically calibrated draws.

        One target per company and nothing finer. The spread is resampled from
        what the kept checkpoints got wrong at earlier cutoffs on companies of
        similar premium, so it is a company-level statement and spreading it
        over cells would invent cell uncertainty this method never claimed.

        Every company is drawn in one pass whatever the segment, and the
        requested one is then selected, so ``predict(segment=a)`` and the
        matching column of ``predict()`` are the same numbers under one seed.
        """
        if self.calibration_ is None:
            raise RuntimeError("call fit() first")
        ci = self.cohort_index(segment)
        cfg = self.config_
        n_draws = n_draws or cfg.n_draws
        rng = np.random.default_rng(
            cohort_stream(seed, label="predict", cohorts=self.cohorts(), field=self._loss_field)
        )
        reserves = self.company_reserves()
        draws = calibrated_draws(
            self.calibration_,
            point=reserves,
            size=self.company_size_,
            n_draws=n_draws,
            rng=rng,
        )
        ultimates = draws + self.company_anchors()[None, :]
        c = self.contract_
        if ci is not None:
            targets = pd.DataFrame([{"label": "total", "premium": float(self.company_size_[ci])}])
            return PredictiveDistribution(samples=ultimates[:, [ci]], targets=targets)
        rows = [
            {
                **dict(c["companies"].iloc[cj]),
                "label": "total",
                "premium": float(self.company_size_[cj]),
            }
            for cj in range(len(c["companies"]))
        ]
        return PredictiveDistribution(samples=ultimates, targets=pd.DataFrame(rows))

    def realized_ultimates(
        self, full_triangle: Triangle, segment: Mapping | None = None
    ) -> np.ndarray:
        """Outcomes aligned to :meth:`predict`'s targets: one total per company.

        Read from the FULL triangle at the deepest development lag of the fit's
        own grid, over the origins the training slice had and the lines the
        company writes - the restriction that keeps a mart carrying accident
        years past the study window from inflating the total.
        """
        if self.contract_ is None:
            raise RuntimeError("call fit() first")
        ci = self.cohort_index(segment)
        c = self.contract_
        company_cols = list(c["companies"].columns)
        df = full_triangle.select_fields(self._loss_field).execute()
        df = df[df["dev_lag"] == c["n_d"] * c["dev_grain_months"]].copy()
        df["origin_period"] = _as_date(df["origin_period"])
        by_key = df.set_index([*company_cols, "line_of_business", "origin_period"])["value"]

        def total(cj: int) -> float:
            crow = tuple(c["companies"].iloc[cj])
            out = 0.0
            for li in np.nonzero(c["line_mask"][cj])[0]:
                lob = c["lob_levels"][li]
                for origin in c["origin_periods"]:
                    out += float(by_key.get((*crow, lob, origin), np.nan))
            return out

        if ci is not None:
            return np.array([total(ci)])
        return np.array([total(cj) for cj in range(len(c["companies"]))])

    # -- read-outs -------------------------------------------------------------------

    def company_reserves(self) -> np.ndarray:
        """``(n_c,)`` the point reserve of each company, over its written lines."""
        if self.point_reserves_ is None:
            raise RuntimeError("call fit() first")
        return np.nansum(self.point_reserves_, axis=(1, 2))

    def member_company_reserves(self) -> np.ndarray:
        """``(n_members, n_c)`` each TRAINED member's point reserve per company.

        Rows follow ``selection_``, so the members the selection dropped are here
        too. The kept ensemble averages its members' forecast cells and a reserve
        is a sum of cells, so ``company_reserves()`` is exactly the mean of the kept
        rows - and any other group of members, the best two of some other ten or all
        of them, is the mean of its rows, scored without refitting. Averaging every
        member is also available as a fit: ``keep = ensemble_size``.
        """
        if self.member_reserves_ is None:
            raise RuntimeError("call fit() first")
        return np.nansum(self.member_reserves_, axis=(2, 3))

    def company_anchors(self) -> np.ndarray:
        """``(n_c,)`` the cumulative paid to date of each company."""
        c = self.contract_
        anchored = np.where(c["line_mask"][:, :, None], c["latest_cum"], 0.0)
        return anchored.sum(axis=(1, 2))

    def predict_reserve_draws(
        self, point, *, seed: int | None = None, n_draws: int | None = None
    ) -> np.ndarray:
        """``(n_draws, n_c)`` calibrated draws around ANY company reserve vector.

        The study calibrates its spread around a BLENDED point - this model's
        reserve shrunk toward a classical one - rather than around this model's
        own. That blend is the notebook's to form, so the calibration is exposed
        for it here instead of being reachable only through :meth:`predict`.
        """
        if self.calibration_ is None:
            raise RuntimeError("call fit() first")
        rng = np.random.default_rng(
            cohort_stream(
                seed, label="predict_reserve_draws", cohorts=self.cohorts(), field=self._loss_field
            )
        )
        return calibrated_draws(
            self.calibration_,
            point=np.asarray(point, dtype=float).reshape(-1),
            size=self.company_size_,
            n_draws=n_draws or self.config_.n_draws,
            rng=rng,
        )

    # -- internals -------------------------------------------------------------------

    @staticmethod
    def _to_tensors(features: dict, torch, dev) -> dict:
        out = {k: torch.tensor(features[k], dtype=torch.float32, device=dev) for k in _FLOAT_KEYS}
        for k in ("fallback_logf", "anchor_logf", "fallback_lr"):
            out[k] = torch.tensor(features[k], dtype=torch.float32, device=dev)
        for k in _LONG_KEYS:
            out[k] = torch.tensor(features[k], dtype=torch.long, device=dev)
        out["written"] = torch.tensor(features["written"], dtype=torch.bool, device=dev)
        out["visible"] = torch.tensor(features["visible"], dtype=torch.bool, device=dev)
        for k in ("mcl_pred", "mcl_C"):
            if k in features:
                out[k] = torch.tensor(features[k], dtype=torch.float32, device=dev)
        return out

    @staticmethod
    def _train_in_processes(preps: list[dict], processes: int, torch) -> list[tuple[list, list]]:
        """Train every member of every protocol in one pool of ``spawn`` workers.

        Each worker builds a protocol's training parts the first time one of its members
        arrives and then trains whole members; only a (protocol, member) pair goes out
        and only its weights and history come back. The models are rebuilt here from
        those weights, in eval mode, as ``train_ensemble`` returns them. Returns
        ``(models, history)`` per protocol, each in member order.
        """
        import multiprocessing
        import os
        import pickle
        import tempfile
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool

        threads = torch.get_num_threads()
        jobs = [
            (i, member)
            for i, prep in enumerate(preps)
            for member in range(prep["payload"]["cfg"].ensemble_size)
        ]
        with tempfile.TemporaryDirectory(prefix="ibnr-tlrn-") as scratch:
            paths = []
            for i, prep in enumerate(preps):
                path = os.path.join(scratch, f"payload{i}.pkl")
                with open(path, "wb") as handle:
                    pickle.dump(prep["payload"], handle, protocol=pickle.HIGHEST_PROTOCOL)
                paths.append(path)
            try:
                with ProcessPoolExecutor(
                    max_workers=min(processes, len(jobs)),
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_start_worker,
                    initargs=(paths, threads),
                ) as pool:
                    trained = list(pool.map(_train_member, jobs))
            except BrokenProcessPool as err:
                raise RuntimeError(
                    "a tlrn worker process stopped before it finished training. The usual "
                    "cause: every worker re-imports the script that started it, so a script "
                    'that calls fit(processes=...) must do so under if __name__ == "__main__": '
                    "- otherwise each worker tries to start workers of its own and Python "
                    "stops it. A notebook needs no guard. The workers' own error is printed "
                    "above this one"
                ) from err
        used = sorted({t for *_, t in trained})
        if used != [threads]:
            raise RuntimeError(
                f"worker processes trained on {used} torch thread(s) where {threads} were "
                "asked for; the weights depend on the thread count, so these members are "
                "not the ones processes=1 would train"
            )
        out: list[tuple[list, list]] = [([], []) for _ in preps]
        for index, _, state, records, _ in trained:
            model = preps[index]["make_model"]()
            model.load_state_dict(state)
            model.eval()
            out[index][0].append(model)
            out[index][1].append(records)
        return out

    @staticmethod
    def _train_with_jax(preps: list[dict], cfg, torch) -> list[tuple[list, list]]:
        """Train every member of every protocol as one JAX program (``backend="jax"``).

        jax is imported here and nowhere else in this module, so the entry registers and
        ``backend="torch"`` fits without the [jax] extra.
        """
        try:
            from ibnr.gallery.nn.tlrn import jax_backend
        except ModuleNotFoundError as err:
            if err.name is None or err.name.split(".")[0] not in ("jax", "jaxlib"):
                raise
            raise ModuleNotFoundError(
                "backend='jax' needs jax, which is not installed: pip install 'ibnr[jax]'. "
                "On a Colab TPU or GPU runtime keep the jax the runtime already has",
                name=err.name,
            ) from err
        return jax_backend.train_protocols(preps, cfg, torch)

    @staticmethod
    def _ay_line_ape(model, tsets: list[dict], forward, n_l: int, n_d: int) -> float:
        """The validation score: absolute (accident year, line) error over actual.

        The numerator and the denominator are accumulated across BOTH validation
        sets before dividing, so the two horizons are weighted by the dollars
        they carry rather than averaged as two ratios.
        """
        import torch

        numerator = denominator = 0.0
        with torch.no_grad():
            for tset in tsets:
                pred = forward(model, tset)["pred"]
                weighted = tset["target_mask"] * tset["premium"]
                error = ((pred - tset["target"]) * weighted).reshape(-1, n_l, n_d).sum(2)
                actual = (tset["target"] * weighted).reshape(-1, n_l, n_d).sum(2)
                numerator += float(error.abs().sum())
                denominator += float(actual.abs().sum())
        return numerator / max(denominator, 1e-8)

    @staticmethod
    def _blend_ay_line_ape(model, tsets: list[dict], forward, n_l: int, n_d: int) -> float:
        """The validation score of the blend, after fitting its weight.

        The network's weight alpha is the exact minimiser of the same
        (accident year, line) absolute error ``_ay_line_ape`` scores, over both
        validation sets together (``blend.blend_weight``), and is written into the
        model before the score is taken: the weight is part of what validates, and
        the checkpoint that validates best keeps its own.
        """
        import torch

        errors, gaps, actuals = [], [], []
        with torch.no_grad():
            for tset in tsets:
                net = forward(model, tset, blend=False)["pred"]
                mcl = tset["mcl_pred"]
                weight = tset["target_mask"] * tset["premium"]

                def by_group(cells, weight=weight):
                    return (cells * weight).reshape(-1, n_l, n_d).sum(2).cpu().numpy()

                errors.append(by_group(mcl - tset["target"]))
                gaps.append(by_group(net - mcl))
                actuals.append(by_group(tset["target"]))
            error, gap = np.concatenate(errors), np.concatenate(gaps)
            alpha = blend_weight(error, gap)
            model.alpha.fill_(alpha)
        score = np.abs(error + alpha * gap).sum()
        return float(score / max(np.abs(np.concatenate(actuals)).sum(), 1e-8))

    @staticmethod
    def _company_totals(pred: np.ndarray, features: dict, n_c: int) -> np.ndarray:
        """``(n_c,)`` dollars: the scored cells of every example of each company."""
        weighted = pred * features["target_mask"] * features["premium"]
        per_example = weighted.sum(axis=1)
        out = np.zeros(n_c)
        np.add.at(out, features["example_company"], per_example)
        return out

    @classmethod
    def _company_ape(cls, model, tsets, forward, sets, n_c: int) -> float:
        """The study's company-level validation error, reported beside the one
        the selection is made on so the two can be compared."""
        import torch

        numerator = denominator = 0.0
        with torch.no_grad():
            for tset, features in zip(tsets, sets, strict=True):
                pred = forward(model, tset)["pred"].cpu().numpy()
                predicted = cls._company_totals(pred, features, n_c)
                actual = cls._company_totals(features["target"], features, n_c)
                numerator += float(np.abs(predicted - actual).sum())
                denominator += float(np.abs(actual).sum())
        return numerator / max(denominator, 1e-8)

    @staticmethod
    def _selection_table(models, history, keep: int, company_ape) -> pd.DataFrame:
        """One row per TRAINED member, with the kept ones flagged.

        Every member is reported, including the ones that were dropped: a
        selection over ten optimiser outcomes is a result in its own right, and
        a table that only showed the survivors would hide how much of the
        reported score is the selection rather than the model.
        """
        rows = []
        for member, records in enumerate(history):
            finite = [r for r in records if math.isfinite(r["val"])]
            best = min(finite, key=lambda r: r["val"]) if finite else None
            rows.append(
                {
                    "member": member,
                    "best_epoch": (best["epoch"] + 1) if best else None,
                    "epochs_run": len(records),
                    "validation_ay_line_ape": best["val"] if best else math.inf,
                    "validation_company_ape": company_ape(models[member]),
                    "kept": False,
                }
            )
        table = pd.DataFrame(rows)
        order = sorted(
            range(len(models)),
            key=lambda m: (table.loc[m, "validation_ay_line_ape"], m),
        )
        table.loc[sorted(order[:keep]), "kept"] = True
        return table

    @staticmethod
    def _ensemble_pred(models, tset: dict, forward) -> np.ndarray:
        import torch

        with torch.no_grad():
            stack = [forward(m, tset)["pred"].cpu().numpy() for m in models]
        return np.mean(stack, axis=0)

    @staticmethod
    def _assemble_point(pred, features, contract, models, tset, forward):
        """Per (company, line, origin) reserves, ultimates and the cumulative grid."""
        import torch

        n_c, n_l, _, n_w, n_d = contract["x"].shape
        weighted = (pred * features["target_mask"] * features["premium"]).reshape(-1, n_l, n_d)
        reserves = weighted.sum(axis=2).reshape(n_c, n_w, n_l).transpose(0, 2, 1)
        with torch.no_grad():
            grids = [forward(m, tset)["C"].cpu().numpy() for m in models]
        projected = np.mean(grids, axis=0).reshape(n_c, n_w, n_l, n_d).transpose(0, 2, 1, 3)
        # the head holds every cell at or before the latest visible lag at the
        # STARTING balance, which is the right thing for the projection and the
        # wrong thing to publish. The observed part of the grid is the triangle's
        # own, which the hole check in the feature builder guarantees is complete.
        lk = features["lk"].reshape(n_c, n_w)[0]  # the same for every company
        observed = np.arange(n_d)[None, :] < lk[:, None]  # (n_w, n_d)
        cumulative = np.where(observed[None, None], contract["values"][:, :, 0], projected)
        ultimates = contract["latest_cum"] + reserves

        absent = ~contract["line_mask"]
        reserves = np.where(absent[:, :, None], np.nan, reserves)
        ultimates = np.where(absent[:, :, None], np.nan, ultimates)
        cumulative = np.where(absent[:, :, None, None], np.nan, cumulative)
        return ultimates, reserves, cumulative

    @classmethod
    def _rescored_totals(cls, models, calibration_sets, tensor_of, forward, n_c):
        """``at(k)``: the kept checkpoints applied at cutoff ``k``, as company totals.

        These models were trained on diagonals up to the latest one, so at an earlier
        cutoff they are scored on cells they have already seen the answer to.
        """
        cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def at(k: int) -> tuple[np.ndarray, np.ndarray]:
            if k not in cache:
                features = calibration_sets[k]
                pred = cls._ensemble_pred(models, tensor_of[id(features)], forward)
                cache[k] = (
                    cls._company_totals(pred, features, n_c),
                    cls._company_totals(features["target"], features, n_c),
                )
            return cache[k]

        return at

    def _retrained_totals(self, runs, cutoffs, n_c):
        """``at(v)``: the whole method retrained from scratch at valuation ``v``.

        ``runs`` holds one trained protocol per cutoff, each rebuilt from what was known
        at that diagonal alone: its own training and validation windows, its own members
        and selection, its own pooled loss ratios and chain ladder. Its forecast of the
        later diagonals is therefore out of sample, which the rescored kind is not. The
        cost is a full fit per valuation date.
        """
        cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        by_cutoff = dict(zip(cutoffs, runs, strict=True))

        def at(v: int) -> tuple[np.ndarray, np.ndarray]:
            if v not in cache:
                run = by_cutoff[v]
                final = run["final_set"]
                pred = self._ensemble_pred(run["kept"], run["final_tensors"], run["forward"])
                cache[v] = (
                    self._company_totals(pred, final, n_c),
                    self._company_totals(final["target"], final, n_c),
                )
            return cache[v]

        return at

    @staticmethod
    def _calibrate(at, contract, cfg, c_max, size):
        """Pool the company-level errors ``at(k)`` returns at the calibration cutoffs."""
        residuals = rolling_residuals(
            lambda k: at(k)[0],
            lambda k: at(k)[1],
            cutoffs=cfg.calibration_cutoffs,
            # the horizon is how far past the cutoff a forecast reaches, and
            # these are scored to the latest observed diagonal
            n_periods=c_max,
            size=size,
        )
        calibration = calibrate(
            residuals,
            horizons=cfg.calibration_horizons,
            n_strata=cfg.n_strata,
            min_per_stratum=cfg.min_per_stratum,
        )
        companies = contract["companies"]
        labelled = residuals.assign(
            **{col: companies[col].to_numpy()[residuals["unit"].to_numpy()] for col in companies}
        )
        return labelled, calibration
