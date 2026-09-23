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
from ibnr.gallery.nn.tlrn.config import TLRNConfig
from ibnr.gallery.registry import register
from ibnr.kernels.contract import _as_date
from ibnr.kernels.multiline import flatten_with_totals, multiline_targets
from ibnr.kernels.nn_contract import cohort_identities, nn_company_data
from ibnr.kernels.nn_features import tlrn_features
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.kernels.residual_calibration import calibrate, calibrated_draws, rolling_residuals
from ibnr.kernels.rng import cohort_stream
from ibnr.triangle.core import Triangle

#: the arrays a feature set carries as float tensors
_FLOAT_KEYS = ("feat", "target", "target_mask", "premium", "c_lk", "anchor_start", "p_lk")
#: and as long tensors
_LONG_KEYS = ("lk", "line_ix", "lag_ix")


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


def _training_parts(cfg, n_l, n_d, n_feat, dev, support, train_tensors, val_tensors):
    """``make_model``, ``forward``, ``train_loss``, ``val_loss`` and ``param_groups``.

    Module level rather than closures inside ``fit`` so that a worker process can
    rebuild exactly what ``fit`` trains with from the arrays alone: a function defined
    inside ``fit`` cannot be sent to a process started by ``spawn``, which is the only
    start method Windows has. ``train_tensors`` maps each training cutoff to its
    tensor set.
    """
    from ibnr.gallery.nn.tlrn import head as tlrn_head
    from ibnr.gallery.nn.tlrn.network import TLRNNetwork

    def make_model():
        return TLRNNetwork(cfg, n_lines=n_l, n_lag=n_d, n_feat=n_feat).to(dev)

    def forward(model, tset: dict, rows=None) -> dict:
        take = (lambda t: t) if rows is None else (lambda t: t[rows])
        return model(
            take(tset["feat"]),
            take(tset["c_lk"]),
            take(tset["p_lk"]),
            take(tset["lk"]),
            tset["line_ix"],
            tset["lag_ix"],
            take(tset["written"]),
            factor_support=support,
            fallback_logf=tset["fallback_logf"],
            anchor_logf=take(tset["anchor_logf"]),
            anchor_start=take(tset["anchor_start"]),
        )

    def train_loss(model, idx, cutoffs):
        k = int(cutoffs[0])
        if not bool((cutoffs == cutoffs[0]).all()):
            raise ValueError(
                "tlrn needs one cutoff for the whole batch, because a batch is scored "
                "against one feature set: set cutoff_sampling='per_epoch'"
            )
        tset = train_tensors[k]
        mask = tset["target_mask"][idx]
        if not float(mask.sum()):
            return None  # this epoch's cutoff left this batch nothing to score
        out = forward(model, tset, idx)
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
        return TLRN._ay_line_ape(model, val_tensors, forward, n_l, n_d)

    def param_groups(m):
        return [
            {"params": [m.phi], "lr": cfg.lr_phi},
            {"params": [p for n, p in m.named_parameters() if n != "phi"]},
        ]

    return make_model, forward, train_loss, val_loss, param_groups


#: what a worker process built once from its payload, reused for every member it trains
_WORKER: dict[str, Any] = {}


def _start_worker(payload: dict, threads: int) -> None:
    """Rebuild the training parts in a worker process, once, from plain arrays.

    The payload carries numpy arrays rather than tensors, so nothing here depends on
    how torch shares memory between processes; the tensors are made in the worker the
    same way ``fit`` makes them. ``threads`` is the calling process's torch thread
    count, because the arithmetic, and so the trained weights, depend on it.
    """
    import torch

    torch.set_num_threads(threads)
    dev = torch.device("cpu")
    support = payload["support"]
    parts = _training_parts(
        payload["cfg"],
        payload["n_l"],
        payload["n_d"],
        payload["n_feat"],
        dev,
        None if support is None else torch.tensor(support, device=dev),
        {k: TLRN._to_tensors(s, torch, dev) for k, s in payload["train_sets"].items()},
        [TLRN._to_tensors(s, torch, dev) for s in payload["val_sets"]],
    )
    _WORKER.update(parts=parts, payload=payload, dev=dev)


def _train_member(member: int) -> tuple[int, dict, list[dict], int]:
    """Train one member in a worker process, exactly as the whole ensemble would.

    Returns the torch thread count it trained on as well, so the caller can check the
    worker used the count it was sent: on a small problem the weights come out the same
    either way, so nothing else would show a worker that ignored it.
    """
    import torch

    make_model, _, train_loss, val_loss, param_groups = _WORKER["parts"]
    payload = _WORKER["payload"]
    cfg = payload["cfg"]
    models, history = train_ensemble(
        payload["n_ex"],
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
    return member, state, history[0], torch.get_num_threads()


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
        """
        import torch

        from ibnr.gallery.nn.tlrn import head as tlrn_head

        if isinstance(processes, bool) or not isinstance(processes, int) or processes < 1:
            raise ValueError(f"processes must be a positive int, got {processes!r}")
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
        if train_end - 1 < cfg.min_cutoff:
            raise ValueError(
                f"the latest observed diagonal is {c_max} and {cfg.val_diagonals} of them "
                f"validate, which leaves training cutoffs {cfg.min_cutoff} to {train_end - 1} "
                "- an empty range. Lower min_cutoff or val_diagonals, or fit on a triangle "
                "with more elapsed diagonals"
            )

        def features(cutoff: int, target_lo: int, target_hi: int) -> dict:
            return tlrn_features(
                contract,
                cutoff=cutoff,
                target_lo=target_lo,
                target_hi=target_hi,
                incurred_field=incurred_field,
                case_field=case_field,
            )

        # one training set per cutoff; each is the same triangle presented as a
        # complete "forecast the next diagonals" task from a different date
        train_sets = {k: features(k, k + 1, train_end) for k in range(cfg.min_cutoff, train_end)}
        # the validation sets score every diagonal still held out at their own
        # cutoff, so the selection is not made on one-step forecasts alone
        val_sets = [features(k, k + 1, c_max) for k in range(train_end, c_max)]
        _refuse_overlap(
            [s["target_mask"] for s in train_sets.values()],
            [s["target_mask"] for s in val_sets],
        )
        final_set = features(c_max, c_max + 1, last_diagonal)
        bad_cutoffs = [k for k in cfg.calibration_cutoffs if not 1 <= k < c_max]
        if bad_cutoffs:
            raise ValueError(
                f"calibration cutoff(s) {bad_cutoffs} are outside 1 to {c_max - 1}: a "
                "calibration forecast is scored against diagonals this triangle has "
                f"already observed, and the latest of those is {c_max}"
            )
        calibration_sets = {k: features(k, k + 1, c_max) for k in cfg.calibration_cutoffs}

        n_feat = final_set["n_feat"]
        n_ex = final_set["n_ex"]
        every_set = [*train_sets.values(), *val_sets, final_set, *calibration_sets.values()]
        tensor_of = {id(s): self._to_tensors(s, torch, dev) for s in every_set}

        support = None
        if cfg.tail_policy == "observed_cl":
            # which factors the TRAINING targets could move; read off the masks
            # and the starting lags, never off a validation or a test value
            support_np = tlrn_head.factor_support(
                [s["target_mask"] for s in train_sets.values()],
                [s["lk"] for s in train_sets.values()],
                n_l,
                n_d,
            )
            support = torch.tensor(support_np, device=dev)
        else:
            support_np = None

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
        )

        def company_ape(model) -> float:
            return self._company_ape(model, val_tensors, forward, val_sets, n_c)

        # keep=None: every member is trained and reported, and the selection is
        # made here so the table can carry the members that were dropped
        if processes == 1:
            models, history = train_ensemble(
                n_ex,
                config=cfg,
                seed=seed,
                make_model=make_model,
                train_loss=train_loss,
                val_loss=val_loss,
                min_cutoff=cfg.min_cutoff,
                val_cutoff=train_end,
                device=dev,
                show_progress=show_progress,
                schedule=warmup_cosine(cfg.max_epochs, cfg.warmup),
                param_groups=param_groups,
                min_epochs=cfg.min_epochs,
                check_every=cfg.check_every,
                cutoff_sampling=cfg.cutoff_sampling,
                keep=None,
            )
        else:
            payload = {
                "cfg": cfg,
                "n_l": n_l,
                "n_d": n_d,
                "n_feat": n_feat,
                "n_ex": n_ex,
                "train_end": train_end,
                "seed": seed,
                "support": support_np,
                "train_sets": train_sets,
                "val_sets": val_sets,
            }
            models, history = self._train_in_processes(
                payload, processes, make_model, cfg.ensemble_size, torch
            )

        selection = self._selection_table(models, history, cfg.keep, company_ape)
        kept_ix = selection.index[selection["kept"]].tolist()
        kept = [models[m] for m in kept_ix]
        kept_history = [history[m] for m in kept_ix]

        final_tensors = tensor_of[id(final_set)]
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
        residuals, calibration = self._calibrate(
            kept, calibration_sets, tensor_of, forward, contract, cfg, c_max, size
        )

        self.contract_ = contract
        self.config_ = cfg
        self._loss_field = loss_field
        self._device = str(dev)
        self.models_, self.history_ = kept, kept_history
        self.selection_ = selection
        self.factor_support_ = support_np
        self.feature_stats_ = {
            "n_feat": n_feat,
            "feature_names": final_set["feature_names"],
            "train_cutoffs": tuple(train_sets),
            "validation_cutoffs": tuple(range(train_end, c_max)),
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
        for k in ("fallback_logf", "anchor_logf"):
            out[k] = torch.tensor(features[k], dtype=torch.float32, device=dev)
        for k in _LONG_KEYS:
            out[k] = torch.tensor(features[k], dtype=torch.long, device=dev)
        out["written"] = torch.tensor(features["written"], dtype=torch.bool, device=dev)
        return out

    @staticmethod
    def _train_in_processes(payload: dict, processes: int, make_model, n_members: int, torch):
        """Train every member in a pool of ``spawn`` worker processes, in member order.

        Each worker rebuilds the training parts once from ``payload`` and then trains
        whole members; only a member index goes out and only its weights and history
        come back. The models are rebuilt here from those weights, in eval mode, as
        ``train_ensemble`` returns them.
        """
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        threads = torch.get_num_threads()
        with ProcessPoolExecutor(
            max_workers=min(processes, n_members),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_start_worker,
            initargs=(payload, threads),
        ) as pool:
            trained = list(pool.map(_train_member, range(n_members)))
        used = sorted({t for *_, t in trained})
        if used != [threads]:
            raise RuntimeError(
                f"worker processes trained on {used} torch thread(s) where {threads} were "
                "asked for; the weights depend on the thread count, so these members are "
                "not the ones processes=1 would train"
            )
        models, history = [], []
        for _, state, records, _ in trained:
            model = make_model()
            model.load_state_dict(state)
            model.eval()
            models.append(model)
            history.append(records)
        return models, history

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
    def _calibrate(cls, models, calibration_sets, tensor_of, forward, contract, cfg, c_max, size):
        """Apply the kept checkpoints at earlier cutoffs and pool their errors."""
        n_c = len(contract["companies"])
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
