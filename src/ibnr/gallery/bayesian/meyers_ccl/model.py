"""Meyers Correlated Chain Ladder (CCL) gallery entry — Stan reference
implementation. See card.md for the model card."""

from __future__ import annotations

import datetime as dt
import os
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ibnr.gallery.entry import GalleryEntry
from ibnr.gallery.registry import register
from ibnr.kernels.contract import realized_values, stan_data
from ibnr.kernels.predictive import PredictiveDistribution
from ibnr.triangle.core import Triangle

STAN_FILE = Path(__file__).parent / "model.stan"

#: keys of the standardized contract dict that form the Stan data block
STAN_DATA_KEYS = ("len_data", "n_w", "n_d", "w", "d", "prev_idx", "logprem", "logloss")


@register
class MeyersCCL(GalleryEntry):
    name = "meyers_ccl"
    family = "bayesian"

    def __init__(self) -> None:
        self.contract_: dict | None = None
        self.fit_ = None
        self._loss_field: str | None = None

    def fit(
        self,
        triangle: Triangle,
        *,
        # Meyers' "incurred" is net of bulk+IBNR (paid + case) = reported_loss
        loss_field: str = "reported_loss",
        premium_field: str = "earned_premium",
        as_of: dt.date | str | None = None,
        chains: int = 4,
        iter_warmup: int = 1000,
        iter_sampling: int = 2500,
        seed: int | None = None,
        show_progress: bool = False,
    ) -> MeyersCCL:
        from cmdstanpy import CmdStanModel

        _ensure_windows_toolchain()
        train = triangle.as_of(as_of) if as_of is not None else triangle
        self.contract_ = stan_data(train, loss_field=loss_field, premium_field=premium_field)
        self._loss_field = loss_field
        model = CmdStanModel(stan_file=str(STAN_FILE))
        self.fit_ = model.sample(
            data={k: self.contract_[k] for k in STAN_DATA_KEYS},
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            show_progress=show_progress,
        )
        return self

    def predict(self, seed: int | None = None) -> PredictiveDistribution:
        """Predictive distribution of ultimates (losses at the last dev period)
        by origin year plus their total, via the monograph's simulation:
        sequentially over origins, mu[w] = logprem[w] + logelr + alpha[w]
        + rho * (log(C[w-1, n_d]) - mu[w-1]), C[w, n_d] ~ lognormal(mu[w], sig[n_d]).
        """
        if self.fit_ is None or self.contract_ is None:
            raise RuntimeError("call fit() first")
        c = self.contract_
        n_w, n_d = c["n_w"], c["n_d"]

        alpha = self.fit_.stan_variable("alpha")  # (draws, n_w)
        logelr = self.fit_.stan_variable("logelr")  # (draws,)
        rho = self.fit_.stan_variable("rho")  # (draws,)
        sig = self.fit_.stan_variable("sig")  # (draws, n_d)
        n_draws = logelr.shape[0]
        sig_last = sig[:, n_d - 1]
        logprem = np.log(c["premium"])  # per origin

        first_mask = (c["w"] == 1) & (c["d"] == n_d)
        if not first_mask.any():
            raise ValueError(
                "the first origin year is not fully developed in the training data; "
                "the CCL simulation needs the observed C[1, n_d] anchor"
            )
        c1_ult = float(c["loss"][first_mask][0])

        rng = np.random.default_rng(seed)
        ults = np.empty((n_draws, n_w))
        ults[:, 0] = c1_ult
        mu_prev = logprem[0] + logelr  # alpha[1] = 0, beta[n_d] = 0
        log_c_prev = np.full(n_draws, np.log(c1_ult))
        for j in range(1, n_w):
            mu = logprem[j] + logelr + alpha[:, j] + rho * (log_c_prev - mu_prev)
            log_c = rng.normal(mu, sig_last)
            ults[:, j] = np.exp(log_c)
            mu_prev, log_c_prev = mu, log_c

        targets = pd.DataFrame(
            {
                "label": [str(o.year) for o in c["origin_periods"]],
                "origin_period": c["origin_periods"],
                "premium": c["premium"],
            }
        )
        pred = PredictiveDistribution(samples=ults, targets=targets)
        return pred.with_total()

    def realized_ultimates(self, full_triangle: Triangle) -> np.ndarray:
        """Outcomes aligned to predict()'s targets (per origin + total), taken
        from the full triangle at the final development lag."""
        if self.contract_ is None:
            raise RuntimeError("call fit() first")
        c = self.contract_
        per_origin = realized_values(
            full_triangle,
            loss_field=self._loss_field,
            dev_lag=c["n_d"] * c["dev_grain_months"],
            origins=c["origin_periods"],
        )
        return np.append(per_origin, per_origin.sum())


def _ensure_windows_toolchain() -> None:
    """Best-effort: put an RTools g++/make on PATH so cmdstan can compile.

    cmdstanpy assumes mingw32-make/RTools40; modern RTools installs (43/44/45)
    ship plain `make`, so we also set MAKE. No-op outside Windows or when a
    toolchain is already reachable.
    """
    if platform.system() != "Windows":
        return
    if shutil.which("g++") and (shutil.which("mingw32-make") or shutil.which("make")):
        os.environ.setdefault("MAKE", "mingw32-make" if shutil.which("mingw32-make") else "make")
        return
    for root in ("C:/rtools45", "C:/rtools44", "C:/rtools43", "C:/rtools40"):
        gxx = Path(root) / "x86_64-w64-mingw32.static.posix" / "bin"
        mk = Path(root) / "usr" / "bin"
        if gxx.exists() and mk.exists():
            os.environ["PATH"] = f"{gxx};{mk};{os.environ['PATH']}"
            os.environ.setdefault("MAKE", "make")
            return
