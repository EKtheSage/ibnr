"""Write ``tests/data/odp_cdr_pin.json``: the one-year CDR's ODP route before the run-off bootstrap.

``tests/test_odp_runoff.py`` recomputes every case with the current code and
requires the same digest, so extending ``kernels/odp_bootstrap.py`` for the
full run-off bootstrap moved no number of the one-year CDR's ODP route by a
single bit:

- ``fit``: ``fit_odp_bootstrap`` with its defaults (the arrays ``inc``,
  ``fitted``, ``residuals`` and ``pool_mask``, and ``phi``, ``n_cells``,
  ``n_params``), fed ``fit_mack_grid``'s factors as the CDR feeds them;
- ``next``: ``draw_next_increments`` for both process laws and the three arms
  (both risk sources, residuals only, process noise only), 2,000 draws from
  ``default_rng(7)``;
- ``cdr``: ``simulate_one_year_cdr(fit, generator=ODPBootstrapDiagonal(...))``
  for both laws, 3,000 draws, seed 11: the samples, digested by their bits.

A triangle the ODP bootstrap refuses is pinned by its reason and message.

The file was frozen from the branch ``feat/tails`` (the code before the run-off
bootstrap) by running this script against that branch's source:

    git archive feat/tails src | tar -x -C <somewhere>
    uv run python scripts/freeze_odp_cdr_pin.py <somewhere>/src

The triangles are the five public ones, the 30 x 30 one built from a formula,
and every fourteenth clrd paid-loss cohort (needs chainladder-python).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "odp_cdr_pin.json"
sys.path.insert(0, str(ROOT / "scripts"))

import freeze_conventional_pin as conventional  # noqa: E402

#: draw_next_increments arms: (process_noise, resample_residuals).
ARMS = ((True, True), (False, True), (True, False))
LAWS = ("od_poisson", "gamma")


def _floats(h, values) -> None:
    h.update(np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes())


def _refused(refusal) -> str:
    return f"refused|{getattr(refusal, 'reason', '')}|{refusal}"


def case_digests(grid) -> dict[str, str]:
    from ibnr.kernels.cdr import ODPBootstrapDiagonal, simulate_one_year_cdr
    from ibnr.kernels.mack import fit_mack_grid
    from ibnr.kernels.odp_bootstrap import draw_next_increments, fit_odp_bootstrap

    out = {}
    try:
        fit = fit_mack_grid(grid, sigma_rule="log_linear")
        boot = fit_odp_bootstrap(
            fit.cum,
            fit.obs_mask,
            fit.latest_dev,
            fit.f,
            origins=fit.origin_periods,
            dev_grain_months=fit.dev_grain_months,
        )
    except ValueError as refusal:
        return {"fit": _refused(refusal)}
    h = hashlib.sha256()
    for name in ("inc", "fitted", "residuals"):
        _floats(h, getattr(boot, name))
    h.update(np.ascontiguousarray(boot.pool_mask).tobytes())
    _floats(h, [boot.phi, boot.n_cells, boot.n_params])
    out["fit"] = h.hexdigest()[:24]
    for law in LAWS:
        for noise, resample in ARMS:
            h = hashlib.sha256()
            draws = draw_next_increments(
                boot,
                n_draws=2000,
                rng=np.random.default_rng(7),
                process=law,
                process_noise=noise,
                resample_residuals=resample,
            )
            _floats(h, draws)
            out[f"next|{law}|{int(noise)}{int(resample)}"] = h.hexdigest()[:24]
        try:
            result = simulate_one_year_cdr(
                fit, generator=ODPBootstrapDiagonal(process=law), n_draws=3000, seed=11
            )
        except ValueError as refusal:
            out[f"cdr|{law}"] = _refused(refusal)
            continue
        h = hashlib.sha256()
        _floats(h, result.samples)
        out[f"cdr|{law}"] = h.hexdigest()[:24]
    return out


def triangles() -> dict[str, list]:
    return {**conventional.public_triangles(), **conventional.clrd_triangles()}


def pin(cases: dict[str, list]) -> dict[str, dict[str, str]]:
    out = {}
    for name, rows in cases.items():
        try:
            grid = conventional.grid_of(rows)
        except ValueError:
            continue  # not a run-off triangle: nothing to fit
        out[name] = case_digests(grid)
    return out


def main() -> None:
    src = sys.argv[1]
    sys.path.insert(0, src)
    import ibnr

    assert Path(ibnr.__file__).resolve().is_relative_to(Path(src).resolve()), ibnr.__file__
    frozen = pin(triangles())
    OUT.write_text(json.dumps(frozen, sort_keys=True, indent=0) + "\n", "utf-8", newline="\n")
    print(f"wrote {OUT}: {len(frozen)} triangles")


if __name__ == "__main__":
    main()
