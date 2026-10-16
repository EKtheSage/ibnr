"""Write ``tests/data/odp_cdr_pin.json``: the one-year CDR's ODP route before the run-off bootstrap.

``tests/test_odp_runoff.py`` recomputes every case with the current code and
requires the same answer, so extending ``kernels/odp_bootstrap.py`` for the
full run-off bootstrap moved no number of the one-year CDR's ODP route:

- ``fit``: ``fit_odp_bootstrap`` with its defaults (the arrays ``inc``,
  ``fitted``, ``residuals`` and ``pool_mask``, and ``phi``, ``n_cells``,
  ``n_params``), fed ``fit_mack_grid``'s factors as the CDR feeds them;
- ``next``: ``draw_next_increments`` for both process laws and the three arms
  (both risk sources, residuals only, process noise only), 2,000 draws from
  ``default_rng(7)``;
- ``cdr``: ``simulate_one_year_cdr(fit, generator=ODPBootstrapDiagonal(...))``
  for both laws, 3,000 draws, seed 11: the samples, digested by their bits.

A triangle the ODP bootstrap refuses is pinned by its reason and message.

Everything is a digest of exact bits except the draws with gamma process noise
(``next|gamma|11``, ``next|gamma|10`` and ``cdr|gamma``, in :data:`SUMS_KEYS`).
numpy's gamma sampler takes ``pow`` and ``log`` for a shape below 1 (230,672
of these draws, where the fitted mean is smaller than the scale), and the math
libraries of Windows, where the pin is frozen, and Linux differ in the last
bits of those functions. Those draws are pinned by their column sums (of ``x``,
``|x|``, ``x ** 2`` and the draw index times ``x``), which the test compares to
1e-12. The fit, the od_poisson draws and the draws with no process noise take
no ``pow``, ``log`` or ``exp`` and stay digests.

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
#: The keys pinned by column sums rather than bits: the draws with gamma noise.
SUMS_KEYS = ("next|gamma|11", "next|gamma|10", "cdr|gamma")


def _floats(h, values) -> None:
    h.update(np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes())


def _sums(values) -> dict:
    """Column sums of an array of draws (draws along the first axis), as numbers."""
    x = np.asarray(values, dtype=float)
    x = x.reshape(x.shape[0], -1)
    index = np.arange(x.shape[0], dtype=float)[:, None]
    return {
        "shape": [int(n) for n in np.shape(values)],
        "sum": x.sum(axis=0).tolist(),
        "sum_abs": np.abs(x).sum(axis=0).tolist(),
        "sum_sq": (x * x).sum(axis=0).tolist(),
        "sum_index": (index * x).sum(axis=0).tolist(),
    }


def _refused(refusal) -> str:
    return f"refused|{getattr(refusal, 'reason', '')}|{refusal}"


def case_digests(grid) -> dict[str, str | dict]:
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
            key = f"next|{law}|{int(noise)}{int(resample)}"
            if key in SUMS_KEYS:
                out[key] = _sums(draws)
                continue
            _floats(h, draws)
            out[key] = h.hexdigest()[:24]
        try:
            result = simulate_one_year_cdr(
                fit, generator=ODPBootstrapDiagonal(process=law), n_draws=3000, seed=11
            )
        except ValueError as refusal:
            out[f"cdr|{law}"] = _refused(refusal)
            continue
        if f"cdr|{law}" in SUMS_KEYS:
            out[f"cdr|{law}"] = _sums(result.samples)
            continue
        h = hashlib.sha256()
        _floats(h, result.samples)
        out[f"cdr|{law}"] = h.hexdigest()[:24]
    return out


def triangles() -> dict[str, list]:
    return {**conventional.public_triangles(), **conventional.clrd_triangles()}


def pin(cases: dict[str, list]) -> dict[str, dict[str, str | dict]]:
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
