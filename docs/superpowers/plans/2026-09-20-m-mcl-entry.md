# Full-matrix multivariate chain ladder, `mcl` (PR M) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `mcl` gallery entry (family `statistical`): the full-matrix multivariate chain ladder, estimated with the R `systemfit` SUR conventions, whose point reserves tie out to the companion study's R implementation on the 82 reconciled Schedule P companies, and which produces a `PredictiveDistribution` like `sur`.

**Architecture:** One directory `src/ibnr/gallery/statistical/mcl/` (`__init__.py`, `model.py`, `card.md`) over the existing `kernels.multiline` contract. Per development transition, each line's next cumulative is regressed on EVERY line's current cumulative (one equation per line, Mack-weighted by the equation's own current cumulative), estimated as a system in ONE feasible-GLS step; thin transitions fall back to the diagonal volume-weighted chain ladder. `point()` is the vector recursion; `predict()` simulates as `sur` does. Two helpers `sur` already has (`nearest_pd`, `mack_tail_variance`) move to `kernels/multiline.py` so both entries share one copy.

**Tech Stack:** numpy, pandas, pytest. No torch.

**Spec:** `docs/superpowers/specs/2026-09-20-tlrn-mcl-point-scores-design.md`, section 5.6. Read it first, and read `src/ibnr/gallery/statistical/sur/model.py` and its `card.md` in full before writing a line: `mcl` is `sur`'s sibling and follows its shape (contract, atomic fit, `cohorts`, `predict` layout through `kernels.multiline`, `realized_ultimates`).

## Global Constraints

- The R reference for every convention is `C:\Users\EthanKang\Projects\transformers_reserving\Code\01_functions.R::fit_mcl_betas` and `fit_mcl_ibnr` (read them; they are 60 lines). That code and its author are NOT named anywhere in this repository. In the card, docstrings, tests and CHANGELOG call it "the companion study's R implementation" or "the reference R implementation".
- `kernels/` never imports `ibnr.gallery`. Torch is not involved.
- Lint: `uv run ruff check . && uv run ruff format .` and `uv run python scripts/lint_md_snippets.py` pass.
- Commit messages and the PR body: conventional prefix, subject plus body, NO attribution lines of any kind.
- Prose rules: plain sentences; never "screen", "panel", "gate", "fingerprint", "membership", "ablation" (any form), "seed noise", "chip", "drain". `align_panel`-style API names may be backticked.
- Refusal tests use `pytest.raises(..., match=...)`. Every new test file is mutation-checked; the PR body lists the mutations tried.
- Branch `feat/mcl-entry`, worktree `C:\Users\EthanKang\Projects\ibnr-wt\mcl`, created from `origin/main` AFTER PR A (`feat/point-scores`) has merged, because the tie-out test reads `mack.point()` and the notebook will feed `mcl.point()` to `reserve_rows`. Run `git log --oneline -15 origin/main` first and confirm `point_scores` is on main.
- Before committing the last task and before opening the PR: `git fetch origin && git log --oneline HEAD..origin/main`; rebase if non-empty; rerun tests.

---

### Task 1: move `nearest_pd` and `mack_tail_variance` to `kernels/multiline.py`

**Files:**
- Modify: `src/ibnr/kernels/multiline.py` (add `EIG_FLOOR`, `nearest_pd`, `mack_tail_variance`, both copied verbatim from `sur/model.py` with the leading underscore dropped)
- Modify: `src/ibnr/gallery/statistical/sur/model.py` (import them; keep `_nearest_pd = nearest_pd` and `_mack_tail_variance = mack_tail_variance` module aliases so `tests/test_sur.py` and any other importer keep working)
- Test: `tests/test_sur.py` must pass unchanged; add `tests/test_multiline.py::test_nearest_pd_floors_eigenvalues_and_symmetrizes` and `::test_mack_tail_variance_rule` (two small closed-form tests: a 2x2 matrix with a negative eigenvalue comes back symmetric with minimum eigenvalue `EIG_FLOOR * scale`; the tail rule on variances `[4, 2]` (d-2 then d-1) returns `min(2^2/4, 2, 4) = 1`).

- [ ] Write the two tests, see them fail on import, move the code, see them and `tests/test_sur.py` pass, commit `refactor: share nearest_pd and the Mack tail rule through kernels.multiline`.

---

### Task 2: the estimator, `fit()` and `point()`

**Files:**
- Create: `src/ibnr/gallery/statistical/mcl/__init__.py`, `model.py`, `card.md` (card in Task 5)
- Modify: `src/ibnr/gallery/statistical/__init__.py` (import `mcl`)
- Test: `tests/test_mcl.py`

**Interfaces:**
- `MCL.fit(triangle, *, loss_field="paid_loss", as_of=None, min_obs_mult=2) -> MCL`
- Fitted state: `contract_` (the `multiline_data` dict), `transitions_` (one dict per transition: `n`, `B` `(K, K)`, `sigma` `(K, K)`, `coef_cov`, `method` in `{"system", "volume_weighted"}`), `pooled_corr_` `(K, K)`, `_loss_field`.
- `MCL.point(segment=None) -> pd.DataFrame`: `multiline_targets(lobs, origins, premium=None)` columns plus `point` (ultimates per (lob, origin), per-lob totals, grand total).
- `MCL.cohorts()`, `MCL.realized_ultimates(full, segment=None)`: as `sur`.

The estimator, per transition `d` (0-based dev index, `d -> d + 1`), with `n` origin pairs (both cells observed; masks are line-aligned by contract) and `K` lines:

```
x_k = C_{k, pairs, d}  (n,)     y_k = C_{k, pairs, d+1}  (n,)
```

1. **System estimate** when `n > min_obs_mult * K` (strict, the R rule `length(obs) > min_obs_mult * n_lines`): whitened equation `k` has response `y_k / sqrt(x_k)` and design columns `x_l / sqrt(x_k)` for `l = 1..K` (`(n, K)`). Step one: OLS per equation, residuals `e_k (n,)`. Step two: residual covariance with `systemfit`'s default `geomean` denominator, `Sigma[k, l] = (e_k . e_l) / sqrt((n - K)(n - K)) = (e_k . e_l) / (n - K)`, not centred. Step three: ONE GLS solve of the stacked system with `Omega = Sigma (x) I_n` - block `(k, m)` of the normal matrix is `Sigma^-1[k, m] * (X_k' X_m)` and block `k` of the right-hand side is `sum_m Sigma^-1[k, m] * (X_k' y_m)`, which is exactly `sur/model.py::_fgls`'s block assembly with `max_iter = 1` and NO eigenvalue repair inside the step. `B[k, l]` is equation `k`'s coefficient on `x_l`. `coef_cov = inverse of the normal matrix` `(K*K, K*K)`, row-major over `(k, l)`. `sigma` for the draws is the covariance of the GLS residuals with the same `(n - K)` denominator, passed through `nearest_pd` (the point is unaffected). If any coefficient is non-finite, or the normal matrix is singular (`LinAlgError`), fall through to 2, recording `method = "volume_weighted"` and `fallback_reason`.
2. **Volume-weighted fallback** otherwise (including the last transition of a square triangle, where `n = 1`): `B = diag(sum(y_k) / sum(x_k))`, exactly the R `diag(map2_dbl(y, x, ~ sum(.x) / sum(.y)))`. `sigma` and `coef_cov` follow `sur._fallback` with the no-intercept design: own whitened residual variances when `n - 1 >= 2`, the Mack tail rule otherwise; cross-line correlation from the pooled `R_bar` of standardised whitened OLS residuals over transitions with `n >= 3` (copy `sur`'s pass 1). `coef_cov` is `(K, K)` diagonal here (variance of each line's own factor, `var_k * (x_k' x_k)^-1` on the whitened scale); store it under the same key and let `method` say which shape it is.

Positivity: `if (cum[mask] <= 0).any(): raise ValueError("non-positive cumulative ...")` with `sur`'s wording. The reference implementation tolerates a zero through its fallbacks; this port refuses, and the card says so (section "Limitations").

`point()`: for each origin `w` with latest observed dev `d0`, `state = C[:, w, d0]`; for `d in range(d0, n_d - 1)`: `state = B_d @ state`; ultimate = `state`. Assemble with `flatten_with_totals` and `multiline_targets`, add the `point` column.

- [ ] **Step 1: Write the failing tests** (`tests/test_mcl.py`)

```python
"""gallery.statistical.mcl: the full-matrix multivariate chain ladder.

Two things are protected here. The estimator follows the R systemfit SUR
conventions (one feasible-GLS step, geomean residual covariance), which a
hand-solved two-line system pins; and the fallback rule by transition (the
system only where origin pairs exceed twice the line count) is what makes
the entry the reference implementation's twin rather than a cousin. The
mart tie-out lives in Task 4.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from ibnr import gallery
from ibnr.gallery.statistical.mcl.model import MCL, system_estimate

from .conftest import make_multiline_triangle
from .test_sur import fit_on_upper, simulate_cl_square  # reuse the CL simulator


def test_system_estimate_matches_a_hand_solved_two_line_system():
    """K = 2, n = 6: OLS per equation, geomean residual covariance, one GLS step.
    The reference values are computed here with dense numpy on the stacked
    system, which is a different code path from the block assembly."""
    rng = np.random.default_rng(0)
    x = rng.uniform(100.0, 200.0, size=(2, 6))
    y = np.stack([1.4 * x[0] + 0.1 * x[1], 0.2 * x[0] + 1.2 * x[1]]) + rng.normal(0, 3.0, (2, 6))
    B, sigma, coef_cov = system_estimate(x, y)
    # dense reference: stacked whitened system
    root = np.sqrt(x)
    X = [np.stack([x[l] / root[k] for l in range(2)], axis=1) for k in range(2)]  # (n, K) each
    Y = [y[k] / root[k] for k in range(2)]
    beta_ols = [np.linalg.lstsq(X[k], Y[k], rcond=None)[0] for k in range(2)]
    e = np.stack([Y[k] - X[k] @ beta_ols[k] for k in range(2)])
    S = e @ e.T / (6 - 2)
    Sinv = np.linalg.inv(S)
    Xs = np.zeros((12, 4))
    Xs[:6, :2], Xs[6:, 2:] = X[0], X[1]
    Ys = np.concatenate(Y)
    Omega_inv = np.kron(Sinv, np.eye(6))
    A = Xs.T @ Omega_inv @ Xs
    beta_gls = np.linalg.solve(A, Xs.T @ Omega_inv @ Ys)
    np.testing.assert_allclose(B.reshape(-1), beta_gls, rtol=1e-10)
    np.testing.assert_allclose(coef_cov, np.linalg.inv(A), rtol=1e-8)
    assert sigma.shape == (2, 2)


def test_zero_cross_terms_collapse_to_the_volume_weighted_chain_ladder(backend_name):
    """Lines that develop independently give a near-diagonal B on the system
    transitions and exactly the volume-weighted factors on the fallback ones."""
    rng = np.random.default_rng(1)
    cum = simulate_cl_square(rng, n_w=12, rho=0.0)
    entry, full = fit_on_upper(backend_name, cum)  # this is SUR's helper: reuse the data only
    tri = full
    mcl = MCL().fit(tri, as_of=dt.date(2000 + 12 - 1, 12, 31))
    methods = [t["method"] for t in mcl.transitions_]
    # 12 origins, K = 2: transition d has 11 - d pairs; system needs > 4 pairs
    assert methods == ["system"] * 7 + ["volume_weighted"] * 4
    for t in mcl.transitions_[7:]:
        assert np.count_nonzero(t["B"] - np.diag(np.diag(t["B"]))) == 0
    # the whole projection stays within a few percent of the per-line chain ladder
    point = mcl.point()
    sur_point = gallery.fit("sur", tri, as_of=dt.date(2011, 12, 31))  # slopes = VW chain ladder
    assert point["point"].iloc[-1] == pytest.approx(
        float(np.mean(sur_point.predict(n_draws=2000, seed=0, param_uncertainty=False).samples[:, -1])),
        rel=0.05,
    )


def test_fallback_rule_by_transition_and_min_obs_mult(backend_name):
    rng = np.random.default_rng(2)
    cum = simulate_cl_square(rng, n_w=8, rho=0.3)
    _, tri = fit_on_upper(backend_name, cum)
    as_of = dt.date(2007, 12, 31)
    default = MCL().fit(tri, as_of=as_of)
    # 8 origins: pairs 7, 6, 5, 4, 3, 2, 1 -> system needs > 4 -> first three only
    assert [t["method"] for t in default.transitions_] == ["system"] * 3 + ["volume_weighted"] * 4
    strict = MCL().fit(tri, as_of=as_of, min_obs_mult=3)
    assert [t["method"] for t in strict.transitions_] == ["system"] + ["volume_weighted"] * 6
    assert [t["n"] for t in default.transitions_] == [7, 6, 5, 4, 3, 2, 1]


def test_point_layout_and_recursion(backend_name):
    rng = np.random.default_rng(3)
    cum = simulate_cl_square(rng, n_w=6, rho=0.2)
    _, tri = fit_on_upper(backend_name, cum)
    entry = MCL().fit(tri, as_of=dt.date(2005, 12, 31))
    frame = entry.point()
    labels = frame["label"].astype(str).tolist()
    assert labels[-1] == "total" and labels[-3:-1] == ["lob_0/total", "lob_1/total"]
    assert len(frame) == 2 * 6 + 2 + 1
    # the grand total is the sum of the per-lob totals, which are sums over origins
    assert frame["point"].iloc[-1] == pytest.approx(frame["point"].iloc[-3:-1].sum())
    # hand recursion for the last origin: latest dev is 0, so every transition applies
    B = [t["B"] for t in entry.transitions_]
    state = entry.contract_["cum"][:, -1, 0]
    for b in B:
        state = b @ state
    np.testing.assert_allclose(frame["point"].to_numpy()[[5, 11]], state, rtol=1e-10)


def test_refusals(backend_name):
    rng = np.random.default_rng(4)
    cum = simulate_cl_square(rng, n_w=6)
    cum[0, 2, 1] = 0.0  # a zero cumulative inside the upper triangle
    lobs = {f"lob_{k}": cum[k] for k in range(2)}
    tri = make_multiline_triangle(backend_name, lobs, start_year=2000)
    with pytest.raises(ValueError, match="non-positive cumulative"):
        MCL().fit(tri, as_of=dt.date(2005, 12, 31))
    with pytest.raises(ValueError, match="min_obs_mult"):
        MCL().fit(tri, as_of=dt.date(2005, 12, 31), min_obs_mult=0)
```

Read `tests/test_sur.py::fit_on_upper` before using it: it returns `(entry, full_triangle)`; only the triangle is reused here. If its signature differs, build the triangle with `make_multiline_triangle` directly.

- [ ] **Step 2: Run to verify they fail**, then **Step 3: Implement** `model.py` (class `MCL`, free functions `system_estimate(x, y) -> (B, sigma, coef_cov)` and `volume_weighted(x, y) -> B`), following `sur/model.py`'s structure line for line where the two agree (contract build, atomic assignment, `cohorts`, `realized_ultimates`). `name = "mcl"`, `family = "statistical"`, no `config_class`.

- [ ] **Step 4: Run the tests to verify they pass**, mutation check (replace the one GLS step with per-equation OLS: the hand-solved test must fail; drop the strict `>` for `>=` in the system rule: the fallback-rule test must fail), commit:

```bash
git commit -m "feat: mcl, the full-matrix multivariate chain ladder

One feasible-GLS step per transition with the systemfit conventions of the
reference R implementation, the diagonal volume-weighted chain ladder where
origin pairs do not exceed twice the line count, and the vector recursion
as the point."
```

---

### Task 3: `predict()` draws

**Files:**
- Modify: `src/ibnr/gallery/statistical/mcl/model.py`
- Test: `tests/test_mcl.py` (append)

Mirror `sur.predict` with these differences: the conditional mean is `einsum("nkl,nl->nk", B_draws, state)`; on a `system` transition parameter risk draws the flat `(K*K,)` vector from `N(vec(B), coef_cov)`; on a `volume_weighted` transition it draws the K diagonal factors from their own variances and keeps the off-diagonal zeros; process noise is `sqrt(max(state, 0)) * (L z)` with `L = chol(sigma)`, floored at zero as `sur` does. Seed derivation through `kernels.rng.cohort_stream(seed, label="predict", cohorts=self.cohorts(), field=self._loss_field)`. `param_uncertainty` keyword as `sur`.

Tests to append:

```python
def test_predict_layout_mean_and_determinism(backend_name):
    rng = np.random.default_rng(5)
    cum = simulate_cl_square(rng, n_w=8, rho=0.3)
    _, tri = fit_on_upper(backend_name, cum)
    entry = MCL().fit(tri, as_of=dt.date(2007, 12, 31))
    pred = entry.predict(n_draws=4000, seed=9)
    assert pred.samples.shape == (4000, 2 * 8 + 2 + 1)
    assert pred.targets["label"].astype(str).tolist() == entry.point()["label"].astype(str).tolist()
    # process-only draws are centred on the point recursion
    only_process = entry.predict(n_draws=4000, seed=9, param_uncertainty=False)
    np.testing.assert_allclose(only_process.mean()[-1], entry.point()["point"].iloc[-1], rtol=0.02)
    again = entry.predict(n_draws=4000, seed=9)
    np.testing.assert_array_equal(pred.samples, again.samples)
    assert not np.array_equal(pred.samples, entry.predict(n_draws=4000, seed=10).samples)
    # the grand total is the within-draw sum of the per-lob totals
    np.testing.assert_allclose(pred.samples[:, -1], pred.samples[:, -3:-1].sum(axis=1))


def test_realized_ultimates_align_to_predict(backend_name):
    rng = np.random.default_rng(6)
    cum = simulate_cl_square(rng, n_w=6, rho=0.1)
    _, tri = fit_on_upper(backend_name, cum)
    entry = MCL().fit(tri, as_of=dt.date(2005, 12, 31))
    realized = entry.realized_ultimates(tri)
    assert realized.shape == (2 * 6 + 2 + 1,)
    np.testing.assert_allclose(realized[:12].reshape(2, 6), cum[:, :, -1])
    scored = entry.evaluate(realized)
    assert set(scored) == {"summary", "percentiles", "crps", "point"}
```

- [ ] Write, fail, implement, pass, mutation check (make the process noise ignore `sigma`'s off-diagonal: the grand-total-within-draw assertion still holds, so instead check that with `rho=0.9` the total's draw variance exceeds the independent sum of line variances by more than 20 percent, and add that as a test), commit `feat: mcl predictive draws with parameter and correlated process risk`.

---

### Task 4: registration, the entry census tests, and the mart tie-out

**Files:**
- Modify: `tests/test_entry_contract.py` (`SINGLE_COHORT` gains `"mcl"`)
- Modify: `tests/test_fit_atomicity.py` (`NAMED_TEST_ENTRIES` gains `"mcl"`, with an atomicity test in `tests/test_mcl.py`: a fit that fails on the positivity guard leaves an already-fitted entry's `transitions_` and `contract_` untouched - copy the `sur` atomicity test's shape)
- Modify: `tests/test_field_annotations.py` / `tests/test_premium_refusal.py` tables if they enumerate entries (run them; `mcl` has no `premium_field`, like `sur` - add it wherever `sur` is listed as premium-free)
- Modify: `README.md` `loss_field` table: `mcl` joins the `"paid_loss"` row; the sentence "Only `mack` and `sur` have no `premium_field` argument at all" becomes "Only `mack`, `mcl` and `sur` ...". Update the count words ("the other eleven" becomes "the other twelve", "All fourteen" stays fourteen).
- Create: `tests/data/tlrn_study_company_reserves.csv` and `tests/data/tlrn_study_pairs.csv`
- Test: `tests/test_mcl_tieout.py`

Vendoring, done once with a short script you run and then delete (do not commit the script):

- Source A: `C:\Users\EthanKang\Projects\transformers_reserving\Outputs\marco_reconciliation\replay_company.csv`. Keep columns `GRCODE, IBNR_CL, IBNR_CL_MCL, IBNR_obs, point_ok`; rename `GRCODE` to `company_code` (as a string, zero-padded exactly as the mart spells it - check one code against `load_schedule_p` before trusting `str(int)`), `IBNR_CL` to `r_chain_ladder_reserve`, `IBNR_CL_MCL` to `r_mcl_reserve`, `IBNR_obs` to `actual_reserve`. 93 rows.
- Source B: `...\selected_input.csv`, distinct `(GRCODE, lob)`; map `lob` through `{"comauto": "commercial_auto", "othliab": "other_liability", "ppauto": "private_passenger_auto", "wkcomp": "workers_compensation"}` to `line_of_business`. 243 rows.
- Add `tests/data/README.md` (or extend it if it exists) with one paragraph: produced 2026-09-20 by replaying the companion study's R implementation on data loaded through `ibnr.data.schedule_p` at publish `20260613_041006`; accident years 1998 to 2007; valuation 2007-12-31; reserves are outstanding paid through 120 months in USD thousands; `point_ok` marks the 82 companies on which every R method returned a finite number.

The tie-out test:

```python
"""mcl and mack against the companion study's R implementation on the Schedule P mart.

Skips unless the pinned gold publish is cached. The 82 companies with
``point_ok`` are the comparison set: on the other 11 the R chain ladder failed
on a line with a zero or negative paid cell, which this repository's contracts
refuse by name, and those refusals are asserted rather than skipped.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ibnr import gallery
from ibnr.data.schedule_p import active_mart_path, load_schedule_p, pinned_source

PUBLISH = "20260613_041006"
SOURCE = pinned_source(f"github://EKtheSage/cas-schedule-p-data-model@{PUBLISH}")
DATA = Path(__file__).parent / "data"
AS_OF = dt.date(2007, 12, 31)
ORIGINS = [dt.date(y, 1, 1) for y in range(1998, 2008)]


def _cached() -> bool:
    try:
        return active_mart_path(SOURCE).exists()
    except Exception:
        return False


pytestmark = [pytest.mark.mart, pytest.mark.skipif(not _cached(), reason=f"publish {PUBLISH} not cached")]


@pytest.fixture(scope="module")
def study():
    companies = pd.read_csv(DATA / "tlrn_study_company_reserves.csv", dtype={"company_code": str})
    pairs = pd.read_csv(DATA / "tlrn_study_pairs.csv", dtype={"company_code": str})
    assert len(companies) == 93 and len(pairs) == 243
    return companies, pairs


def _company_triangle(code: str, lines: list[str]):
    tri = load_schedule_p(SOURCE, companies=[code], lines=lines)
    tri = tri.filter(tri.expr.origin_period.isin(ORIGINS))
    return tri.with_expr(tri.expr.drop("company_name"))


def test_mcl_and_mack_reserves_tie_out_on_the_82_companies(study):
    companies, pairs = study
    scored = companies[companies["point_ok"]]
    assert len(scored) == 82
    rows = []
    for code in scored["company_code"]:
        lines = sorted(pairs.loc[pairs["company_code"] == code, "line_of_business"])
        tri = _company_triangle(code, lines)
        mcl = gallery.fit("mcl", tri, loss_field="paid_loss", as_of=AS_OF)
        frame = mcl.point()
        latest = float(mcl.contract_["cum"][mcl.contract_["obs_mask"]].size and _latest_sum(mcl))
        mcl_reserve = float(frame["point"].iloc[-1]) - latest
        mack_reserve = 0.0
        for line in lines:
            lt = tri.filter(tri.expr.line_of_business == line)
            fit = gallery.fit("mack", lt, loss_field="paid_loss", as_of=AS_OF)
            mack_reserve += float(fit.fit_.reserve.sum())
        rows.append(dict(company_code=code, mcl=mcl_reserve, mack=mack_reserve))
    got = pd.DataFrame(rows).merge(scored, on="company_code")
    np.testing.assert_allclose(got["mack"], got["r_chain_ladder_reserve"], rtol=1e-6, atol=1e-2)
    np.testing.assert_allclose(got["mcl"], got["r_mcl_reserve"], rtol=1e-6, atol=1e-2)


def _latest_sum(entry) -> float:
    """Sum over lines and origins of the latest observed cumulative."""
    c = entry.contract_
    cum, mask = c["cum"], c["obs_mask"]
    total = 0.0
    for k in range(c["n_lob"]):
        for w in range(c["n_w"]):
            devs = np.nonzero(mask[k, w])[0]
            total += float(cum[k, w, devs[-1]])
    return total


def test_the_eleven_excluded_companies_are_refused_by_name(study):
    companies, pairs = study
    for code in companies.loc[~companies["point_ok"], "company_code"]:
        lines = sorted(pairs.loc[pairs["company_code"] == code, "line_of_business"])
        tri = _company_triangle(code, lines)
        with pytest.raises(ValueError, match="non-positive cumulative"):
            gallery.fit("mcl", tri, loss_field="paid_loss", as_of=AS_OF)
```

Tidy `_latest_sum`'s call site (the odd `size and` expression above is a placeholder for you to replace with a plain call). If a refused company is refused for a different named reason than positivity (say a missing line), assert that reason and say so in the PR. If the tie-out disagrees on some companies, do NOT loosen the tolerance: print the per-company differences, look for the convention that explains them (the residual covariance denominator, the strictness of the system rule, the last-transition handling), fix the port, and record what it was in the card.

- [ ] Vendor the two CSVs; add the census entries; write the tie-out test; run `uv run pytest tests/test_mcl_tieout.py -q -rs -m mart` and confirm it EXECUTED (the publish is cached on this machine from the 2026-09-20 reconciliation; if it is not, say so and stop); run `uv run pytest -q -rs`; commit `test: mcl and mack tie out to the reference R reserves on 82 Schedule P companies`.

---

### Task 5: the card, CHANGELOG, docs

`card.md` sections, in `sur/card.md`'s style: Model (the full-matrix regression with the K x K `B_d`, Zhang 2010's general form); Estimation (the systemfit conventions, one GLS step, geomean denominator, the `n > 2K` rule and what it means on a 10 by 10 Schedule P square with four lines: only the first transition is a system estimate; with two lines the first five); Prediction (as `sur`); Data contract; Tie-out (82 companies, tolerance, the file names); Limitations (positivity refusal where the reference tolerates a zero; late transitions are chain ladder; the eigenvalue floor on `sigma`). Cite Zhang (2010) and Henningsen and Hamann (2007), *systemfit: A Package for Estimating Systems of Simultaneous Equations in R*, Journal of Statistical Software 23(4). Describe the reference implementation neutrally.

CHANGELOG `## Unreleased`: one paragraph for the new entry and the tie-out. `great-docs.yml` needs no change (entries are documented through the gallery). `README.md` table updated in Task 4.

- [ ] Commit `docs: mcl model card`, then lint, full suite, rebase, push `feat/mcl-entry`, open the PR (title `feat: mcl, the full-matrix multivariate chain ladder, tied out to the reference R reserves`; body: what it is, the conventions, the tie-out result with the max relative difference over the 82 companies, verification counts, mutation list). Do not merge. Report back: PR URL, files, verification, the tie-out numbers, anything unresolved.
