# Contract `values`, the feature builder and the `tlrn` entry (PR C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `tlrn` gallery entry (family `nn`) reproducing the companion study's Transformer Loss Reserving Network: engineered cutoff-dependent features over company x accident-year examples of line-by-lag tokens, axial attention, a positive development-factor head with cumulative projection, the point objective, the checkpoint protocol with best-2-of-10 seed selection, and historically calibrated company-level draws.

**Architecture:** Three layers. (1) `kernels.nn_contract` gains a `values` key (raw field grids). (2) `kernels/nn_features.py::tlrn_features` builds the R study's example tensors from a company contract at a cutoff, torch-free. (3) `gallery/nn/tlrn/` holds `config.py`, `network.py` (axial block and the `TLRN` module), `head.py` (factor head, projection, losses, factor support) and `model.py` (the entry: fit through the extended `train_ensemble`, `point`, `predict` through `kernels.residual_calibration`, `realized_ultimates`, `cohorts`). No held-out mixins.

**Tech Stack:** numpy, pandas, torch (inside fit/predict only), pytest.

**Spec:** `docs/superpowers/specs/2026-09-20-tlrn-mcl-point-scores-design.md`, sections 5.4 and 5.5. **Depends on PR B** (`train_ensemble`'s `schedule`, `param_groups`, `min_epochs`, `check_every`, `cutoff_sampling`, `keep`; `warmup_cosine`; `kernels.residual_calibration`) and on PR A (`point_scores` is what the notebook will feed `point()` to). Confirm both are on `origin/main` before starting.

## Global Constraints

- The R reference is `C:\Users\EthanKang\Projects\transformers_reserving\Code\05_transformers.R` (sections 1 to 7: `prepare_arrays`, `lag_stats`, `own_ratios`, `cl_factors`, `origin_cl_logf`, `build_examples`, `axial_block`, `tlrn`, the losses, `training_factor_support`, `learning_rate_multiplier`, `train_checkpoint_tlrn`, `checkpoint_validation`, `predict_cells`, `ibnr_from`, `make_prebuilt`, `fit_checkpoint_seed`) and `06_uncertainty.R` (the backtest and calibration). Read them in full before writing code; every definition below is a transcription of them. Neither the file, its repository nor its author is named anywhere in this repository: write "the companion study's R implementation" or "the reference implementation".
- Torch is imported inside `fit`/`predict` (and inside `network.py`/`head.py`, which are imported from inside those methods only). `ibnr.gallery` must import without torch. `kernels/` never imports the gallery or torch.
- Lint: `uv run ruff check . && uv run ruff format .` and `uv run python scripts/lint_md_snippets.py` pass.
- Commit messages and PR body: conventional prefix, subject plus body, NO attribution lines.
- Prose rules: plain sentences; never "screen", "panel", "gate", "fingerprint", "membership", "ablation" (any form), "seed noise", "chip", "drain". "seed" only in its random-number sense.
- Refusal tests use `pytest.raises(..., match=...)`. Every new test file is mutation-checked; the PR body lists the mutations.
- Branch `feat/tlrn-entry`, worktree `C:\Users\EthanKang\Projects\ibnr-wt\tlrn`, from `origin/main` after PR B merged. `uv sync --extra nn`. Confirm NN tests execute (`-rs`), not skip.
- Before the last commit and before opening the PR: `git fetch origin && git log --oneline HEAD..origin/main`; rebase if non-empty; rerun tests. PR M (`mcl`) edits `README.md`'s loss-field table and `tests/test_entry_contract.py` in parallel; expect a small rebase there.

## Index conventions used throughout

- Contract grids are 0-based: company `c`, line `l`, origin `w`, dev `d`. Calendar diagonal `cal(w, d) = w + d + 1` (1-based; `contract["cal_idx"]`). A cutoff `K` is a 1-based diagonal; a cell is visible when `cal <= K`.
- The R code is 1-based: accident year `a = w + 1`, lag `j = d + 1`. Its `lk = min(K - a + 1, n_j)` is `lk = min(K - w, n_d)` here, a 1-based count of visible lags for that origin; `has_history = (K - w) >= 1`.
- Tokens are line-major: token `t = l * n_d + d`, `n_tok = L * n_d`. An example is one (company, origin): `e = c * n_w + w`, `n_ex = n_c * n_w`.
- Embedding indices are 1-based as in R, with row 0 unused, so the parameter count matches the reference's 14,309: `line_emb` has `L + 1` rows, `lag_emb` `n_d + 1`, `nobs_emb` `n_d + 2` (index `lk + 1`).

---

### Task 1: `values` in the NN contract

**Files:**
- Modify: `src/ibnr/kernels/nn_contract.py` (`nn_data` and `nn_company_data`)
- Test: `tests/test_nn_contract.py` (append)

**Interfaces:**
- `nn_data(...)["values"]`: `(n_c, n_f, n_w, n_d)` float64, the field's raw value as the triangle reported it (cumulative for an increment field, the level itself for a level field), NaN where absent. `nn_company_data(...)["values"]`: `(n_c, L, n_f, n_w, n_d)`, NaN on absent lines.

- [ ] **Step 1: Write the failing tests**

```python
def test_values_carries_the_raw_grid(backend_name):
    """`values` is the cumulative amount (or the level) exactly as reported, NaN
    where the triangle has no cell - not the increment, not zero-filled."""
    cum = np.array([[100.0, 150.0, 175.0], [110.0, 165.0, np.nan], [120.0, np.nan, np.nan]])
    tri = make_multiline_triangle(backend_name, {"a": cum, "b": cum * 2}, premium_by_lob={"a": np.full(3, 1e3), "b": np.full(3, 1e3)}, start_year=2010)
    c = nn_data(tri, loss_field="paid_loss")
    assert c["values"].shape == (2, 1, 3, 3)
    np.testing.assert_array_equal(c["values"][0, 0], cum)  # NaN where absent, equal elsewhere
    # the increment channel is still x: 100, 50, 25 on the first origin
    np.testing.assert_allclose(c["x"][0, 0, 0], [0.1, 0.05, 0.025])
    company = nn_company_data(tri, loss_field="paid_loss")
    assert company["values"].shape == (1, 2, 1, 3, 3)
    np.testing.assert_array_equal(company["values"][0, 1, 0], cum * 2)


def test_values_of_a_level_field_is_the_level(backend_name):
    ...  # build a triangle with a case_reserve field (see test_nn_transformer.featured_triangle),
         # fit nn_data(level_fields=("case_reserve",)) and assert values[:, 1] equals the
         # reported case reserve grid and x[:, 1] equals values[:, 1] / premium where present
```

Write the second test out in full using `featured_triangle` from `tests/test_nn_transformer.py` (import it) or a local builder; the assertion is the one in the comment.

- [ ] **Step 2: Fail, Step 3: Implement** - in `nn_data`, append `cum` (the per-cohort `(n_f, n_w, n_d)` grid built before differencing) to a `values_list` and return `"values": np.stack(values_list)`; in `nn_company_data`, allocate `np.full((n_c, n_l, n_f, n_w, n_d), np.nan)` and scatter `flat["values"][k]` beside `x`. Document the key in both module-docstring contract lists. **Step 4: Pass**, run `tests/test_nn_contract.py tests/test_nn_transformer.py tests/test_nn_transformer_ml.py -q -rs`, commit `feat: the NN contract carries the raw field grids as values`.

---

### Task 2: `kernels/nn_features.py`

**Files:**
- Create: `src/ibnr/kernels/nn_features.py`
- Test: `tests/test_nn_features.py`

**Interfaces:**

```python
def pooled_cl_factors(values_paid, cutoff) -> np.ndarray          # (L, n_d - 1), floored at 1 + 1e-6
def origin_cl_log_factors(values_paid, written, cutoff, pooled_tail_one) -> np.ndarray  # (n_c, L, n_d - 1)
def tlrn_features(contract, *, cutoff, target_lo, target_hi, incurred_field=None, case_field=None, clamp=5.0) -> dict
```

`contract` is a `nn_company_data` dict whose channel 0 is the paid target; with both `incurred_field` and `case_field` given, they name the increment channel and the level channel of the contract's `fields` (refuse if only one is given, or if a named field is not a channel, or if `case_field` is not a level channel). Returned keys (numpy, no torch):

| key | shape | meaning |
|---|---|---|
| `feat` | `(n_ex, n_tok, n_feat)` float | 8 or 13 features, order below |
| `target` | `(n_ex, n_tok)` | paid increment ratio (0 where unusable) |
| `target_mask` | `(n_ex, n_tok)` float 0/1 | `target_lo <= cal <= target_hi`, written line, has history |
| `premium` | `(n_ex, n_tok)` | origin premium of the token's line (0 where none) |
| `c_lk`, `anchor_start`, `p_lk` | `(n_ex, L)` | see below |
| `lk` | `(n_ex,)` int | 1-based visible lags, clamped to `[1, n_d]` |
| `has_history` | `(n_ex,)` bool | `K - w >= 1` |
| `written` | `(n_ex, L)` bool | the company's `line_mask` |
| `line_ix`, `lag_ix` | `(n_tok,)` int | 1-based |
| `fallback_logf` | `(L, n_d - 1)` | log pooled factors, 0 at steps `j >= K` |
| `anchor_logf` | `(n_ex, L, n_d - 1)` | per-company as-of chain ladder log factors |
| `example_company`, `example_origin` | `(n_ex,)` int | the example's `(c, w)` |
| `n_dropped` | int | target cells of examples without history |
| `feature_names` | tuple[str, ...] | |
| `n_l`, `n_d`, `n_feat` | int | |

Definitions (transcribed; `nan_where_unusable(x, x_obs)` means `np.where(x_obs, x, nan)`):

- `Y[c, l, w, d]` = paid increment ratio: `nan_where_unusable(x[:, :, 0], x_obs[:, :, 0])`; `Yf` = the same with NaN as 0 (this is the contract's `x[:, :, 0]`).
- `CP` = `values[:, :, 0]` (cumulative paid, NaN absent); `CLP = CP / premium[..., None]`.
- With incurred and case: `CI = values[:, :, i_inc]` with `CI[CI <= 0] = NaN`; `YI = nan_where_unusable(x[:, :, i_inc], x_obs[:, :, i_inc])`; `CLI = CI / premium[..., None]`; `CLC = values[:, :, i_case] / premium[..., None]`.
- `lag_stats(arr, K)`: for each `(l, d)`, the finite values of `arr[:, l, w, d]` over all companies and `w <= K - d - 1`; `(mean, sd)` if at least two values and `sd > 1e-8`, else `(0, 1)`... with the R nuance: when at least two values exist, the mean is used even if the sd is degenerate, and the sd falls back to 1. `st = lag_stats(Y, K)`, `st_i = lag_stats(YI, K)`.
- `own_ratios(K)`: `R[c, l, d] = clip(sum(CP[c, l, w, d] - CP[c, l, w, d - 1]) / sum(CP[c, l, w, d - 1]), -0.5, 3)` over `w <= K - d - 1` with both cells finite and the denominator cell `> 0`; `0` when the ratio is not finite; `R[..., 0] = 0`.
- `ctx_p[c, l, d]` = mean over visible `w` (`cal(w, d) <= K`) of `Y[c, l, w, d]` ignoring NaN, `0` if not finite; `ctx_i` likewise on `YI`.
- Per `(c, w)` with `lk = min(K - w, n_d)` and history: `cum_lr[c, l, w] = clip(CLP[c, l, w, lk - 1], 0, 5)` (NaN as 0); `cum_amt = max(CP[c, l, w, lk - 1], 0)` (NaN as 0); `pti = clip(CP / CI, 0, 2)` at `lk - 1` (not finite: 1); `clir = clip(CLI, 0, 5)` (NaN as 0); `clcr = clip(CLC, -1, 3)` (NaN as 0). Without history the defaults are `0, 0, 1, 0, 0`.
- `vec_stats(arr)`: `(mean, sd)` over all finite entries, `(0, 1)` with fewer than two, sd fallback 1 when `sd <= 1e-8`. Applied to `R`, `cum_lr`, `pti`, `clir`, `clcr`.
- `z(m, mu, sd) = clip((m - mu) / sd, -clamp, clamp)`.
- Features per example `(c, w)`, each an `(L, n_d)` matrix flattened line-major (`obs_m[l, d] = cal(w, d) <= K`; `w_m[l, d] = written[c, l]`):
  1. `z(Yf[c, :, w, :], st) * obs_m`
  2. `obs_m`
  3. `w_m`
  4. `z(ctx_p[c], st)`
  5. `z(R[c], s_r)`
  6. `z(cum_lr[c, :, w], s_clr)` repeated over lags
  7. `lk / n_d` (constant)
  8. `max(j - lk, 0) / n_d` with `j = d + 1`, repeated over lines
  9. `z(ctx_i[c], st_i)` (13-feature form only)
  10. `z(YIf[c, :, w, :], st_i) * obs_m`
  11. `z(pti[c, :, w], s_pti)` repeated
  12. `z(clir[c, :, w], s_cli)` repeated
  13. `z(clcr[c, :, w], s_clc)` repeated
  Non-finite entries become 0 after assembly.
- `target[e] = Yf[c, :, w, :]` flattened; `target_mask = tgt_m * w_m * has_history` with `tgt_m[l, d] = target_lo <= cal(w, d) <= target_hi`; `n_dropped += sum(tgt_m * w_m)` for examples without history.
- `premium[e, t] = premium[c, l, w]` (NaN as 0). `c_lk[e, l] = max(cum_amt[c, l, w], 1)` with history else 1; `anchor_start[e, l] = CP[c, l, w, lk - 1]` (NaN as 0) with history else 0; `p_lk[e, l] = max(premium[c, l, w], 1e-8)` with history else `1e-8`.
- `pooled_cl_factors(CP, K)`: for `l`, `d in 0..n_d-2`: origins `w <= K - (d + 1) - 1` (R's `amax = K - j`); `f = nansum(CP[:, l, w, d + 1]) / nansum(CP[:, l, w, d])` over all companies if the denominator `> 0`, else 1; then `f = max(f, 1 + 1e-6)`. `fallback_logf = log(f)` with `fallback_logf[:, d] = 0` for `d + 1 >= K`.
- `origin_cl_log_factors(CP, written, K, pooled_tail_one)`: `pooled_tail_one` is the pooled factor matrix with steps `d + 1 >= K` set to 1; for each written `(c, l)` and `d` with `amax = min(K - (d + 1), n_w) >= 1`: `before = CP[c, l, :amax, d]`, `after = CP[c, l, :amax, d + 1]`, both finite; `ratio = sum(after) / sum(before)` if any pair and `sum(before) > 0`; `f = ratio` if finite and `> 0` else `pooled_tail_one[l, d]`; unwritten lines keep 1. Log, then broadcast to `(n_ex, L, n_d - 1)` by repeating each company `n_w` times.

- [ ] **Step 1: Write the failing tests** (`tests/test_nn_features.py`)

Fixture: a builder `study_triangle(backend_name, n_companies=3, n_w=6, seed=0)` writing the long frame directly (fields `paid_loss`, `incurred_loss`, `case_reserve`, `earned_premium`) for 3 companies x 2 lines, full 6x6 squares, `START = 2000`, with `incurred >= paid > 0` and `case = incurred - paid` so the level is a real balance. Emit every field at every cell; premium constant per (line, origin). Company codes "0001".."0003". `as_of = 2005-12-31` gives `c_max = 6`.

Tests, each with concrete expected values computed in the test from the same arrays by a SEPARATE, simple implementation (loops over the long frame, not a copy of the module):

1. `test_shapes_and_feature_order`: 8 features without incurred, 13 with; `feat.shape == (18, 12, n_feat)`; `feature_names` equals the ordered list above; `n_tok == 12`, `lk` for `(c, w)` at `K = 4` equals `min(4 - w, 6)` clipped to `>= 1` and `has_history` is `w <= 2`.
2. `test_visibility_and_masks_at_a_cutoff`: at `K = 4`, feature 2 equals `cal <= 4` for every example; `target_mask` with `target_lo = 5, target_hi = 6` is 1 exactly on visible-later cells of written lines for examples with history, and `n_dropped` counts the rest.
3. `test_standardisation_uses_visible_cells_only`: compute `lag_stats` by hand for one `(l, d)` from cells with `cal <= K`; check feature 1 at a visible cell equals the clipped z-score; feature 4 equals the z-scored per-lag context mean.
4. `test_own_ratio_and_latest_balances`: hand-compute `R[c, l, d]` and `cum_lr`, `pti`, `clir`, `clcr` for one company at `K = 4` and compare features 5, 6, 11, 12, 13 after standardisation with `vec_stats` computed by hand.
5. `test_pooled_factors_and_fallback_tail`: `pooled_cl_factors` at `K = 6` equals the pooled volume-weighted factors computed by hand; `fallback_logf` is 0 at steps `d + 1 >= K` and positive elsewhere.
6. `test_anchor_factors_fall_back_to_pooled_on_thin_lines`: zero out one company's line (unwritten) and check its `anchor_logf` row is `log(pooled_tail_one)`.
7. `test_hidden_cells_do_not_reach_the_inputs`: build at `K = 4`, then multiply every `values`/`x` entry with `cal > 4` by 1.37 and add 0.123 (keep `x_obs` unchanged), rebuild; every array in the returned dict except `target` and `n_dropped` must be equal (`np.array_equal`, NaN-aware where relevant); `target` must differ.
8. `test_refusals`: `incurred_field` without `case_field` -> `ValueError` match "both"; a field not in `contract["fields"]` -> match "not a channel"; `case_field` naming an increment channel -> match "level"; `cutoff` outside `[1, n_w + n_d - 1]` -> match "cutoff"; `target_lo <= cutoff` -> match "target_lo"; a company with a hole inside its visible region (delete one interior paid cell before building the contract) -> match "hole".

- [ ] **Step 2: Fail, Step 3: Implement**, vectorised with numpy where natural (loops over `(l, d)` for the statistics are fine at these sizes: 4 lines x 10 lags). Keep every clip constant as a module-level named constant with the R value beside it. **Step 4: Pass, mutation checks** (make feature 1 read `Yf` without `obs_m`: test 2 or 3 must fail; compute `lag_stats` over all cells: test 3 must fail; drop the `+ 1e-6` floor: test 5 must fail), commit `feat: tlrn_features builds the reference study's example tensors from a company contract`.

---

### Task 3: `network.py` and `head.py`

**Files:**
- Create: `src/ibnr/gallery/nn/tlrn/__init__.py`, `network.py`, `head.py`
- Test: `tests/test_tlrn.py` (create; network and head sections)

**Interfaces:**

`network.py`:

```python
class AxialBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout, n_lines, n_lag, cross_line=True): ...
    def forward(self, x, written, need_weights=False) -> tuple[Tensor, Tensor | None, Tensor | None]
        # x (B, T, d); written (B, L) bool
class TLRN(nn.Module):
    def __init__(self, cfg, *, n_lines, n_lag, n_feat): ...
    def forward(self, feat, c_lk, p_lk, lk, line_ix, lag_ix, written, *, use_residual=True,
                need_weights=False, factor_support=None, fallback_logf=None, anchor_logf=None,
                anchor_start=None) -> dict   # pred (B, T), logf (B, L, D-1), C (B, L, D), attn_line, attn_lag
```

`AxialBlock.forward`: `h = ln1(x)` reshaped `(B, L, D, d)` then `(B*D, L, d)` (lines within each lag); `attn_line(h, h, h, key_padding_mask=~written repeated per lag (B*D, L), need_weights=need_weights)`; residual add with dropout; `h = ln2(x)` reshaped `(B*L, D, d)` (lags within each line); `attn_lag`; residual; `x = x + ff(ln3(x))` with `ff = Linear(d, 2d), GELU, Dropout, Linear(2d, d)`. `nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)`. With `cross_line=False` the line attention is skipped (module still constructed so the count is unchanged; document). Refuse an example with no written line (all-padded attention gives NaN).

`TLRN.__init__`: `inp = Linear(n_feat, d)`; `line_emb = Embedding(L + 1, d)`, `lag_emb = Embedding(n_lag + 1, d)`, `nobs_emb = Embedding(n_lag + 2, d)`; `blocks = ModuleList([AxialBlock(...) for _ in range(cfg.n_layers)])`; `ln = LayerNorm(d)`; `head = Linear(d, 1)` with `weight *= 0.01` and `bias = 0` (weight zeroed when `cfg.cl_anchor`); `phi = Parameter((L, n_lag - 1))` initialised to `log(exp(init_step / m) - 1)` at step `m = 1..n_lag-1` (so `softplus(phi) = init_step / m`); `initial_logf` buffer with the same `init_step / m` values (non-persistent).

`TLRN.forward`: with `use_residual and not cfg.factors_only`: `h = inp(feat) + line_emb(line_ix) + lag_emb(lag_ix) + nobs_emb(lk + 1)[:, None, :]`; through the blocks; `net = head(ln(h)).squeeze(-1).reshape(B, L, D)[:, :, :D-1]`; `logf = softplus(phi + eps * net)`; else `logf = softplus(phi)` expanded to `(B, L, D-1)`. Then `head.anchor` when `cfg.cl_anchor`, `head.apply_support` when `factor_support` is given, `head.project`.

`head.py` (free functions over tensors):

```python
def log_factors(phi, net, eps) -> Tensor                      # softplus(phi + eps * net); net None -> softplus(phi)
def anchor(logf, anchor_logf, initial_logf, width) -> Tensor  # anchor_logf + width * tanh((logf - initial) / width)
def apply_support(logf, support, fallback) -> Tensor          # where(support[None], logf, fallback)
def project(logf, c_lk, p_lk, lk, *, anchor_start=None) -> tuple[Tensor, Tensor]   # (pred (B, T), C (B, L, D))
def masked_mse(pred, targ, mask, denominator=None) -> Tensor
def pool_pe_loss(pred, targ, mask, prem, denominator=None) -> Tensor
def ay_line_ape_loss(pred, targ, mask, prem, n_l, n_d, denominator=None) -> Tensor
def point_loss(pred, targ, mask, prem, n_l, n_d, *, w_pe, w_mse, mse_scale) -> Tensor
def factor_support(target_masks, lks, n_l, n_d) -> np.ndarray   # (L, D-1) bool, numpy, from every training set
```

`project`, in 0-based torch with 1-based `lk (B,)`:

```python
B, L, Dm1 = logf.shape
D = Dm1 + 1
zero = logf.new_zeros(B, L, 1)
S = torch.cat([zero, logf.cumsum(dim=2)], dim=2)                       # S[..., j] = sum_{m < j} logf_m
S_lk = S.gather(2, (lk - 1).view(B, 1, 1).expand(B, L, 1))             # sum up to lag lk (1-based)
j = torch.arange(D, device=logf.device).view(1, 1, D)
fwd = (j >= lk.view(B, 1, 1)).to(logf.dtype)                           # cells past the latest visible lag
if anchor_start is None:
    C = c_lk.unsqueeze(2) * (1 - fwd) + torch.exp(c_lk.log().unsqueeze(2) + S - S_lk) * fwd
else:
    C = anchor_start.unsqueeze(2) * ((1 - fwd) + torch.exp(S - S_lk) * fwd)
C_prev = torch.cat([torch.zeros_like(C[:, :, :1]), C[:, :, :-1]], dim=2)
pred = ((C - C_prev) / p_lk.unsqueeze(2)).reshape(B, L * D)
return pred, C
```

Losses (the checkpoint protocol's minibatch denominators when `denominator is None`):

```python
def ay_line_ape_loss(pred, targ, mask, prem, n_l, n_d, denominator=None):
    e = ((pred - targ) * mask * prem).reshape(-1, n_l, n_d).sum(2)
    a = (targ * mask * prem).reshape(-1, n_l, n_d).sum(2)
    den = a.abs().sum() if denominator is None else denominator
    return e.abs().sum() / (den + 1e-8)

def pool_pe_loss(pred, targ, mask, prem, denominator=None):
    e = ((pred - targ) * mask * prem).sum()
    a = (targ * mask * prem).sum()
    den = a.abs() if denominator is None else denominator
    return (e / (den + 1e-8)).abs()

def masked_mse(pred, targ, mask, denominator=None):
    n = mask.sum() if denominator is None else denominator
    if float(n) == 0:
        return pred.new_zeros(())
    return (((pred - targ) ** 2) * mask).sum() / n

def point_loss(...):
    return ay_line_ape_loss(...) + w_pe * pool_pe_loss(...) + w_mse * masked_mse(...) / mse_scale
```

`factor_support(target_masks, lks, n_l, n_d)`: over every training set: `support[l, d] |= any example with lk <= d + 1 and target_mask on line l at lags with 0-based index > d`.

Tests (network and head sections of `tests/test_tlrn.py`):

1. `test_parameter_count_matches_the_reference`: `TLRN(cfg(d_model=32, n_heads=2, n_layers=1), n_lines=4, n_lag=10, n_feat=13)` has 14,309 parameters; `n_feat=8` has 14,149.
2. `test_project_matches_a_hand_chain_ladder`: random `logf (1, 2, 5)`, `c_lk`, `p_lk`, `lk = [3]`: `C[:, :, :3]` equals `c_lk`; `C[:, :, 3] = c_lk * exp(logf[:, :, 2])`; `C[:, :, 5] = c_lk * exp(logf[..., 2:5].sum())`; `pred` at token `(l, d=3)` equals `(C[l, 3] - C[l, 2]) / p_lk[l]`.
3. `test_head_reproduces_the_pooled_chain_ladder`: the R unit test. Build the fixture contract (Task 2's builder), features at `K = 6` (`target_lo = 7, target_hi = 11`); `f = exp(fallback_logf)` is NOT the right input here because its tail is 1; use `pooled_cl_factors(values_paid, 6)` and set `phi = log(exp(log f) - 1)`; forward with `use_residual=False`; company-line reserve = sum over target-masked tokens of `pred * premium`; compare with a direct chain ladder computed in the test from each origin's latest visible cumulative (floored at 1) and the same pooled factors: `max relative difference < 1e-4` (float32).
4. `test_factor_support_substitutes_the_fallback`: with `factor_support` False at the last step, `logf[:, :, -1]` equals `fallback_logf[:, -1]` for every example.
5. `test_anchor_keeps_zero_correction_at_the_chain_ladder`: with `cl_anchor`, zeroed head, `logf` equals `anchor_logf` when `phi` is at its initial value (tanh(0) = 0).
6. `test_losses_closed_forms`: two lines x three lags, hand-chosen numbers, check `ay_line_ape_loss` sums within (example, line) before the absolute value, `pool_pe_loss` is the absolute pooled ratio, `masked_mse` divides by the mask count, and `point_loss` composes with the weights.
7. `test_attention_never_reads_an_unwritten_line`: perturb an unwritten line's tokens; written lines' outputs unchanged (`key_padding_mask` works); and refuse an example with no written line by name.

- [ ] Write, fail, implement, pass, mutation checks (drop `key_padding_mask`: test 7 fails; use `lk` instead of `lk - 1` in the gather: tests 2 and 3 fail; drop the head weight scaling: no test can see it, so add an assertion in test 1 that `head.weight.abs().max() < 0.01` after construction), commit `feat: tlrn network and factor head`.

---

### Task 4: `config.py` and `model.py`

**Files:**
- Create: `src/ibnr/gallery/nn/tlrn/config.py`, `model.py`
- Test: `tests/test_tlrn.py` (entry section)

**Config** (`TLRNConfig`, dataclass, defaults = the reference checkpoint protocol):

```
network:      d_model=32, n_heads=2, n_layers=1, dropout=0.3, eps=0.5, init_step=0.57,
              cross_line=True, factors_only=False, cl_anchor=False, anchor_width=0.1,
              tail_policy="observed_cl"   # or "legacy_init": no support fallback
loss:         w_pe=0.5, w_mse=0.1, mse_scale=0.005
optimisation: lr=3e-3, lr_phi=3e-2, weight_decay=0.0, warmup=20, batch_size=64,
              max_epochs=3000, min_epochs=500, check_every=5, patience=600, grad_clip=1.0
data:         min_cutoff=2, val_diagonals=2, cutoff_sampling="per_epoch"
ensemble:     ensemble_size=10, keep=2
uncertainty:  calibration_cutoffs=(5, 6, 7, 8, 9), calibration_horizons=(3, 4, 5),
              n_strata=4, min_per_stratum=40, n_draws=4000
```

`__post_init__` refuses: `keep` outside `[1, ensemble_size]`; `patience < 1`; `check_every < 1`; `min_epochs > max_epochs`; `val_diagonals < 1`; `tail_policy` and `cutoff_sampling` outside their vocabularies; `n_heads` not dividing `d_model`. `patience` is counted in validation checks, as the loop counts it; the docstring says the default never fires inside 3000 epochs, which is what the reference run did, and that `patience=120` is the 600-epoch early-stopping variant the notebook also reports.

**Entry** (`TLRN(GalleryEntry)`, `name = "tlrn"`, `family = "nn"`, `config_class = TLRNConfig`; no held-out mixins):

```python
def fit(self, triangle, *, loss_field="paid_loss", incurred_field=None, case_field=None,
        premium_field="earned_premium", as_of=None, config=None, device=None, seed=None,
        show_progress=False) -> TLRN
```

Why `incurred_field`/`case_field` and not `feature_fields`: the features are engineered by ROLE (paid, incurred, case), not per channel, so the entry names the roles; the docstring says so and refuses one without the other. Contract: `nn_company_data(train, loss_field=loss_field, feature_fields=(incurred_field, case_field) or (), level_fields=(case_field,) or (), premium_field=premium_field)`.

Fit, in order (build everything, assign at the end, as every sibling does):

1. `obs_any = contract["obs_mask"].any(axis=1)`; `_, _, train_end = splits(obs_any, cal_idx, cfg.val_diagonals)` gives `c_max = train_end + val_diagonals`. Refuse if `train_end - 1 < cfg.min_cutoff` (nothing to train on).
2. Training feature sets: `{K: tlrn_features(contract, cutoff=K, target_lo=K + 1, target_hi=train_end, ...) for K in range(cfg.min_cutoff, train_end)}`. Validation sets: `K in range(train_end, c_max)` with `target_lo=K + 1, target_hi=c_max`. Final set: `K = c_max`, `target_lo = c_max + 1`, `target_hi = n_w + n_d - 1`. Calibration sets: each `K in cfg.calibration_cutoffs` with `target_lo = K + 1, target_hi = c_max` (refuse a calibration cutoff `>= c_max` or `< 1`).
3. Tensors: per set, float32 `feat, target, target_mask, premium, c_lk, anchor_start, p_lk, fallback_logf, anchor_logf`; long `lk, line_ix, lag_ix`; bool `written`. `factor_support = head.factor_support(...)` over the training sets when `tail_policy == "observed_cl"`, else `None`.
4. `make_model`: `TLRN(cfg, n_lines=L, n_lag=n_d, n_feat=n_feat).to(device)`.
5. `train_loss(model, idx, cutoffs)`: `K = int(cutoffs[0])` (all equal under `per_epoch`; refuse if not all equal, naming `cutoff_sampling`); the set for `K`; batch rows `idx`; forward with `factor_support`, the set's `fallback_logf`/`anchor_logf`/`anchor_start`; `point_loss` with minibatch denominators; return `None` when the batch's `target_mask` sums to 0.
6. `val_loss(model)`: the AY/line APE summed over both validation sets (numerator and denominator accumulated across sets, then divided), the reference's `checkpoint_validation`.
7. `train_ensemble(n_ex, config=cfg, seed=seed, make_model=..., train_loss=..., val_loss=..., min_cutoff=cfg.min_cutoff, val_cutoff=train_end, device=dev, schedule=warmup_cosine(cfg.max_epochs, cfg.warmup), param_groups=lambda m: [{"params": [m.phi], "lr": cfg.lr_phi}, {"params": [p for n, p in m.named_parameters() if n != "phi"]}], min_epochs=cfg.min_epochs, check_every=cfg.check_every, cutoff_sampling=cfg.cutoff_sampling, keep=cfg.keep)`. `config.lr` is the network rate; `phi` gets `lr_phi`.
8. `selection_`: a DataFrame with one row per TRAINED member (`member`, `best_epoch`, `epochs_run`, `validation_ay_line_ape`, `validation_company_ape`, `kept`), the reference's seed table. Compute `validation_company_ape` (the reference's `val_pool_ape`: company-level APE over the validation sets) for every member before `keep` drops any - so call `train_ensemble` with `keep=None`, compute the table, and select the `cfg.keep` members yourself by `(validation_ay_line_ape, member)`; this keeps the table complete. (Then `train_ensemble`'s `keep` is unused by this entry; say so in a comment rather than passing it inertly.)
9. Point at the final set: mean over kept members of `pred`; reserve per `(c, l, w)` = `sum_d pred * target_mask * premium` over the example's tokens of line `l`; `ultimate = latest_cum[c, l, w] + reserve`. Store `point_ultimates_ (n_c, L, n_w)` and `point_reserves_`.
10. Calibration: `forecast_at(K)` = company totals of `pred * mask * premium` at set `K` (mean over kept members), `actual_at(K)` = company totals of `target * mask * premium`; `size = company premium = nansum(premium[c])` over written lines and origins; `rolling_residuals(..., cutoffs=cfg.calibration_cutoffs, n_periods=n_w, size=size)` then `calibrate(..., horizons=cfg.calibration_horizons, n_strata=cfg.n_strata, min_per_stratum=cfg.min_per_stratum)`. Store `calibration_`, `company_size_`, `backtest_` (the residual frame).
11. Assign: `contract_`, `config_`, `models_` (kept), `history_` (kept), `selection_`, `factor_support_`, `norm_`? (none: this entry standardises inside the feature builder; store the final set's statistics as `feature_stats_` for the notebook's inspection), `_loss_field`, `_device`, `point_*`, `calibration_`, `company_size_`, `backtest_`.

`cohorts()`: `cohort_identities(contract_, key="companies")`.

`point(segment=None) -> pd.DataFrame`: for one company, `multiline_targets(present lobs, origin_periods, premium=premium[ci, present])` plus `point = flatten_with_totals(point_ultimates_[ci, present])`; for `None`, one row per (company, line, origin) with the company columns, no totals.

`predict(segment=None, n_draws=None, seed=None) -> PredictiveDistribution`: `n_draws = n_draws or cfg.n_draws`; `rng = default_rng(cohort_stream(seed, label="predict", cohorts=self.cohorts(), field=self._loss_field))`; reserve points per company = `point_reserves_` summed over written lines and origins; `draws = calibrated_draws(calibration_, point=reserves, size=company_size_, n_draws=n_draws, rng=rng)`; ultimates = `draws + anchor` with `anchor = latest_cum` summed per company; targets: `[{"label": "total", "premium": size}]` for one company; one row per company (company columns, `label = "total"`) for `None`.

`realized_ultimates(full_triangle, segment=None)`: the ML entry's lookup at `dev_lag = n_d * dev_grain_months` over the training origins and present lines, summed to one number per company; `(1,)` for a segment, `(n_c,)` for `None`.

Also expose `predict_reserve_draws(point, seed=None, n_draws=None)`: draws around ANY company-level point vector (the notebook draws around the blended point, which is what the reference study calibrated). One-line wrapper over `calibrated_draws` with the entry's calibration and sizes.

Tests (entry section):

Fixture `TINY = TLRNConfig(d_model=8, n_heads=1, n_layers=1, dropout=0.0, max_epochs=3, min_epochs=0, check_every=1, patience=5, batch_size=8, ensemble_size=2, keep=1, min_cutoff=2, val_diagonals=2, calibration_cutoffs=(3, 4, 5), calibration_horizons=(1, 2, 3), n_strata=1, min_per_stratum=3, n_draws=50, warmup=1)` on the Task 2 fixture (3 companies, `c_max = 6`).

1. `test_fit_selection_and_history`: `entry.selection_` has 2 rows, one `kept`; `history_` and `models_` have length 1; `factor_support_` shape `(2, 5)`.
2. `test_point_layout_and_reserve_identity`: `point(segment)` has `2 * 6 + 2 + 1` rows with labels from `multiline_targets`; grand total equals the sum of per-lob totals; `point()` has 36 rows; `point_ultimates_ - latest_cum` equals `point_reserves_`.
3. `test_predict_is_calibrated_draws_around_the_point`: `predict(segment)` has one target labelled `total`, 50 draws; draws' median is within a few percent of the point ultimate (the pools are median-centred); same seed twice identical; different seeds differ; `predict()` has 3 targets.
4. `test_realized_ultimates_align`: `(1,)` for a segment, equal to the sum over the company's lines and origins of the dev-6 cumulative; `(3,)` for `None`; `evaluate(realized, segment)` runs and carries the four keys.
5. `test_seed_determinism`: two fits with `seed=5` give identical `point_ultimates_`; `seed=6` differs.
6. `test_thirteen_feature_form`: `fit(incurred_field="incurred_loss", case_field="case_reserve")` -> `models_[0].inp.in_features == 13`; paid-only -> 8; `incurred_field` alone -> `ValueError` match "both".
7. `test_refusals`: `TLRNConfig(keep=3, ensemble_size=2)` match "keep"; `val_diagonals=0` match "val_diagonals"; `calibration_cutoffs=(9,)` on the 6-diagonal fixture match "calibration cutoff"; `premium_field=None` refused by `nn_data` (already covered by `tests/test_premium_refusal.py` once the entry joins its table).
8. `test_no_leak_through_the_entry`: fit on the fixture with `as_of = 2005-12-31`, then fit on a copy where every cell with `eval_date > 2005-12-31` is multiplied by 1.37 and shifted by 0.123 (the `as_of` slice is identical); `point_ultimates_` must be bit-identical.

- [ ] Write, fail, implement, pass, mutation checks (feed the validation sets into training targets: test 8 still passes because the perturbation is beyond `as_of`, so ALSO perturb the last training diagonal in a second fit and assert the point CHANGES - the model must see its own training data; drop `keep` selection so both members are kept: test 1 fails; forget to centre draws: test 3's median check fails), commit `feat: the tlrn entry - checkpoint protocol, seed selection, calibrated company draws`.

---

### Task 5: registration, census tests, parameter pin, CI, README, card

**Files:**
- Modify: `src/ibnr/gallery/__init__.py` (import `tlrn` beside the other NN entries)
- Modify: `tests/test_entry_contract.py` (`POOLED` gains `"tlrn"`)
- Modify: `tests/test_fit_atomicity.py` (`NN_ENTRIES` gains `("tlrn", "ibnr.gallery.nn.tlrn.model.train_ensemble")`; the fit kwargs table gains `"tlrn": dict(loss_field="paid_loss")`; read how the fixture triangle is built there - `nn_transformer_ml` already needs company cohorts, so the same triangle serves)
- Modify: `tests/test_nn_parameter_counts.py` (`DISCLOSED["tlrn"] = (Claim("the tlrn network at d_model 32 with 13 features on 4 lines and 10 lags", lambda: _tlrn()),)` with a `_tlrn()` builder; the card carries `**14,309 parameters**` once)
- Modify: `tests/test_field_annotations.py` / `tests/test_premium_refusal.py` tables (add `tlrn` wherever the six NN entries are listed as refusing `premium_field=None` through `nn_data`)
- Modify: `tests/test_nn_segment_contract.py` only if it enumerates entries (it does not; leave it)
- Modify: `.github/workflows/test.yml`: add `tests/test_tlrn.py` and `tests/test_nn_features.py` to the `nn` leg's `required_files`, and `tests/test_tlrn.py` to the `all` leg's
- Modify: `README.md` loss-field table: `tlrn` joins the `"paid_loss"` row; the "all six NN entries" phrase becomes "all seven"
- Create: `src/ibnr/gallery/nn/tlrn/card.md`
- Modify: `CHANGELOG.md`

Card sections: Lineage (neutral: "the Transformer Loss Reserving Network of a companion manuscript in preparation, reproduced from its R implementation"; what is reproduced, what is not); Data (company cohorts, the 8 and 13 feature forms, every feature defined in one line each); Network (token layout, axial attention, the counts: `**14,309 parameters**` at `d_model=32` with 13 features); Head (the factor head, support fallback, anchor variant); Training (the objective with its weights, the schedule, minibatch denominators, the checkpoint protocol, `patience=600` never firing inside 3000 epochs by design, seed selection best 2 of 10); Prediction (the point; the historically calibrated company draws; that they are NOT a native predictive distribution; that the reference study calibrated around its blended point and `predict_reserve_draws` exists for that; the calibration overlap with training targets at cutoffs 5 to 7); Evaluate flow (a code block: fit, `point`, `predict`, `reserve_rows`, `shrink_toward`); Limitations (no per-cell draws; no held-out mixins; loss aggregation by company not built; seeds differ from R torch so a rerun reproduces the protocol, not the digits).

- [ ] Wire everything, run `uv run pytest -q -rs` with torch and confirm the NN files executed, commit `feat: register tlrn; card, census tests, parameter pin, CI wiring`.

---

### Task 6: time the published protocol, lint, PR

- [ ] **Measure one seed's epoch time on the study's company set.** With the pinned publish cached (`pinned_source("github://EKtheSage/cas-schedule-p-data-model@20260613_041006")`), load accident years 1998 to 2007 for the four lines, apply the study's selection rule (complete 10 by 10 grids with positive premium; finite positive cumulative paid and cumulative loss ratio below 3 on calendar indices 1 to 5; at least two eligible lines per company - port `select_cohort` from `01_functions.R` in a scratch script), and fit `tlrn` with `TLRNConfig(ensemble_size=1, keep=1, max_epochs=20, min_epochs=0, patience=1000, check_every=5)` paid-only and 13-feature. Report seconds per epoch, projected seconds for 3000 epochs and for 10 seeds, and the thread count (`torch.get_num_threads()`). Put the numbers in the PR body and in the card's Training section. Do not commit the scratch script.
- [ ] Lint, `uv run pytest -q -rs` (torch installed; NN files executed), rebase if main moved, push `feat/tlrn-entry`, open the PR (title `feat: tlrn, the transformer loss reserving network, on the company contract`; body: what it reproduces, the two forms, the timing, verification counts, mutation list, departures from this plan). Do not merge. Report back: PR URL, files, verification, timing, anything unresolved.
