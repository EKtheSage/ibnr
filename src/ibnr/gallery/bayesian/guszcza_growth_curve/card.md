# guszcza_growth_curve - Single-Company Growth Curve, Hierarchical Across Accident Years

**This entry is hierarchical across ACCIDENT YEARS, not across companies.**
The source post's title means the latter - it pools ten insurers - and this
entry cannot, because the package fits one cohort at a time. What is fitted
here is Guszcza's single-company model: a random ultimate loss ratio per
accident year, shared growth-curve parameters. See "What collapsed" below;
it changes the marginal prior, not only the parameter count.

**Family:** bayesian · **References:** Guszcza, *Hierarchical Growth Curve
Models for Loss Reserving* (CAS Forum, 2008) for the model structure; the
implementation reference - and the ground truth for the likelihood and the
priors - is Gesmann, "Hierarchical loss reserving with growth curves using
brms" (magesblog.com, 2018-07-15), whose `brm()` call is the specification.
Lineage note: the `compartmental` entry (Gesmann & Morris, 2020) is this
model's successor - same author, same hierarchical philosophy, an ODE
compartment system where this has a parametric curve. This entry is the
simpler ancestor.

## Model

Cumulative paid **loss ratios** follow a lognormal around a growth-curve
share of a per-accident-year ultimate loss ratio:

```
y[w,d]  = C[w,d] / premium[w]
y[w,d]  ~ lognormal(log(ulr[w] * G(t; omega, theta)), sigma)
ulr[w]  = ulr_pop + sd_ulr * z_ulr[w],   z_ulr[w] ~ N(0, 1)
```

`t = d * dev_grain_months / 12` is the development age in **years** (the
post's `dev_year` = 1..10 on the annual grain), so `theta`'s prior reads
"half of ultimate emerges by about four years" on any grain. The growth
curve is selected as data (`curve` = 1/2), and its algebra is **reused**
from `gallery/statistical/clark/model.py::growth` - one algebra, three
readers (that module, `scorer.py`, the Stan `functions` block), equivalence
pinned by test:

```
loglogistic (default, the post's):  G = t^omega / (t^omega + theta^omega)
weibull (ablatable alternative):    G = 1 - exp(-(t/theta)^omega)
```

Both curves appear in Guszcza (2008) (he leads with the weibull) and in
Zhang, Dukic & Guszcza (2012), the post's other stated ancestor; the curve
is this entry's one ablatable dimension, everything else shared.

**Units: `theta` here is in YEARS.** `growth()` is unit-agnostic - `x` and
`theta` appear only as the ratio `theta/x` - so it is correct on any age
unit as long as both sides agree. The Clark entries pass MONTHS (Clark's own
convention); this entry passes years so that `theta ~ normal(4, 1)` keeps
the post's meaning on any dev grain. **A fitted `theta` is therefore not
comparable between this card and `clark_growth_curve`'s**, and not only by a
factor of 12: the age conventions differ too (Clark measures from the
origin's average accident date, `12d - 6`; this entry from the period end,
`12d`).

Relation to `clark_growth_curve`: same growth curves, different model. Clark
is fixed-effects with an ODP *quasi*-likelihood on increments - not a
normalized density on any scale, hence CRPS-only. This entry's per-AY random
ultimates and proper lognormal density on cumulative ratios are exactly what
make it ELPD-eligible: a new density member for the milestone-6 board.

## Priors (each line the post's `my_priors` verbatim; the marginal differs - see below)

```
ulr_pop ~ lognormal(log(0.6), log(2))     prior(lognormal(log(0.6), log(2)), nlpar="ulr", lb=0)
omega   ~ normal(2, 1),  omega > 0        prior(normal(2, 1), nlpar="omega", lb=0)
theta   ~ normal(4, 1),  theta > 0        prior(normal(4, 1), nlpar="theta", lb=0)
sigma   ~ student_t(3, 0, 1),  sigma > 0  prior(student_t(3, 0, 1), class="sigma")
sd_ulr  ~ student_t(3, 0, 1),  sd_ulr > 0 prior(student_t(3, 0, 1), class="sd", nlpar="ulr")
```

**What collapsed, and why.** The post fits ten workers' compensation
insurers jointly: `ulr`, `omega` and `theta` carry correlated per-company
effects (`(1|ID|entity_name)`, `lkj(2)`), and `ulr` additionally carries the
per-accident-year-within-company effect (`(1|origin_year:entity_name)`).
This package's contract is one cohort per fit, and a single company cannot
identify a between-company sd or an LKJ correlation - those effects are
exactly confounded with the population intercepts. So the company level
collapses into `ulr_pop`/`omega`/`theta`, their `student_t(3, 0, 1)` sds and
the `lkj(2)` prior drop with it, and what remains is Guszcza's own model:
per-AY random ultimate, shared curve. (A cross-company pooled variant would
need a multi-cohort contract - the same open ablation the compartmental card
records.)

**This changes the marginal prior, and in the over-confident direction.**
Every prior line above is the post's verbatim, but the priors above are not
the whole prior: dropping the entity level also removes the entity-level
`student_t(3, 0, 1)` sd from the *marginal* prior on this cohort's
parameters. The post's implied single-company marginal on `omega` is roughly
`N(2, sqrt(1 + sd^2))`; ours is `N(2, 1)` - the same centre, a narrower
spread, on `omega`, `theta` and `ulr` alike. The effect is modest at these
scales, but it runs the way that costs calibration rather than the way that
is safe: a too-tight prior is precisely how `compartmental`'s gaussian arm
came out too sharp (CV around 2.5%, combined D = 39.9). Worth revisiting
when this entry gets its Schedule P retrospective - a widened-prior arm is
the natural ablation if the PIT comes back over-confident.

## Parameterization (held constant across all three backends)

- **Non-centered** AY effects: `ulr = ulr_pop + sd_ulr * z_ulr` with
  `z_ulr ~ std_normal()`, brms's own parameterization of group effects.
- **Bounds mirror brms's `lb=0` declarations** (`ulr_pop`, `omega`, `theta`,
  `sd_ulr`, `sigma` all `<lower=0>`). This family's bounds are load-bearing
  (see CLAUDE.md on `sd_ay` and `a_ig`); ports must keep every one.
- `ulr[w]` itself is **not** bounded: the AY effect is additive on the ulr
  scale, so a draw can push it non-positive, `log()` goes NaN and the
  proposal is rejected - the same implicit truncation the brms-generated
  Stan code has. Every *retained* draw therefore has `ulr[w] > 0` for every
  trained origin, which `predict()` and the scorer rely on.
- **Init strategy:** partial inits - `z_ulr` starts at 0 (so the initial
  `ulr = ulr_pop > 0` and the first `mu` is finite for every chain),
  everything else at Stan's random default so the chains stay dispersed.
- **Sampler settings:** the post's own `control` list, `adapt_delta = 0.999`
  and `max_treedepth = 15`, as the defaults. The model is small (n_w + 5
  parameters), so they are cheap here.

## Data contract

`kernels.contract.stan_data` (the shared cumulative-loss contract: one
cohort, positive losses, per-origin premium required). The entry derives
`t = d * dev_grain_months / 12` and `y = loss / premium[w]` from it - no new
contract builder. The contract's non-positive-loss refusal is exactly the
lognormal's support requirement.

## Predictive distribution

`predict()` returns cumulative paid at the triangle's final development age
per origin plus the total: each not-fully-developed origin draws
`premium[w] * lognormal(log(ulr[w] * G(t_final)), sigma)` per posterior
draw; fully developed origins anchor at the observed value (zero variance),
the Meyers-family convention. No tail beyond the final age - the
retrospective scores realized paid there, and `G(t_final) < 1` means an
extrapolated ultimate would score a different quantity.

```python
from ibnr import gallery

entry = gallery.fit(
    "guszcza_growth_curve",
    triangle,
    loss_field="paid_loss",
    premium_field="earned_premium",
    as_of="1997-12-31",
    growth_curve="loglogistic",
)
pred = entry.predict(seed=0)
```

## Held-out scoring (milestone 6)

- `heldout_measure = "loss_ratio"`: the density is on cumulative paid loss
  ratios; `ScoresHeldout.log_lik_at` carries it to Lebesgue-on-amount by
  `- log premium`. Normalization on the amount space is pinned by test.
- `heldout_draw_scale = "cumulative"`: draws are cumulative amounts (the
  scorer scales the ratio draw by premium), so `predict_at` is a
  pass-through on the cumulative Schedule P triangles.
- The in-sample agreement gate (`-m slow`) scores the training cells through
  the held-out path and reproduces the Stan fit's own `log_lik` elementwise,
  for both curves, with a perturbation negative control.

## Backends (three ports, one data block)

| file | backend | sampler |
|---|---|---|
| `model.stan` | `stan` (reference, ground truth) | cmdstanpy NUTS |
| `model_numpyro.py` | `numpyro` | NumPyro NUTS (JAX) |
| `model_pymc.py` | `pymc` | PyMC NUTS (PyTensor, or `nuts_sampler="numpyro"` over the same graph) |

All three consume the identical Stan `data` block the entry assembles - the ages
in years, the loss ratios and the integer curve code included - so no port can
drift on the age convention or quietly fit the other curve. All three re-expose
`ulr` as a deterministic, because that is what the held-out scorer reads out of
`posterior` (`scorer.REQUIRED_DRAWS`); a port that fitted without it would raise
only when something tried to score it.

`parallel_chains` is a cmdstan-level control and is **rejected** by the ports
rather than silently ignored. `max_treedepth` is **not** in that category and is
shared by all three: this entry's default of 15 comes from the post's own
`control` list and differs from NumPyro's and PyMC's default of 10, so holding it
constant is part of what parity means here. `_shared.py` holds the pieces that
are not PPL-specific (the curve codes, that tree depth, the data guard) so the
two ports cannot drift apart on them.

### There is no zero-age hazard here, and that is worth stating

`clark_growth_curve` needs a gradient-safe zero-age branch because its
mid-period `age_lo` clamps to exactly 0 on every origin's first cell. **This
entry does not**, and `model.stan`'s `growth_curve` is written without the
`if (x <= 0)` guard Clark's carries: `t = d * dev_grain_months / 12` with
`d >= 1`, so every age is strictly positive.

What the ports guard instead is the data, because Stan's
`vector<lower=0>[len_data] t` is *looser* than the invariant the missing branch
relies on. At `t = 0` the loglogistic's value is finite (`G -> 0`) and both
derivatives are NaN, so `_shared.check_data` refuses a non-positive age by name
rather than letting NUTS die on the first leapfrog. Ages are data, so the check
is static and free.

### The real hazard: `ulr` is unbounded and Stan rejects it

`ulr[w] = ulr_pop + sd_ulr * z_ulr[w]` has no lower bound - the accident-year
effect is additive on the ulr scale, exactly as brms builds an nlpar's linear
predictor - so a draw can push it non-positive. Stan then evaluates `log()` of a
non-positive number, raises a domain error and **rejects the proposal**, and that
rejection is what confines the posterior to the positive orthant. It is not an
edge case - the reference fits emit `lognormal_lpdf: Location parameter[i] is
nan` warnings during warmup even on data drawn from the model itself - and it is
why every retained draw satisfies `ulr[w] > 0`, a property `predict()` and the
scorer both rely on (`scorer.check_ulr_positive`).

Each port reproduces the rejection as a log-density of `-inf`, with a safe dummy
substituted **inside** the `log`. Measured on the shipped model against an
unsubstituted one, at a point where every `ulr` is negative:

| form | density | all six gradients |
|---|---|---|
| substituted (shipped) | `-inf` | finite |
| unsubstituted | **NaN** | **NaN** |

`-inf` is a rejection NUTS understands; NaN is not.

**The mechanism is not Clark's, and the difference decides where the
substitution goes.** Clark's trap is `theta/0 -> inf` inside an unselected
`where` branch, where `0 * inf` poisons the reverse pass, so masking the result
loses the gradient. Here the offending operation is `log` of a negative number,
whose derivative `1/x` is perfectly finite - so masking only the `log`'s result
*does* keep the gradient, and the first version of this entry's negative control
did exactly that and passed on the broken model. What actually breaks is the
density: `mu` feeds `LogNormal(mu, sigma).log_prob(y)`, so one NaN cell makes the
observed site NaN and `NaN + (-inf) = NaN`. `tests/test_parity_guszcza.py` pins
both halves, because conflating them is how the guard ends up in the wrong place.

### Priors a port can get subtly wrong

`omega ~ normal(2, 1)` under `<lower=0>` is a **truncated** normal that keeps its
location at 2, not a half-normal about 0 - the easiest mis-port in this model,
and invisible to any shape assertion. `sd_ulr` and `sigma` are half Student-t:
PyMC has `HalfStudentT` natively, while NumPyro needs Stan's own construction (a
positive-constrained `ImproperUniform` plus the density as a `numpyro.factor`),
because `TruncatedDistribution(StudentT)` requires a Student-t CDF that raises
`ImportError: tensorflow_probability` - the same finding `compartmental` records.

Rather than assert shapes, `test_ports_match_the_stan_target` reimplements
`model.stan`'s whole target in numpy and requires each port to differ from it by
a **constant** across a grid of parameter points, then checks that the constant
is the one the two constructions predict: `-log P(N(2,1) > 0) - log P(N(4,1) > 0)`
for NumPyro, plus `2 log 2` for PyMC's two half-t's. That pins every prior's
location and scale, the non-centering and the likelihood at once. Its tolerances
are **relative** to the density's magnitude, because JAX is single-precision by
default and these densities run to ~2000, where float32 rounding alone is ~2e-4.

## Cross-backend parity & convergence (milestone 5)

Compared parameters are `GUSZCZA_PARITY_VARS` = `ulr_pop, sd_ulr, ulr, omega,
theta, sigma` - 15 elements on a 10-origin triangle. `ulr` is in the set because
it is what `predict()` and the held-out scorer actually consume (the same reason
CCL compares `alpha`); the raw `z_ulr` is out, being the non-centered nuisance
whose only content is `ulr`, exactly as CCL excludes `a_ig`. Comparing in MCSE
units is what makes including `ulr` safe: a late origin with two cells has a
large `mcse_sd`, so its noisier estimate is tolerated automatically instead of
tripping a fixed band.

**Result: 4 of 4 PASS** - every z-score below 2.4 against a tolerance of 4, and
zero divergences in every backend on every cohort. Two WC companies as of
1997-12-31, 4 chains x 2500 draws after 1000 warmup, `target_accept = 0.999`
and `max_treedepth = 15` (the post's own control list) shared by all three
backends, common seed, loglogistic curve. Full data in
`analysis/results/{parity,convergence}_guszcza.csv`; reproduce with

```
uv run python scripts/parity_gallery.py --model guszcza_growth_curve \
    --line workers_compensation --companies 11347 38687 --nuts-sampler numpyro
```

| company | backend | max &#124;z_mean&#124; | max &#124;z_sd&#124; | max KS | verdict |
|---|---|---|---|---|---|
| 11347 | numpyro | 1.29 | 1.01 | 0.022 | PASS |
| 11347 | pymc | 1.32 | 2.00 | 0.028 | PASS |
| 38687 | numpyro | 1.55 | 2.32 | 0.025 | PASS |
| 38687 | pymc | 1.04 | 1.83 | 0.022 | PASS |

| company | backend | runtime | max R-hat | min ESS-bulk | min ESS-tail | divergences /10000 |
|---|---|---|---|---|---|---|
| 11347 | stan | 66.6s | 1.00 | 1941 | 2459 | 0 |
| 11347 | numpyro | 37.6s | 1.00 | 1829 | 2530 | 0 |
| 11347 | pymc:numpyro | 18.8s | 1.00 | 2058 | 2575 | 0 |
| 38687 | stan | 35.7s | 1.00 | 2095 | 3180 | 0 |
| 38687 | numpyro | 35.7s | 1.00 | 1902 | 2729 | 0 |
| 38687 | pymc:numpyro | 8.7s | 1.00 | 1993 | 2805 | 0 |

**Do not read the runtime column as an implementation ranking.** The `pymc` rows
ran through `nuts_sampler="numpyro"` (see the Backends section - native PyTensor
is ~130x here), and PyMC's JAX path defaults to `chain_method="parallel"` while
this repo's NumPyro ports deliberately run chains **sequentially**, the fair
single-core convention against cmdstan's `parallel_chains=1`. So `pymc:numpyro`
being the fastest row is mostly four chains at once, not a better graph; the
`backend` column records what actually sampled, which is why it reads
`pymc:numpyro` rather than `pymc`.

ESS lands at ~1800-3200 of 10000 draws in every backend - the cost of
`adapt_delta = 0.999`, which the post specifies because this nonlinear hierarchy
diverges at brms defaults. That it is uniform across backends is the point:
the ports are paying the same price for the same geometry.

## Validation status

Unit-tested (closed-form scorer checks, growth-curve equivalence against the
Clark entries' `growth()`, normalization, draw moments, seed threading, and
the slow agreement gate), plus the cross-backend parity above and the
port-level gates in `tests/test_parity_guszcza.py` (the whole-target
reimplementation, the `-inf`-not-NaN rejection, the treedepth-delivery checks).
The Meyers-style 200-company Schedule P retrospective and the held-out
leaderboard run are follow-up work.
