# guszcza_growth_curve - Hierarchical Growth Curve (Guszcza / Gesmann)

**Family:** bayesian · **References:** Guszcza, *Hierarchical Growth Curve
Models for Loss Reserving* (CAS Forum, 2008) for the model structure; the
implementation reference - and the ground truth for the likelihood and the
priors, held verbatim - is Gesmann, "Hierarchical loss reserving with growth
curves using brms" (magesblog.com, 2018-07-15), whose `brm()` call is the
specification. Lineage note: the `compartmental` entry (Gesmann & Morris,
2020) is this model's successor - same author, same hierarchical philosophy,
an ODE compartment system where this has a parametric curve. This entry is
the simpler ancestor.

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

Relation to `clark_growth_curve`: same growth curves, different model. Clark
is fixed-effects with an ODP *quasi*-likelihood on increments - not a
normalized density on any scale, hence CRPS-only. This entry's per-AY random
ultimates and proper lognormal density on cumulative ratios are exactly what
make it ELPD-eligible: a new density member for the milestone-6 board.

## Priors (the post's `my_priors`, verbatim)

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
per-AY random ultimate, shared curve. The per-AY effect and every retained
prior are the post's, untouched. (A cross-company pooled variant would need
a multi-cohort contract - the same open ablation the compartmental card
records.)

## Parameterization (to be held constant by the milestone-5-style ports)

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

## Backends

Stan only (`model.stan`, cmdstanpy). The NumPyro/PyMC ports and their
`kernels.parity` gates are a follow-up task, matching how milestones 4 -> 5
sequenced the rest of the family; `BACKENDS` reserves the seam. Ports must
re-expose `ulr` as a deterministic (the scorer reads it from `posterior`)
and hold the parameterization section above constant.

## Validation status

Unit-tested (closed-form scorer checks, growth-curve equivalence against the
Clark entries' `growth()`, normalization, draw moments, seed threading, and
the slow agreement gate). The Meyers-style 200-company Schedule P
retrospective and the held-out leaderboard run are follow-up work - the
leaderboard harness is being built on a parallel branch.
