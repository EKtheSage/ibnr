// Hierarchical growth-curve loss reserving - Guszcza (2008) structure with
// the likelihood and priors of Gesmann's brms implementation ("Hierarchical
// loss reserving with growth curves using brms", magesblog.com, 2018-07-15;
// its brm() call is the specification and the priors below are its
// `my_priors` verbatim).
//
// The post fits TEN companies jointly, with correlated per-company effects
// on (ulr, omega, theta). This entry fits ONE cohort at a time - the same
// single-cohort contract every Bayesian entry consumes - so the company
// level collapses into the population intercepts (one company cannot
// identify a between-company sd or an LKJ correlation) and what remains is
// exactly Guszcza's hierarchical growth-curve model: a per-accident-year
// random effect on the ultimate loss ratio, shared curve parameters.
//
//   y[i]   = C[w,d] / premium[w]                cumulative paid loss ratio
//   y[i]   ~ lognormal(log(ulr[w[i]] * G(t[i]; omega, theta)), sigma)
//   ulr[w] = ulr_pop + sd_ulr * z_ulr[w]        non-centered AY effect
//
// t is the development age in YEARS (the post's dev_year = 1..10 on the
// annual grain), so theta ~ normal(4, 1) reads "half of ultimate emerges by
// about four years". G is selected as data so a port cannot drift on which
// curve it fits: the loglogistic (the post's curve) or the weibull
// (Guszcza's other curve; same pair as the Clark entries).
functions {
  // Shared algebra with gallery/statistical/clark/model.py::growth (the
  // loglogistic is written in its numerically stable 1/(1 + (theta/t)^omega)
  // form, identical to t^omega / (t^omega + theta^omega)); the equivalence
  // is pinned by test. t > 0 always here (t = d * grain / 12, d >= 1), so
  // no zero-age branch is needed - unlike Clark, whose mid-period age_lo
  // clamps to exactly 0.
  real growth_curve(real t, real omega, real theta, int curve) {
    if (curve == 1) {
      return 1 / (1 + pow(theta / t, omega)); // loglogistic
    }
    return 1 - exp(-pow(t / theta, omega)); // weibull
  }
}
data {
  int<lower=1> len_data;
  int<lower=1> n_w;
  array[len_data] int<lower=1, upper=n_w> w; // accident-year index
  vector<lower=0>[len_data] t; // dev age in YEARS at the cell (d * grain/12)
  vector<lower=0>[len_data] y; // cumulative paid loss ratio, > 0 upstream
  int<lower=1, upper=2> curve; // 1 = loglogistic (the post's), 2 = weibull
}
parameters {
  // bounds mirror the brms lb=0 declarations exactly; this family's bounds
  // are load-bearing (CLAUDE.md: sd_ay, a_ig), so a port must keep them
  real<lower=0> ulr_pop; // population ultimate loss ratio (brms b_ulr)
  real<lower=0> omega; // growth-curve shape (brms b_omega)
  real<lower=0> theta; // growth-curve scale, years (brms b_theta)
  real<lower=0> sd_ulr; // sd of the AY effects (brms sd, origin_year:...)
  vector[n_w] z_ulr; // non-centered standard normals
  real<lower=0> sigma; // lognormal residual scale (brms sigma)
}
transformed parameters {
  // additive AY effect on the ulr scale, exactly as brms builds the nlpar's
  // linear predictor; ulr[w] is therefore NOT bounded below by 0, and a draw
  // that pushes it non-positive makes mu NaN and is rejected - the same
  // implicit truncation the brms-generated Stan code has
  vector[n_w] ulr = ulr_pop + sd_ulr * z_ulr;
  vector[len_data] mu;
  for (i in 1 : len_data) {
    mu[i] = log(ulr[w[i]] * growth_curve(t[i], omega, theta, curve));
  }
}
model {
  // the post's my_priors, verbatim, less the company-level pieces that
  // collapse for a single cohort: the (1|ID|entity_name) effects on
  // ulr/omega/theta, their student_t(3,0,1) sds and the lkj(2) correlation
  ulr_pop ~ lognormal(log(0.6), log(2));
  omega ~ normal(2, 1); // half-normal via the <lower=0> bound (brms lb=0)
  theta ~ normal(4, 1);
  sd_ulr ~ student_t(3, 0, 1); // half-t via the bound (brms class sd)
  z_ulr ~ std_normal();
  sigma ~ student_t(3, 0, 1); // half-t via the bound (brms class sigma)
  y ~ lognormal(mu, sigma);
}
generated quantities {
  // pointwise log density ON THE LOSS-RATIO SCALE (the model's own measure);
  // ScoresHeldout.log_lik_at carries it to Lebesgue-on-amount (-log premium)
  vector[len_data] log_lik;
  for (i in 1 : len_data) {
    log_lik[i] = lognormal_lpdf(y[i] | mu[i], sigma);
  }
}
