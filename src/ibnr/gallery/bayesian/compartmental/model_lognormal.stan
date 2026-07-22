// Hierarchical compartmental reserving — Gesmann & Morris (2020) case-study
// Model 2 (their appendix 7.2.3, published brms code = ground truth): the
// LOGNORMAL variant the monograph selects over Model 1 ("incompatibilities
// of a Gaussian process distribution").
//
// Same EX -> OS -> PD closed-form compartments as model.stan, but:
// - observations are loss RATIOS: outstanding (delta = 0, a level) and
//   INCREMENTAL paid (delta = 1; cell d covers (t - devfreq, t]);
// - y ~ lognormal(log(mu), sigma[delta]), so non-positive cells cannot
//   enter (the entry drops them and reports the count);
// - every compartmental parameter carries varying effects by BOTH accident
//   year and development period ("row" and "column" effects on the
//   parameters, not the outcome): (oRLR, oRRF) accident-year effects are
//   correlated (LKJ(1), the brms |ID| term); all dev effects and the
//   ker/kp accident-year effects are independent.
//
// brms nlf parameterization, exactly as Model 1:
//   ker[i,j] = 3 * exp(0.1 * (b_oker + u_ker_ay[w] + u_ker_dev[d]))
//   kp[i,j]  = 1 * exp(0.1 * (b_okp  + u_kp_ay[w]  + u_kp_dev[d]))
//   RLR[i,j] = 0.7 * exp(0.2 * (b_oRLR + u_ay[1, w] + u_RLR_dev[d]))
//   RRF[i,j] = 0.8 * exp(0.1 * (b_oRRF + u_ay[2, w] + u_RRF_dev[d]))
// The incremental paid mean differences paid_curve at t and t - devfreq
// with the SAME cell parameters (the brms claimsprocess with devfreq).
//
// NOTE the appendix prior on log-sigma is normal(log(0.2), 0.2) while the
// monograph text (eq. 8 discussion) says lognormal(log(0.1), 0.2) for
// sigma; the appendix code produced the published results and wins.
// Closed-form solution of the EX -> OS -> PD system, byte-identical to
// model.stan (and to model.py's numpy mirror). No ODE integrator.
functions {
  // Case outstanding loss ratio at age t (builds at ker, drains at kp).
  real os_curve(real t, real ker, real kp, real RLR) {
    return RLR * ker / (ker - kp) * (exp(-kp * t) - exp(-ker * t));
  }

  // Cumulative paid loss ratio at age t; -> RLR * RRF (the ULR) as t -> inf.
  real paid_curve(real t, real ker, real kp, real RLR, real RRF) {
    return RLR * RRF / (ker - kp)
           * (ker * (1 - exp(-kp * t)) - kp * (1 - exp(-ker * t)));
  }
}
// DATA BLOCK = the same delta-stacked contract as Model 1, with three
// differences the entry applies in model.py::_lognormal_stan_data:
// y is a LOSS RATIO (loss / premium, so no premium vector is needed — the
// curves are already loss ratios), the paid block is INCREMENTAL, and every
// non-positive cell has been dropped before sampling (counted in the entry's
// dropped_cells_). d is now carried too, because Model 2 puts varying
// effects on the development period as well as the accident year.
data {
  int<lower=1> len_data; // surviving stacked rows (<= 2 * observed cells)
  int<lower=1> n_w; // accident years
  int<lower=1> n_d; // development periods
  array[len_data] int<lower=1, upper=n_w> w; // accident-year index
  array[len_data] int<lower=1, upper=n_d> d; // development-period index
  array[len_data] real<lower=0> t; // dev age in years (period end)
  array[len_data] int<lower=0, upper=1> delta; // 0 = OS level, 1 = incr paid
  array[len_data] real<lower=0> y; // loss ratios, strictly positive
  real<lower=0> devfreq; // dev period length in years
}
// PARAMETERS. Same unconstrained (o-prefixed) population coefficients as
// Model 1, plus EIGHT vectors of varying effects: each of the four
// compartmental parameters gets an accident-year ("row") and a
// development-period ("column") effect. All are non-centered (z_* standard
// normals scaled by an sd_*). This extra freedom is what lets the curve
// escape the binding case-study priors — the card's explanation for why the
// lognormal arm's median estimate/outcome sits at 0.99-1.02 per line while
// the gaussian arm's is 0.84 on other liability.
parameters {
  real b_oRLR;
  real b_oRRF;
  real b_oker;
  real b_okp;
  vector<lower=0>[2] sd_ay; // correlated AY effects (oRLR, oRRF)
  cholesky_factor_corr[2] L_ay; // only (RLR, RRF) x AY are correlated
  matrix[2, n_w] z_ay;
  vector<lower=0>[2] sd_dev; // independent dev effects (oRLR, oRRF)
  vector[n_d] z_RLR_dev;
  vector[n_d] z_RRF_dev;
  vector<lower=0>[2] sd_ker; // oker (AY, dev) scales
  vector<lower=0>[2] sd_kp; // okp (AY, dev) scales
  vector[n_w] z_ker_ay; // ker/kp DO vary by accident year here (Model 1: no)
  vector[n_d] z_ker_dev;
  vector[n_w] z_kp_ay;
  vector[n_d] z_kp_dev;
  real log_sigma_os; // residual scale is now on the LOG (loss-ratio) scale
  real log_sigma_paid;
}
transformed parameters {
  // correlated AY effects for (oRLR, oRRF), non-centered exactly as Model 1
  matrix[2, n_w] u_ay = diag_pre_multiply(sd_ay, L_ay) * z_ay;
  // every remaining varying effect is independent: a plain scale * z
  vector[n_d] u_RLR_dev = sd_dev[1] * z_RLR_dev;
  vector[n_d] u_RRF_dev = sd_dev[2] * z_RRF_dev;
  vector[n_w] u_ker_ay = sd_ker[1] * z_ker_ay;
  vector[n_d] u_ker_dev = sd_ker[2] * z_ker_dev;
  vector[n_w] u_kp_ay = sd_kp[1] * z_kp_ay;
  vector[n_d] u_kp_dev = sd_kp[2] * z_kp_dev;
  vector[len_data] mu; // expected LOSS RATIO per cell (no premium factor)
  for (i in 1:len_data) {
    // per-CELL compartmental parameters (Model 1 has one set per accident
    // year): the brms nlf transforms with the same medians and CoVs as
    // Model 1, now summing an AY and a dev effect on the log scale.
    // Mirrored in model.py::_predict_lognormal — keep the two in step.
    real ker = 3 * exp(0.1 * (b_oker + u_ker_ay[w[i]] + u_ker_dev[d[i]]));
    real kp = 1 * exp(0.1 * (b_okp + u_kp_ay[w[i]] + u_kp_dev[d[i]]));
    real RLR = 0.7 * exp(0.2 * (b_oRLR + u_ay[1, w[i]] + u_RLR_dev[d[i]]));
    real RRF = 0.8 * exp(0.1 * (b_oRRF + u_ay[2, w[i]] + u_RRF_dev[d[i]]));
    if (delta[i] == 0) {
      // outstanding is a LEVEL, so it is read straight off the curve
      mu[i] = os_curve(t[i], ker, kp, RLR);
    } else if (t[i] > devfreq) {
      // incremental paid over (t - devfreq, t], differenced with the SAME
      // cell parameters on both ends (the monograph's claimsprocess with
      // devfreq) — NOT the neighbouring cell's parameters
      mu[i] = paid_curve(t[i], ker, kp, RLR, RRF)
              - paid_curve(t[i] - devfreq, ker, kp, RLR, RRF);
    } else {
      // first development period: paid_curve(0) = 0, so the cumulative value
      // is already the increment
      mu[i] = paid_curve(t[i], ker, kp, RLR, RRF);
    }
  }
}
// MODEL BLOCK. Every prior is held VERBATIM from the monograph's appendix
// 7.2 brms code (its `mypriors2` object); the milestone-5 ports must match
// them exactly, and any retuning belongs in a new ablatable variant (see the
// card's open `hierarchical` ablation), not here.
model {
  // the case study's mypriors2, verbatim
  // population-level effects: unchanged from Model 1, so the two variants
  // share the same prior medians (ker 3, kp 1, RLR 0.7, RRF 0.8)
  b_oRLR ~ normal(0, 1);
  b_oRRF ~ normal(0, 1);
  b_oker ~ normal(0, 1);
  b_okp ~ normal(0, 1);
  // half-Student-t(10) scales, MUCH wider than Model 1's (0.7/0.5 vs
  // 0.2/0.1): with eight groups of varying effects the monograph deliberately
  // lets the AY and dev effects move, and the same prior is used for both
  // groupings of a given parameter
  sd_ay[1] ~ student_t(10, 0, 0.7); // sd(oRLR), both groupings
  sd_ay[2] ~ student_t(10, 0, 0.5); // sd(oRRF)
  sd_dev[1] ~ student_t(10, 0, 0.7);
  sd_dev[2] ~ student_t(10, 0, 0.5);
  // the rates are kept tighter (0.3): settlement speed should drift, not jump
  sd_ker ~ student_t(10, 0, 0.3); // both elements: (AY, dev)
  sd_kp ~ student_t(10, 0, 0.3);
  L_ay ~ lkj_corr_cholesky(1); // uniform over correlations, as Model 1
  // non-centered standard normals for all eight varying-effect vectors
  to_vector(z_ay) ~ std_normal();
  z_RLR_dev ~ std_normal();
  z_RRF_dev ~ std_normal();
  z_ker_ay ~ std_normal();
  z_ker_dev ~ std_normal();
  z_kp_ay ~ std_normal();
  z_kp_dev ~ std_normal();
  // sigma is now a relative (log-scale) CV, so it can be given a genuinely
  // informative prior: median 0.2. See the header note — the monograph TEXT
  // quotes LN(log 0.1, 0.2) but its appendix code, which produced the
  // published results, uses normal(log(0.2), 0.2) on the log-sigma
  // coefficients. Code wins; ports must not silently adopt the text version.
  log_sigma_os ~ normal(log(0.2), 0.2);
  log_sigma_paid ~ normal(log(0.2), 0.2);
  // Lognormal likelihood on loss ratios: multiplicative error, so the
  // predictive band scales with the cell rather than being a constant dollar
  // width (Model 1's defect). Its support (0, inf) is why non-positive cells
  // had to be dropped upstream.
  for (i in 1:len_data) {
    y[i] ~ lognormal(log(mu[i]), delta[i] == 0 ? exp(log_sigma_os) : exp(log_sigma_paid));
  }
}
generated quantities {
  real sigma_os = exp(log_sigma_os);
  real sigma_paid = exp(log_sigma_paid); // used by _predict_lognormal's draws
  // RLR-RRF accident-year correlation (the reserving cycle), as in Model 1
  real rho_ay = multiply_lower_tri_self_transpose(L_ay)[1, 2];
  // pointwise log-lik over the surviving stacked cells; note the ELPD is on
  // the loss-ratio scale here and the amount scale in Model 1, so the two
  // variants' log_lik values are NOT directly comparable
  vector[len_data] log_lik;
  for (i in 1:len_data) {
    log_lik[i] = lognormal_lpdf(y[i] | log(mu[i]),
                                delta[i] == 0 ? sigma_os : sigma_paid);
  }
}
