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
functions {
  real os_curve(real t, real ker, real kp, real RLR) {
    return RLR * ker / (ker - kp) * (exp(-kp * t) - exp(-ker * t));
  }

  real paid_curve(real t, real ker, real kp, real RLR, real RRF) {
    return RLR * RRF / (ker - kp)
           * (ker * (1 - exp(-kp * t)) - kp * (1 - exp(-ker * t)));
  }
}
data {
  int<lower=1> len_data;
  int<lower=1> n_w;
  int<lower=1> n_d;
  array[len_data] int<lower=1, upper=n_w> w;
  array[len_data] int<lower=1, upper=n_d> d;
  array[len_data] real<lower=0> t; // dev age in years (period end)
  array[len_data] int<lower=0, upper=1> delta; // 0 = OS level, 1 = incr paid
  array[len_data] real<lower=0> y; // loss ratios, strictly positive
  real<lower=0> devfreq; // dev period length in years
}
parameters {
  real b_oRLR;
  real b_oRRF;
  real b_oker;
  real b_okp;
  vector<lower=0>[2] sd_ay; // correlated AY effects (oRLR, oRRF)
  cholesky_factor_corr[2] L_ay;
  matrix[2, n_w] z_ay;
  vector<lower=0>[2] sd_dev; // independent dev effects (oRLR, oRRF)
  vector[n_d] z_RLR_dev;
  vector[n_d] z_RRF_dev;
  vector<lower=0>[2] sd_ker; // oker (AY, dev) scales
  vector<lower=0>[2] sd_kp; // okp (AY, dev) scales
  vector[n_w] z_ker_ay;
  vector[n_d] z_ker_dev;
  vector[n_w] z_kp_ay;
  vector[n_d] z_kp_dev;
  real log_sigma_os;
  real log_sigma_paid;
}
transformed parameters {
  matrix[2, n_w] u_ay = diag_pre_multiply(sd_ay, L_ay) * z_ay;
  vector[n_d] u_RLR_dev = sd_dev[1] * z_RLR_dev;
  vector[n_d] u_RRF_dev = sd_dev[2] * z_RRF_dev;
  vector[n_w] u_ker_ay = sd_ker[1] * z_ker_ay;
  vector[n_d] u_ker_dev = sd_ker[2] * z_ker_dev;
  vector[n_w] u_kp_ay = sd_kp[1] * z_kp_ay;
  vector[n_d] u_kp_dev = sd_kp[2] * z_kp_dev;
  vector[len_data] mu;
  for (i in 1:len_data) {
    real ker = 3 * exp(0.1 * (b_oker + u_ker_ay[w[i]] + u_ker_dev[d[i]]));
    real kp = 1 * exp(0.1 * (b_okp + u_kp_ay[w[i]] + u_kp_dev[d[i]]));
    real RLR = 0.7 * exp(0.2 * (b_oRLR + u_ay[1, w[i]] + u_RLR_dev[d[i]]));
    real RRF = 0.8 * exp(0.1 * (b_oRRF + u_ay[2, w[i]] + u_RRF_dev[d[i]]));
    if (delta[i] == 0) {
      mu[i] = os_curve(t[i], ker, kp, RLR);
    } else if (t[i] > devfreq) {
      mu[i] = paid_curve(t[i], ker, kp, RLR, RRF)
              - paid_curve(t[i] - devfreq, ker, kp, RLR, RRF);
    } else {
      mu[i] = paid_curve(t[i], ker, kp, RLR, RRF);
    }
  }
}
model {
  // the case study's mypriors2, verbatim
  b_oRLR ~ normal(0, 1);
  b_oRRF ~ normal(0, 1);
  b_oker ~ normal(0, 1);
  b_okp ~ normal(0, 1);
  sd_ay[1] ~ student_t(10, 0, 0.7); // sd(oRLR), both groupings
  sd_ay[2] ~ student_t(10, 0, 0.5); // sd(oRRF)
  sd_dev[1] ~ student_t(10, 0, 0.7);
  sd_dev[2] ~ student_t(10, 0, 0.5);
  sd_ker ~ student_t(10, 0, 0.3);
  sd_kp ~ student_t(10, 0, 0.3);
  L_ay ~ lkj_corr_cholesky(1);
  to_vector(z_ay) ~ std_normal();
  z_RLR_dev ~ std_normal();
  z_RRF_dev ~ std_normal();
  z_ker_ay ~ std_normal();
  z_ker_dev ~ std_normal();
  z_kp_ay ~ std_normal();
  z_kp_dev ~ std_normal();
  log_sigma_os ~ normal(log(0.2), 0.2);
  log_sigma_paid ~ normal(log(0.2), 0.2);
  for (i in 1:len_data) {
    y[i] ~ lognormal(log(mu[i]), delta[i] == 0 ? exp(log_sigma_os) : exp(log_sigma_paid));
  }
}
generated quantities {
  real sigma_os = exp(log_sigma_os);
  real sigma_paid = exp(log_sigma_paid);
  real rho_ay = multiply_lower_tri_self_transpose(L_ay)[1, 2];
  vector[len_data] log_lik;
  for (i in 1:len_data) {
    log_lik[i] = lognormal_lpdf(y[i] | log(mu[i]),
                                delta[i] == 0 ? sigma_os : sigma_paid);
  }
}
