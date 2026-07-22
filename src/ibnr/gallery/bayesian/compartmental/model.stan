// Hierarchical compartmental reserving — Gesmann & Morris, "Hierarchical
// Compartmental Reserving Models" (CAS Research Paper, 2020), case-study
// Model 1 (their appendix 7.2.2, published brms code = ground truth).
//
// Compartments EX -> OS -> PD with rates ker (earning+reporting) and kp
// (payment), reported loss ratio RLR and reserve robustness factor RRF:
//   dEX/dt = -ker * EX,  dOS/dt = ker * RLR * EX - kp * OS,
//   dPD/dt = kp * RRF * OS,   EX(0) = 1 (premium enters at t = 0)
// solved in closed form below. The likelihood is GAUSSIAN on loss AMOUNTS,
// jointly on case outstanding (delta = 0, a level) and cumulative paid
// (delta = 1), with a separate sigma per delta (brms: sigma ~ 0 + deltaf on
// the log link). ker and kp are fixed across accident years; (RLR, RRF)
// carry correlated accident-year varying effects with an LKJ(1) prior —
// the family's signature reserving-cycle structure.
//
// Parameterization matches the brms nlf transforms exactly:
//   ker = 3 * exp(0.1 * oker),      kp  = 1 * exp(0.1 * okp)
//   RLR[w] = 0.7 * exp(0.2 * (b_oRLR + u_RLR[w]))
//   RRF[w] = 0.8 * exp(0.1 * (b_oRRF + u_RRF[w]))
// so the population priors are lognormal with medians (3, 1, 0.7, 0.8) and
// CoVs (10%, 10%, 20%, 10%). t is the development age in YEARS at the
// cell's period end (the case study's Lag = 1..10) — ker/kp are per-year.
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
  array[len_data] int<lower=1, upper=n_w> w;
  array[len_data] real<lower=0> t; // dev age in years (period end)
  array[len_data] int<lower=0, upper=1> delta; // 0 = outstanding, 1 = paid
  array[len_data] real loss; // amounts; OS may be <= 0, Gaussian takes it
  vector<lower=0>[n_w] premium;
}
parameters {
  real b_oRLR; // population-level oRLR (brms oRLR intercept)
  real b_oRRF;
  real b_oker;
  real b_okp;
  vector<lower=0>[2] sd_ay; // AY varying-effect scales (oRLR, oRRF)
  cholesky_factor_corr[2] L_ay;
  matrix[2, n_w] z_ay; // non-centered AY effects (brms default)
  real log_sigma_os; // brms log-link sigma coefficients per delta
  real log_sigma_paid;
}
transformed parameters {
  matrix[2, n_w] u_ay = diag_pre_multiply(sd_ay, L_ay) * z_ay;
  real ker = 3 * exp(0.1 * b_oker);
  real kp = 1 * exp(0.1 * b_okp);
  vector<lower=0>[n_w] RLR;
  vector<lower=0>[n_w] RRF;
  vector[len_data] mu;
  for (j in 1:n_w) {
    RLR[j] = 0.7 * exp(0.2 * (b_oRLR + u_ay[1, j]));
    RRF[j] = 0.8 * exp(0.1 * (b_oRRF + u_ay[2, j]));
  }
  for (i in 1:len_data) {
    real lr = delta[i] == 0
              ? os_curve(t[i], ker, kp, RLR[w[i]])
              : paid_curve(t[i], ker, kp, RLR[w[i]], RRF[w[i]]);
    mu[i] = premium[w[i]] * lr;
  }
}
model {
  // the case study's mypriors1, verbatim
  b_oRLR ~ normal(0, 1);
  b_oRRF ~ normal(0, 1);
  b_oker ~ normal(0, 1);
  b_okp ~ normal(0, 1);
  sd_ay[1] ~ student_t(10, 0, 0.2); // sd(oRLR)
  sd_ay[2] ~ student_t(10, 0, 0.1); // sd(oRRF)
  L_ay ~ lkj_corr_cholesky(1);
  to_vector(z_ay) ~ std_normal();
  log_sigma_os ~ student_t(1, 0, 1000); // brms class b on log-sigma
  log_sigma_paid ~ student_t(1, 0, 1000);
  for (i in 1:len_data) {
    loss[i] ~ normal(mu[i], delta[i] == 0 ? exp(log_sigma_os) : exp(log_sigma_paid));
  }
}
generated quantities {
  real sigma_os = exp(log_sigma_os);
  real sigma_paid = exp(log_sigma_paid);
  real rho_ay = multiply_lower_tri_self_transpose(L_ay)[1, 2];
  vector[len_data] log_lik;
  for (i in 1:len_data) {
    log_lik[i] = normal_lpdf(loss[i] | mu[i], delta[i] == 0 ? sigma_os : sigma_paid);
  }
}
