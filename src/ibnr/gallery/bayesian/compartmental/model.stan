// Hierarchical compartmental reserving - Gesmann & Morris, "Hierarchical
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
// carry correlated accident-year varying effects with an LKJ(1) prior -
// the family's signature reserving-cycle structure.
//
// Parameterization matches the brms nlf transforms exactly:
//   ker = 3 * exp(0.1 * oker),      kp  = 1 * exp(0.1 * okp)
//   RLR[w] = 0.7 * exp(0.2 * (b_oRLR + u_RLR[w]))
//   RRF[w] = 0.8 * exp(0.1 * (b_oRRF + u_RRF[w]))
// so the population priors are lognormal with medians (3, 1, 0.7, 0.8) and
// CoVs (10%, 10%, 20%, 10%). t is the development age in YEARS at the
// cell's period end (the case study's Lag = 1..10) - ker/kp are per-year.
// CLOSED-FORM ODE SOLUTION. The linear EX -> OS -> PD system is solved
// analytically, so there is no ODE integrator anywhere in this program (the
// monograph's own simplification for the two-rate case). Both curves are
// LOSS RATIOS - EX(0) = 1 means one unit of premium - so callers multiply by
// the origin's premium to get an amount. Mirrored in numpy in model.py
// (os_curve / paid_curve) for the predictive simulation; the two must stay in
// step. The (ker - kp) denominator is singular at ker == kp; the priors
// (medians 3 vs 1) keep the sampler away from it.
functions {
  // Case outstanding at age t: builds at the reporting rate ker, drains at
  // the settlement rate kp, so OS -> 0 as t -> inf (a hump, not a curve).
  real os_curve(real t, real ker, real kp, real RLR) {
    return RLR * ker / (ker - kp) * (exp(-kp * t) - exp(-ker * t));
  }

  // Cumulative paid at age t = integral of kp * RRF * OS; -> RLR * RRF as
  // t -> inf, i.e. the ultimate loss ratio.
  real paid_curve(real t, real ker, real kp, real RLR, real RRF) {
    return RLR * RRF / (ker - kp)
           * (ker * (1 - exp(-kp * t)) - kp * (1 - exp(-ker * t)));
  }
}
// DATA BLOCK = the contract (kernels/contract.py::compartmental_stan_data).
// Rows are DELTA-STACKED: the outstanding block (delta = 0) first, then the
// paid block (delta = 1), each sorted by (w, d) over the same cells - so
// len_data = 2 * (number of observed triangle cells) and the two blocks share
// one set of compartmental parameters. This joint layout is what makes this
// the only gallery entry fitting case reserves and paid together.
data {
  int<lower=1> len_data; // stacked rows = 2 * observed cells
  int<lower=1> n_w; // number of accident years
  array[len_data] int<lower=1, upper=n_w> w; // accident-year index of the cell
  array[len_data] real<lower=0> t; // dev age in years (period end)
  array[len_data] int<lower=0, upper=1> delta; // 0 = outstanding, 1 = paid
  // OS = reported - paid (a level, not a cumulative); paid is cumulative.
  // Both are AMOUNTS in Model 1. No positivity is enforced: redundant case
  // reserves make OS <= 0 and the Gaussian takes that natively - the reason
  // this arm needs no clamp in a mechanical 200-company retrospective.
  array[len_data] real loss; // amounts; OS may be <= 0, Gaussian takes it
  vector<lower=0>[n_w] premium; // per accident year; scales the loss ratios
}
// PARAMETERS. Everything is sampled on an UNCONSTRAINED (o-prefixed) scale
// and mapped to the strictly positive compartmental parameters by the
// lognormal transforms in transformed parameters - the monograph's stated
// reason for this parameterization (no boundary, so Stan's default U(-2, 2)
// init on the unconstrained scale is always valid).
parameters {
  real b_oRLR; // population-level oRLR (brms oRLR intercept)
  real b_oRRF; // population-level oRRF
  real b_oker; // ker and kp have NO accident-year effects in Model 1
  real b_okp;
  vector<lower=0>[2] sd_ay; // AY varying-effect scales (oRLR, oRRF)
  cholesky_factor_corr[2] L_ay; // correlation between the two AY effects
  matrix[2, n_w] z_ay; // non-centered AY effects (brms default)
  real log_sigma_os; // brms log-link sigma coefficients per delta
  real log_sigma_paid; // separate residual scale for the paid block
}
transformed parameters {
  // Non-centered varying effects: u = diag(sd) * L * z with z ~ N(0, 1) gives
  // (u_RLR, u_RRF) ~ MVN(0, D Omega D). This is brms's own default and ports
  // must keep it - a centered version is a different geometry, not the same
  // model at finite sample size (CLAUDE.md design note 7).
  matrix[2, n_w] u_ay = diag_pre_multiply(sd_ay, L_ay) * z_ay;
  // The brms nlf transforms, verbatim: lognormal population priors with
  // medians (3, 1, 0.7, 0.8) and CoVs (10%, 10%, 20%, 10%). Actuarially:
  // exposure is reported ~3x faster than it is paid (ker = 3/yr vs kp = 1/yr),
  // ~70% of premium is expected to be reported as loss, and case reserves are
  // expected to run off ~20% redundant (RRF = 0.8) - hence ULR ~ 0.56.
  real ker = 3 * exp(0.1 * b_oker);
  real kp = 1 * exp(0.1 * b_okp);
  vector<lower=0>[n_w] RLR;
  vector<lower=0>[n_w] RRF;
  vector[len_data] mu;
  for (j in 1:n_w) {
    // per accident year: the reserving-cycle structure lives here - a hard
    // market moves RLR and RRF together (rho_ay > 0 = prudent case reserves)
    RLR[j] = 0.7 * exp(0.2 * (b_oRLR + u_ay[1, j]));
    RRF[j] = 0.8 * exp(0.1 * (b_oRRF + u_ay[2, j]));
  }
  for (i in 1:len_data) {
    // one shared curve system serves both blocks: delta picks which
    // compartment the row observes, and premium puts the loss ratio on the
    // amount scale the Gaussian likelihood works on
    real lr = delta[i] == 0
              ? os_curve(t[i], ker, kp, RLR[w[i]])
              : paid_curve(t[i], ker, kp, RLR[w[i]], RRF[w[i]]);
    mu[i] = premium[w[i]] * lr;
  }
}
// MODEL BLOCK. Every prior below is held VERBATIM from the monograph's
// appendix 7.2 brms code (its `mypriors1` object) - do not retune them here;
// they are the ablation baseline and the milestone-5 ports must match them.
// The card documents what transferring them mechanically costs: they were
// calibrated on one fast-settling workers' comp book, and on long-tailed
// other liability the gaussian arm's median estimate/outcome is 0.84.
model {
  // the case study's mypriors1, verbatim
  // population-level effects: N(0, 1) on the unconstrained scale, i.e. the
  // lognormal population priors quoted in transformed parameters
  b_oRLR ~ normal(0, 1); // -> RLR ~ LN(log 0.7, 0.2)
  b_oRRF ~ normal(0, 1); // -> RRF ~ LN(log 0.8, 0.1)
  b_oker ~ normal(0, 1); // -> ker ~ LN(log 3, 0.1)
  b_okp ~ normal(0, 1); // -> kp  ~ LN(log 1, 0.1), ~63% of OS paid per year
  // half-Student-t(10) scales: heavier-tailed than half-normal, so an
  // accident year with a genuinely different loss ratio is not shrunk away,
  // while nu = 10 still regularizes the AY effects toward zero
  sd_ay[1] ~ student_t(10, 0, 0.2); // sd(oRLR)
  sd_ay[2] ~ student_t(10, 0, 0.1); // sd(oRRF)
  // LKJ(1) = uniform over 2x2 correlation matrices: the sign and size of the
  // RLR-RRF (reserving-cycle) correlation is learned, not assumed
  L_ay ~ lkj_corr_cholesky(1);
  to_vector(z_ay) ~ std_normal(); // non-centered standard normals
  // brms's default class-b prior on the log-sigma coefficients: Cauchy-like
  // (nu = 1) and very wide, because sigma here is on the AMOUNT scale and a
  // company's scale is unknown a priori. Effectively flat.
  log_sigma_os ~ student_t(1, 0, 1000); // brms class b on log-sigma
  log_sigma_paid ~ student_t(1, 0, 1000);
  // Joint Gaussian likelihood, one shared mu vector, residual scale switched
  // by block (brms: sigma ~ 0 + deltaf on the log link). A single constant
  // amount-scale sigma per block is Model 1's known weakness - mature and
  // green accident years get the same dollar band, hence the ~2.5% total CV.
  for (i in 1:len_data) {
    loss[i] ~ normal(mu[i], delta[i] == 0 ? exp(log_sigma_os) : exp(log_sigma_paid));
  }
}
generated quantities {
  real sigma_os = exp(log_sigma_os); // back on the amount scale
  real sigma_paid = exp(log_sigma_paid); // read by model.py::_predict_gaussian
  // posterior RLR-RRF accident-year correlation: > 0 means years with a high
  // reported loss ratio also hold proportionally stronger case reserves
  real rho_ay = multiply_lower_tri_self_transpose(L_ay)[1, 2];
  // pointwise log-likelihood over the STACKED cells, so ELPD/LOO in kernels/
  // scores the joint paid+outstanding fit rather than paid alone
  vector[len_data] log_lik;
  for (i in 1:len_data) {
    log_lik[i] = normal_lpdf(loss[i] | mu[i], delta[i] == 0 ? sigma_os : sigma_paid);
  }
}
