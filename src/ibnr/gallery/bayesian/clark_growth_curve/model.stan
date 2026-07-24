// Bayesian Clark growth-curve reserving (Cape Cod form) - likelihood from
// Clark, "LDF Curve-Fitting and Stochastic Reserving: A Maximum Likelihood
// Approach" (CAS Forum 2003); priors are this package's choice (there is no
// published Stan ground truth for a Bayesian Clark - the model card pins
// them, and the milestone-5 ports must hold them constant).
//
// Incremental losses X[w,d] are over-dispersed Poisson around a growth-curve
// share of a Cape Cod ultimate:
//   E[X]   = elr * premium[w] * (G(x_hi) - G(x_lo))
//   Var[X] = phi * E[X]
// with ages measured from the origin's average accident date (the same
// convention as the statistical `clark` entry and chainladder's ClarkLDF).
//
// phi is a PLUG-IN nuisance (data): the Pearson scale from the Clark MLE fit
// of the identical model, mirroring how england_verrall_odp treats its scale.
functions {
  real odp_lpdf(real x, real mu, real phi) {
    return (x / phi) * log(mu / phi) - mu / phi - lgamma(x / phi + 1);
  }

  real growth_curve(real x, real omega, real theta, int curve) {
    if (x <= 0) {
      return 0;
    }
    if (curve == 1) {
      return 1 / (1 + pow(theta / x, omega));  // loglogistic
    }
    return 1 - exp(-pow(x / theta, omega));  // weibull
  }
}
data {
  int<lower=1> len_data;
  int<lower=1> n_w;
  array[len_data] int<lower=1, upper=n_w> w;
  array[len_data] real<lower=0> age_lo;
  array[len_data] real<lower=0> age_hi;
  array[len_data] real<lower=0> inc_loss;
  vector[n_w] logprem_w;
  real<lower=0> phi;
  int<lower=1, upper=2> curve; // 1 = loglogistic, 2 = weibull
  real theta_prior_median; // months; 4 * dev grain, so the prior tracks the grain
}
parameters {
  real logelr;
  real<lower=0> omega;
  real<lower=0> theta;
}
transformed parameters {
  vector[len_data] mu;
  for (i in 1:len_data) {
    mu[i] = exp(logprem_w[w[i]] + logelr)
            * (growth_curve(age_hi[i], omega, theta, curve)
               - growth_curve(age_lo[i], omega, theta, curve));
  }
}
model {
  logelr ~ normal(-0.4, sqrt(10.0)); // the family's variance-10 ELR prior
  omega ~ lognormal(log(1.5), 0.5); // shape: prior mass on 0.6 - 4
  theta ~ lognormal(log(theta_prior_median), 1.0); // scale: median at 4 dev periods
  for (i in 1:len_data) {
    inc_loss[i] ~ odp(mu[i], phi);
  }
}
generated quantities {
  vector[len_data] log_lik;
  for (i in 1:len_data) {
    log_lik[i] = odp_lpdf(inc_loss[i] | mu[i], phi);
  }
}
