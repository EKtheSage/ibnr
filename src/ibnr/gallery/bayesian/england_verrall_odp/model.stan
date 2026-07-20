// Bayesian over-dispersed Poisson (ODP) cross-classified chain ladder —
// England & Verrall, "Stochastic Claims Reserving in General Insurance",
// B.A.J. 8 III (2002), sections 3.2 (ODP), 7.11 (Bayesian implementation).
//
// Incremental losses X[w,d]:
//   E[X] = m[w,d],  Var[X] = phi * m[w,d]
//   log m[w,d] = logprem[w] + c + alpha[w] + beta[d]     (log link; E&V 7.11.8
//                note the log-link linear predictor is the stable form)
//   alpha[1] = 0, beta[1] = 0
//
// The premium offset only re-centers alpha (each origin has a free level), so
// the model family — and its MLE, which reproduces chain-ladder reserves —
// is identical to E&V's; the offset just makes the vague priors exchangeable
// across origins of different sizes.
//
// phi is a PLUG-IN nuisance (data, not a parameter): the GLM Pearson
// chi-square / dof estimate, exactly as England & Verrall treat the scale.
// With phi fixed, the quasi-likelihood below is the exact od-Poisson mass at
// X/phi up to an X-and-phi-only constant, so the posterior is proper.
functions {
  real odp_lpdf(real x, real mu, real phi) {
    return (x / phi) * log(mu / phi) - mu / phi - lgamma(x / phi + 1);
  }
}
data {
  int<lower=1> len_data;
  int<lower=1> n_w;
  int<lower=1> n_d;
  array[len_data] int<lower=1, upper=n_w> w;
  array[len_data] int<lower=1, upper=n_d> d;
  array[len_data] real<lower=0> inc_loss;
  array[len_data] real logprem;
  real<lower=0> phi;
}
parameters {
  real c;
  vector[n_w - 1] r_alpha;
  vector[n_d - 1] r_beta;
}
transformed parameters {
  vector[n_w] alpha;
  vector[n_d] beta;
  vector[len_data] log_mu;

  alpha[1] = 0;
  alpha[2:n_w] = r_alpha;
  beta[1] = 0;
  beta[2:n_d] = r_beta;

  for (i in 1:len_data) {
    log_mu[i] = logprem[i] + c + alpha[w[i]] + beta[d[i]];
  }
}
model {
  c ~ normal(0, sqrt(10.0));
  r_alpha ~ normal(0, sqrt(10.0));
  r_beta ~ normal(0, sqrt(10.0));
  for (i in 1:len_data) {
    inc_loss[i] ~ odp(exp(log_mu[i]), phi);
  }
}
generated quantities {
  vector[len_data] log_lik;
  for (i in 1:len_data) {
    log_lik[i] = odp_lpdf(inc_loss[i] | exp(log_mu[i]), phi);
  }
}
