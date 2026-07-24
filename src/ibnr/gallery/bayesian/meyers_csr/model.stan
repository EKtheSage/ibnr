// Changing Settlement Rate (CSR) - Meyers, "Stochastic Loss Reserving Using
// Bayesian MCMC Models", CAS Monograph 1 (2015) section 8 / Monograph 8
// (2019, 2nd ed.) section 7. The paid-loss counterpart of CCL/CAY: the
// cross-classified lognormal with a settlement-rate trend.
//
// log(C[w,d]) ~ normal(mu[w,d], sig[d])
//   mu[w,d] = logprem[w] + logelr + alpha[w] + beta[d] * (1 - gamma)^(w-1)
//
// gamma > 0: settlement speedup (log-development factors beta[d]*speedup[w]
// shrink toward 0 for later origins); gamma < 0: slowdown. gamma == 0
// recovers the plain cross-classified (CRC) model.
//
// Syntax modernized for Stan >= 2.33 (array keyword, gamma_cdf '|'); the
// math and priors follow Meyers' published CSR.R exactly.
data {
  int<lower=1> len_data;
  int<lower=1> n_w;
  int<lower=1> n_d;
  array[len_data] int<lower=1, upper=n_w> w;
  array[len_data] int<lower=1, upper=n_d> d;
  array[len_data] real logprem;
  array[len_data] real logloss;
}
parameters {
  vector[n_w - 1] r_alpha;
  vector[n_d - 1] r_beta;
  real logelr;
  // inverse-gamma reparameterization: gamma_cdf(1/a_ig | 1, 1) ~ uniform(0,1)
  // (monograph technical note: avoids hard boundaries near zero)
  vector<lower=0, upper=100000>[n_d] a_ig;
  real gamma;
}
transformed parameters {
  vector[n_w] alpha;
  vector[n_d] beta;
  vector[n_w] speedup;
  vector<lower=0>[n_d] sig2;
  vector<lower=0>[n_d] sig;
  vector[len_data] mu;

  alpha[1] = 0;
  alpha[2:n_w] = r_alpha;
  beta[1:(n_d - 1)] = r_beta;
  beta[n_d] = 0;

  speedup[1] = 1;
  for (i in 2:n_w) {
    speedup[i] = speedup[i - 1] * (1 - gamma);
  }

  // sig2[d] = sum_{i=d}^{n_d} a_i with a_i ~ uniform(0,1): forces
  // sig2[1] > sig2[2] > ... > sig2[n_d] (more settled claims -> less variance)
  sig2[n_d] = gamma_cdf(1 / a_ig[n_d] | 1, 1);
  for (i in 1:(n_d - 1)) {
    sig2[n_d - i] = sig2[n_d + 1 - i] + gamma_cdf(1 / a_ig[n_d - i] | 1, 1);
  }
  sig = sqrt(sig2);

  for (i in 1:len_data) {
    mu[i] = logprem[i] + logelr + alpha[w[i]] + beta[d[i]] * speedup[w[i]];
  }
}
model {
  logelr ~ normal(-0.4, sqrt(10.0));
  r_alpha ~ normal(0, sqrt(10.0));
  r_beta ~ normal(0, sqrt(10.0));
  a_ig ~ inv_gamma(1, 1);
  gamma ~ normal(0, 0.05);
  logloss ~ normal(mu, sig[d]);
}
generated quantities {
  vector[len_data] log_lik;
  for (i in 1:len_data) {
    log_lik[i] = normal_lpdf(logloss[i] | mu[i], sig[d[i]]);
  }
}
