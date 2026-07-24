// Correlated Chain Ladder (CCL) - Meyers, "Stochastic Loss Reserving Using
// Bayesian MCMC Models", CAS Monograph 1 (2015); renamed CAY (Correlated
// Accident Year) in the 2nd edition, CAS Monograph 8 (2019), section 8.
//
// log(C[w,d]) ~ normal(mu[w,d], sig[d])
//   mu[1,d] = logprem[1] + logelr + beta[d]
//   mu[w,d] = logprem[w] + logelr + alpha[w] + beta[d]
//             + rho * (log(C[w-1,d]) - mu[w-1,d])      for w > 1
//
// Syntax modernized for Stan >= 2.33 (array keyword, gamma_cdf '|'); the
// math and priors follow the monograph exactly. The prev_idx indirection
// replaces the original code's reliance on row ordering for the rho term.
data {
  int<lower=1> len_data;
  int<lower=1> n_w;
  int<lower=1> n_d;
  array[len_data] int<lower=1, upper=n_w> w;
  array[len_data] int<lower=1, upper=n_d> d;
  // row index of the observation at (w-1, d); 0 when w == 1.
  // Rows are sorted by (w, d), so prev_idx[i] < i.
  array[len_data] int<lower=0, upper=len_data> prev_idx;
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
  real<lower=0, upper=1> r_rho;
}
transformed parameters {
  vector[n_w] alpha;
  vector[n_d] beta;
  vector<lower=0>[n_d] sig2;
  vector<lower=0>[n_d] sig;
  real<lower=-1, upper=1> rho;
  vector[len_data] mu;

  alpha[1] = 0;
  alpha[2:n_w] = r_alpha;
  beta[1:(n_d - 1)] = r_beta;
  beta[n_d] = 0;
  rho = 2 * r_rho - 1;

  // sig2[d] = sum_{i=d}^{n_d} a_i with a_i ~ uniform(0,1): forces
  // sig2[1] > sig2[2] > ... > sig2[n_d] (more settled claims -> less variance)
  sig2[n_d] = gamma_cdf(1 / a_ig[n_d] | 1, 1);
  for (i in 1:(n_d - 1)) {
    sig2[n_d - i] = sig2[n_d + 1 - i] + gamma_cdf(1 / a_ig[n_d - i] | 1, 1);
  }
  sig = sqrt(sig2);

  for (i in 1:len_data) {
    mu[i] = logprem[i] + logelr + alpha[w[i]] + beta[d[i]];
    if (prev_idx[i] > 0) {
      mu[i] += rho * (logloss[prev_idx[i]] - mu[prev_idx[i]]);
    }
  }
}
model {
  logelr ~ normal(-0.4, sqrt(10.0));
  r_alpha ~ normal(0, sqrt(10.0));
  r_beta ~ normal(0, sqrt(10.0));
  a_ig ~ inv_gamma(1, 1);
  r_rho ~ beta(2, 2);
  logloss ~ normal(mu, sig[d]);
}
generated quantities {
  vector[len_data] log_lik;
  for (i in 1:len_data) {
    log_lik[i] = normal_lpdf(logloss[i] | mu[i], sig[d[i]]);
  }
}
