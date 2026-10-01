# Regenerates tests/data/r_bootchainladder.json: R ChainLadder's
# BootChainLadder(Triangle, R = 20000, process.distr, seed) on RAA and GenIns,
# which ibnr's run-off ODP bootstrap with R's conventions (the degrees-of-freedom
# adjustment, every residual resampled, not centred) must agree with to Monte
# Carlo error.
#
# Each run records the total IBNR's mean, standard deviation and quantiles
# (R's type 7, which is numpy's linear rule), each origin's mean and standard
# deviation, the pool of residuals R resamples (its size, how many are exactly
# zero, its mean) and the chain-ladder reserve. The gamma process runs with
# seeds 1 and 2; od.pois (R's negative binomial with the over-dispersed
# Poisson's two moments) with seed 1.
#
# Run from the repository root (CI has no R, so the output is committed):
#   Rscript scripts/r_bootchainladder.R > tests/data/r_bootchainladder.json

suppressMessages(library(ChainLadder))

number <- function(x) ifelse(is.finite(x), sprintf("%.16g", x), "null")
vector <- function(x) paste0("[", paste(number(unname(x)), collapse = ", "), "]")

levels <- c(0.5, 0.75, 0.9, 0.95, 0.99)
runs <- list(
  list(name = "RAA", distr = "gamma", seed = 1),
  list(name = "RAA", distr = "gamma", seed = 2),
  list(name = "GenIns", distr = "gamma", seed = 1),
  list(name = "GenIns", distr = "gamma", seed = 2),
  list(name = "RAA", distr = "od.pois", seed = 1),
  list(name = "GenIns", distr = "od.pois", seed = 1)
)
out <- character(0)
for (run in runs) {
  tri <- get(run$name)
  b <- BootChainLadder(tri, R = 20000, process.distr = run$distr, seed = run$seed)
  total <- b$IBNR.Totals
  by <- b$IBNR.ByOrigin[, 1, ]
  pool <- as.vector(b$ChainLadder.Residuals)
  pool <- pool[!is.na(pool)]
  out <- c(out, sprintf(
    paste0(
      '    {"triangle": "%s", "process": "%s", "seed": %d, "n_draws": 20000, ',
      '"mean": %s, "sd": %s, "levels": %s, "quantiles": %s, ',
      '"origin_mean": %s, "origin_sd": %s, ',
      '"pool_size": %d, "pool_zeros": %d, "pool_mean": %s, "chain_ladder_reserve": %s}'
    ),
    run$name, run$distr, run$seed,
    number(mean(total)), number(sd(total)), vector(levels),
    vector(quantile(total, levels, type = 7)),
    vector(apply(by, 1, mean)), vector(apply(by, 1, sd)),
    length(pool), sum(pool == 0), number(mean(pool)),
    number(sum(summary(MackChainLadder(tri))$ByOrigin$IBNR))
  ))
}
cat("{\n")
cat(sprintf(
  '  "source": "R %s, ChainLadder %s, scripts/r_bootchainladder.R",\n',
  paste(R.version$major, R.version$minor, sep = "."),
  as.character(packageVersion("ChainLadder"))
))
cat('  "runs": [\n')
cat(paste(out, collapse = ",\n"))
cat("\n  ]\n}\n")
