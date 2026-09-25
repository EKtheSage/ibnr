# Regenerates tests/data/r_chainladder_delta.json: R ChainLadder's development
# factors on RAA and GenIns for delta = 0, 1 and 2, which ibnr's
# average = "regression", "volume" and "simple" must reproduce.
#
# R ChainLadder's chainladder(Triangle, weights, delta) fits, at each age, a
# regression through the origin of the next cumulative on this one, weighted by
# weights / x^delta. delta = 1 is the volume-weighted chain ladder, delta = 2
# the simple average of the link ratios and delta = 0 ordinary least squares.
# The last case weights out the highest link ratio from 12 to 24 months, so the
# test can check that a dropped ratio leaves the regression average too.
#
# Run from the repository root (CI has no R, so the output is committed):
#   Rscript scripts/r_chainladder_delta.R > tests/data/r_chainladder_delta.json

suppressMessages(library(ChainLadder))

number <- function(x) sprintf("%.17g", x)
vector <- function(x) paste0("[", paste(number(x), collapse = ", "), "]")
factors <- function(tri, delta, weights = NULL) {
  if (is.null(weights)) weights <- matrix(1, nrow(tri), ncol(tri))
  fit <- chainladder(tri, weights = weights, delta = delta)
  unname(sapply(fit$Models, function(m) coef(m)[1]))
}

entries <- c()
for (name in c("RAA", "GenIns")) {
  tri <- get(name)
  w <- matrix(1, nrow(tri), ncol(tri))
  w[which.max(tri[, 2] / tri[, 1]), 1] <- 0
  entries <- c(entries, sprintf(
    paste0(
      '  "%s": {\n',
      '    "delta_0": %s,\n',
      '    "delta_1": %s,\n',
      '    "delta_2": %s,\n',
      '    "delta_0_without_the_highest_first_ratio": %s\n',
      "  }"
    ),
    name,
    vector(factors(tri, 0)),
    vector(factors(tri, 1)),
    vector(factors(tri, 2)),
    vector(factors(tri, 0, w))
  ))
}
cat("{\n")
cat(sprintf(
  '  "source": "R %s, ChainLadder %s, scripts/r_chainladder_delta.R",\n',
  paste(R.version$major, R.version$minor, sep = "."),
  as.character(packageVersion("ChainLadder"))
))
cat(paste(entries, collapse = ",\n"))
cat("\n}\n")
