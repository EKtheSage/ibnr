# Reference numbers for ibnr's Tweedie GLM (kernels/glm.py, methods.tweedie_glm).
#
# R's glm() with statmod::tweedie(var.power = p, link.power = 0) fitted to the
# observed increments of four public triangles from R's ChainLadder package,
# with value ~ factor(origin) + factor(dev), at p = 0, 1, 1.5 and 2. Also: the
# identity link at p = 0 on GenIns, a calendar trend with development factors
# (and no origin factors) on GenIns at p = 1, and ChainLadder::glmReserve as a
# second R route to the same reserves.
#
# CI has no R, so the output is committed as tests/data/tweedie_glm_r.json and
# tests/test_glm.py reads it. To regenerate, from the repository root:
#
#   Rscript scripts/r/tweedie_glm_reference.R > tests/data/tweedie_glm_r.json
#
# Written with R 4.5.3, ChainLadder 0.2.21 and statmod 1.5.2 on 2026-09-25.
# The JSON is written by hand (jsonlite is not needed), with 17 significant
# digits so that every double survives the round trip.

suppressPackageStartupMessages({
  library(ChainLadder)
  library(statmod)
})

num <- function(x) {
  ifelse(is.na(x), "null", sprintf("%.17g", x))
}
arr <- function(x) paste0("[", paste(num(x), collapse = ", "), "]")
str_arr <- function(x) paste0("[", paste0('"', x, '"', collapse = ", "), "]")
obj <- function(fields) {
  paste0("{", paste(sprintf('"%s": %s', names(fields), unlist(fields)), collapse = ", "), "}")
}

long_inc <- function(tri) {
  inc <- cum2incr(tri)
  d <- expand.grid(origin = seq_len(nrow(inc)), dev = seq_len(ncol(inc)))
  d$value <- as.vector(inc)
  d$obs <- !is.na(d$value)
  d$cal <- d$origin + d$dev
  d
}

control <- glm.control(epsilon = 1e-12, maxit = 100)

fit_json <- function(tri, formula, p, link_power = 0) {
  d <- long_inc(tri)
  fam <- tweedie(var.power = p, link.power = link_power)
  m <- glm(formula, family = fam, data = d[d$obs, ], control = control)
  s <- summary(m)
  mu <- predict(m, newdata = d, type = "response")
  reserve <- tapply(ifelse(d$obs, 0, mu), d$origin, sum)
  obj(list(
    power = num(p),
    link = if (link_power == 0) '"log"' else '"identity"',
    formula = paste0('"', paste(deparse(formula), collapse = ""), '"'),
    iterations = m$iter,
    converged = if (m$converged) "true" else "false",
    reserve_by_origin = arr(as.vector(reserve)),
    fitted = arr(as.vector(t(matrix(mu, nrow = nrow(tri))))),  # row by row
    deviance = num(m$deviance),
    pearson_chi2 = num(sum(residuals(m, type = "pearson")^2)),
    dispersion = num(s$dispersion),
    df_residual = m$df.residual,
    coefficient_names = str_arr(names(coef(m))),
    coefficients = arr(coef(m)),
    std_errors = arr(s$coefficients[, "Std. Error"])
  ))
}

cumulative_rows <- function(tri) {
  rows <- lapply(seq_len(nrow(tri)), function(i) {
    r <- as.vector(tri[i, ])
    arr(r[!is.na(r)])
  })
  paste0("[", paste(rows, collapse = ", "), "]")
}

main <- ~ factor(origin) + factor(dev)
triangles <- list()
for (nm in c("GenIns", "UKMotor", "ABC", "MW2014")) {
  tri <- get(nm)
  fits <- sapply(c(0, 1, 1.5, 2), function(p) fit_json(tri, update(main, value ~ .), p))
  glm_reserve <- sapply(c(1, 1.5, 2), function(p) {
    g <- glmReserve(tri, var.power = p, link.power = 0)
    num(g$summary["total", "IBNR"])
  })
  cl <- MackChainLadder(tri, est.sigma = "Mack")
  triangles[[nm]] <- obj(list(
    first_origin = as.integer(rownames(tri)[1]),
    cumulative = cumulative_rows(tri),
    fits = paste0("[", paste(fits, collapse = ", "), "]"),
    glm_reserve_total_ibnr = obj(setNames(as.list(glm_reserve), c("1", "1.5", "2"))),
    chain_ladder_ibnr_by_origin = arr(summary(cl)$ByOrigin$IBNR)
  ))
}

extras <- obj(list(
  genins_identity_p0 = fit_json(GenIns, value ~ factor(origin) + factor(dev), 0, link_power = 1),
  genins_calendar_p1 = fit_json(GenIns, value ~ factor(dev) + cal, 1)
))

cat(obj(list(
  source = paste0(
    '"scripts/r/tweedie_glm_reference.R: R ', R.version$major, ".", R.version$minor,
    ", ChainLadder ", packageVersion("ChainLadder"), ", statmod ", packageVersion("statmod"), '"'
  ),
  triangles = obj(triangles),
  extras = extras
)), "\n", sep = "")
