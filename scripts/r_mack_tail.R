# Regenerates tests/data/r_mack_tail.json: R ChainLadder's
# MackChainLadder(Triangle, alpha, est.sigma, tail, tail.se, tail.sigma) on RAA,
# GenIns, UKMotor, ABC and MW2014, which ibnr's Mack fit with the same tail must
# reproduce.
#
# The tails are a constant 1.05 (R reads the tail's sigma and the tail factor's
# standard error off straight lines through the logs of the sigmas and the
# factors' standard errors, at the age where a straight line through
# log(f - 1) reaches log(1.05 - 1)); R's own exponential curve (tail = TRUE,
# which is ibnr's tail="exponential" with 100 steps); and 1.05 with the tail's
# sigma 0.1 and standard error 0.02 given. Each is run with alpha 1 (the volume
# average); the constant 1.05 also with alpha 0 (the simple average) and alpha 2
# (least squares through the origin), where the tail step's process variance is
# divided by the last observed amount to the power alpha.
#
# Each fit is run under est.sigma = "log-linear" and "Mack". R switches the
# log-linear rule to Mack's, with a warning, when the log-linear regression's
# slope has a p-value above 0.05; the output records it as "switched", and the
# test then compares R's numbers with ibnr's sigma_rule="mack". Any error R
# raises is recorded as the fit's "error".
#
# Run from the repository root (CI has no R, so the output is committed):
#   Rscript scripts/r_mack_tail.R > tests/data/r_mack_tail.json

suppressMessages(library(ChainLadder))

number <- function(x) ifelse(is.finite(x), sprintf("%.16g", x), "null")
vector <- function(x) paste0("[", paste(number(unname(x)), collapse = ", "), "]")
matrix_json <- function(m) {
  paste0("[", paste(apply(m, 1, vector), collapse = ", "), "]")
}

settings <- list(
  list(alpha = 1, tail = "1.05", se = NA, sigma = NA),
  list(alpha = 1, tail = "TRUE", se = NA, sigma = NA),
  list(alpha = 1, tail = "1.05", se = 0.02, sigma = 0.1),
  list(alpha = 0, tail = "1.05", se = NA, sigma = NA),
  list(alpha = 2, tail = "1.05", se = NA, sigma = NA)
)

datasets <- c()
fits <- c()
for (name in c("RAA", "GenIns", "UKMotor", "ABC", "MW2014")) {
  tri <- get(name)
  n <- ncol(tri)
  datasets <- c(datasets, sprintf('    "%s": %s', name, matrix_json(unclass(tri))))
  for (s in settings) {
    tail <- if (s$tail == "TRUE") TRUE else as.numeric(s$tail)
    for (es in c("log-linear", "Mack")) {
      warn <- character(0)
      args <- list(tri, alpha = s$alpha, est.sigma = es, tail = tail)
      if (!is.na(s$se)) args$tail.se <- s$se
      if (!is.na(s$sigma)) args$tail.sigma <- s$sigma
      fit <- withCallingHandlers(
        tryCatch(do.call(MackChainLadder, args), error = function(e) e),
        warning = function(wn) {
          warn <<- c(warn, conditionMessage(wn))
          invokeRestart("muffleWarning")
        }
      )
      head <- sprintf(
        paste0(
          '      "dataset": "%s", "alpha": %d, "tail": %s, "tail_se": %s, ',
          '"tail_sigma": %s, "est_sigma": "%s"'
        ),
        name, s$alpha, ifelse(s$tail == "TRUE", '"exponential"', s$tail),
        ifelse(is.na(s$se), "null", s$se), ifelse(is.na(s$sigma), "null", s$sigma), es
      )
      if (inherits(fit, "error")) {
        fits <- c(fits, sprintf(
          '    {\n%s,\n      "error": "%s"\n    }', head, gsub('"', "'", conditionMessage(fit))
        ))
        next
      }
      last <- ncol(fit$Mack.S.E)
      fits <- c(fits, sprintf(
        paste0(
          "    {\n%s,\n",
          '      "switched": %s,\n',
          '      "f": %s,\n',
          '      "sigma": %s,\n',
          '      "f_se": %s,\n',
          '      "ultimate": %s,\n',
          '      "mack_se": %s,\n',
          '      "process_se": %s,\n',
          '      "parameter_se": %s,\n',
          '      "total_mack_se": %s, "total_process_se": %s, "total_parameter_se": %s\n',
          "    }"
        ),
        head,
        ifelse(any(grepl("overwritten to 'Mack'", warn)), "true", "false"),
        vector(fit$f[seq_len(n)]),
        vector(fit$sigma[seq_len(n)]),
        vector(fit$f.se[seq_len(n)]),
        vector(fit$FullTriangle[, ncol(fit$FullTriangle)]),
        vector(fit$Mack.S.E[, last]),
        vector(fit$Mack.ProcessRisk[, last]),
        vector(fit$Mack.ParameterRisk[, last]),
        number(fit$Total.Mack.S.E),
        number(fit$Total.ProcessRisk[last]),
        number(fit$Total.ParameterRisk[last])
      ))
    }
  }
}
cat("{\n")
cat(sprintf(
  '  "source": "R %s, ChainLadder %s, scripts/r_mack_tail.R",\n',
  paste(R.version$major, R.version$minor, sep = "."),
  as.character(packageVersion("ChainLadder"))
))
cat('  "triangles": {\n')
cat(paste(datasets, collapse = ",\n"))
cat("\n  },\n")
cat('  "fits": [\n')
cat(paste(fits, collapse = ",\n"))
cat("\n  ]\n}\n")
