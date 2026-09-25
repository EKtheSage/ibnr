# Regenerates tests/data/r_mack_alpha_weights.json: R ChainLadder's
# MackChainLadder(Triangle, weights, alpha, est.sigma) on RAA, GenIns, UKMotor,
# ABC and MW2014 under 27 development settings each, which ibnr's Mack fit with
# the same options must reproduce.
#
# alpha is Mack's exponent: 0 is the simple average of the link ratios, 1 the
# volume-weighted average and 2 least squares through the origin. The settings
# are each alpha with every link ratio, the latest 5 and the latest 3 at each
# age, each of those with and without the second origin's first link ratio left
# out, and each alpha with the highest ratio left out at every age, the lowest,
# and the latest 5 with the highest left out. This script turns each setting into
# the 0/1 weights matrix R takes, by the rules ibnr's documentation states: the
# window first, then the exclusion, then the trim, and a trim is not made at an
# age where it would leave no ratio. The test checks ibnr chose the same ratios.
#
# Each fit is run under est.sigma = "Mack" and "log-linear". R switches the
# log-linear rule to Mack's, with a warning, when the log-linear regression's
# slope has a p-value above 0.05; the output records it as "switched", and the
# test compares the log-linear rule only where R did not switch. R returns an
# infinite standard error when an age keeps one link ratio and it is not the
# first origin's (its f.se falls back to the first origin's cell and divides by
# its weight of 0); those fits are recorded with "finite": false and not
# compared. Any error R raises is recorded as the fit's "error".
#
# Run from the repository root (CI has no R, so the output is committed):
#   Rscript scripts/r_mack_alpha_weights.R > tests/data/r_mack_alpha_weights.json

suppressMessages(library(ChainLadder))

number <- function(x) ifelse(is.finite(x), sprintf("%.16g", x), "null")
vector <- function(x) paste0("[", paste(number(unname(x)), collapse = ", "), "]")
matrix_json <- function(m) {
  paste0("[", paste(apply(m, 1, vector), collapse = ", "), "]")
}
# a 0/1 weights matrix as one string of digits per origin
weights_json <- function(w) {
  paste0('["', paste(apply(w, 1, paste, collapse = ""), collapse = '", "'), '"]')
}

weights_for <- function(tri, history, exclude, trim) {
  n <- ncol(tri)
  # 1 everywhere a link ratio is not left out, as R's own default: R also reads
  # the weight of a cell with no successor (the latest diagonal and the future)
  # as the weight of the development still to come from it
  w <- matrix(1, nrow(tri), n)
  pair <- function(j) which(!is.na(tri[, j]) & !is.na(tri[, j + 1]))
  for (j in seq_len(n - 1)) {
    pairs <- pair(j)
    if (!is.na(history) && length(pairs) > history) w[head(pairs, length(pairs) - history), j] <- 0
  }
  if (exclude) w[2, 1] <- 0
  if (trim != "none") {
    for (j in seq_len(n - 1)) {
      used <- intersect(pair(j), which(w[, j] == 1))
      if (length(used) < 2) next
      ratio <- tri[used, j + 1] / tri[used, j]
      pick <- if (trim == "high") which.max(ratio) else which.min(ratio)
      w[used[pick], j] <- 0
    }
  }
  w
}

settings <- list()
for (alpha in c(0, 1, 2)) {
  for (history in c(NA, 5, 3)) {
    for (exclude in c(FALSE, TRUE)) {
      settings[[length(settings) + 1]] <- list(
        alpha = alpha, history = history, exclude = exclude, trim = "none"
      )
    }
  }
  settings[[length(settings) + 1]] <- list(alpha = alpha, history = NA, exclude = FALSE, trim = "high")
  settings[[length(settings) + 1]] <- list(alpha = alpha, history = NA, exclude = FALSE, trim = "low")
  settings[[length(settings) + 1]] <- list(alpha = alpha, history = 5, exclude = FALSE, trim = "high")
}

datasets <- c()
fits <- c()
for (name in c("RAA", "GenIns", "UKMotor", "ABC", "MW2014")) {
  tri <- get(name)
  n <- ncol(tri)
  datasets <- c(datasets, sprintf('    "%s": %s', name, matrix_json(unclass(tri))))
  for (s in settings) {
    w <- weights_for(tri, s$history, s$exclude, s$trim)
    for (es in c("Mack", "log-linear")) {
      warn <- character(0)
      fit <- withCallingHandlers(
        tryCatch(
          MackChainLadder(tri, weights = w, alpha = s$alpha, est.sigma = es),
          error = function(e) e
        ),
        warning = function(wn) {
          warn <<- c(warn, conditionMessage(wn))
          invokeRestart("muffleWarning")
        }
      )
      head <- sprintf(
        paste0(
          '      "dataset": "%s", "alpha": %d, "history_periods": %s, ',
          '"exclude_second_origin_first_link": %s, "trim": "%s", "est_sigma": "%s"'
        ),
        name, s$alpha, ifelse(is.na(s$history), "null", s$history),
        ifelse(s$exclude, "true", "false"), s$trim, es
      )
      if (inherits(fit, "error")) {
        fits <- c(fits, sprintf(
          '    {\n%s,\n      "error": "%s"\n    }', head, gsub('"', "'", conditionMessage(fit))
        ))
        next
      }
      values <- c(fit$Mack.S.E[, n], fit$Total.Mack.S.E, fit$f.se[seq_len(n - 1)])
      if (!all(is.finite(values))) {
        fits <- c(fits, sprintf('    {\n%s,\n      "finite": false\n    }', head))
        next
      }
      fits <- c(fits, sprintf(
        paste0(
          "    {\n%s,\n",
          '      "switched": %s, "finite": true,\n',
          '      "weights": %s,\n',
          '      "f": %s,\n',
          '      "sigma": %s,\n',
          '      "f_se": %s,\n',
          '      "mack_se": %s,\n',
          '      "process_se": %s,\n',
          '      "parameter_se": %s,\n',
          '      "total_mack_se": %s, "total_process_se": %s, "total_parameter_se": %s\n',
          "    }"
        ),
        head,
        ifelse(any(grepl("overwritten to 'Mack'", warn)), "true", "false"),
        weights_json(w),
        vector(fit$f[seq_len(n - 1)]),
        vector(fit$sigma[seq_len(n - 1)]),
        vector(fit$f.se[seq_len(n - 1)]),
        vector(fit$Mack.S.E[, n]),
        vector(fit$Mack.ProcessRisk[, n]),
        vector(fit$Mack.ParameterRisk[, n]),
        number(fit$Total.Mack.S.E),
        number(fit$Total.ProcessRisk[n]),
        number(fit$Total.ParameterRisk[n])
      ))
    }
  }
}
cat("{\n")
cat(sprintf(
  '  "source": "R %s, ChainLadder %s, scripts/r_mack_alpha_weights.R",\n',
  paste(R.version$major, R.version$minor, sep = "."),
  as.character(packageVersion("ChainLadder"))
))
cat('  "triangles": {\n')
cat(paste(datasets, collapse = ",\n"))
cat("\n  },\n")
cat('  "fits": [\n')
cat(paste(fits, collapse = ",\n"))
cat("\n  ]\n}\n")
