#' Component-Wise Gradient Boosting Across Multiply Imputed Datasets
#'
#' Applies component-wise gradient boosting to multiply imputed datasets.
#' Depending on the settings, either a separate model is reported for each
#' imputed dataset, or the M models are pooled to yield a single final model.
#' For pooling, one can choose the algorithm \emph{MIBoost} (Boosting after
#' Multiple Imputation), which enforces a uniform variable-selection scheme
#' across all imputations, or the more conventional approaches of
#' estimate-averaging and selection-frequency thresholding.
#'
#' @param X_list       List of length M; each element is an \eqn{n \times p} numeric
#'   predictor matrix from one imputed dataset.
#' @param y_list       List of length M; each element is a length-\eqn{n} numeric
#'   response vector from one imputed dataset.
#' @param X_list_val   Optional validation list (same structure as \code{X_list}).
#' @param y_list_val   Optional validation list (same structure as \code{y_list}).
#' @param ny           Learning rate. Defaults to \code{0.1}.
#' @param mstop        Number of boosting iterations (default \code{250}).
#' @param type         Type of loss function. One of:
#'   \code{"gaussian"} (mean squared error) for continuous responses,
#'   or \code{"logistic"} (binomial deviance) for binary responses.
#' @param MIBoost      Logical. If \code{TRUE}, applies the MIBoost algorithm,
#'   which enforces uniform variable selection across all imputed datasets. If
#'   \code{FALSE}, variables are selected independently within each imputed
#'   dataset, and pooling is governed by \code{pool_threshold}.
#' @param pool         Logical. If \code{TRUE}, models across the \eqn{M} imputed
#'   datasets are aggregated into a single final model. If \code{FALSE},
#'   \eqn{M} separate models are returned.
#' @param pool_threshold Only used when \code{MIBoost = FALSE} and \code{pool = TRUE}.
#'   Controls the pooling rule when aggregating the \eqn{M} models obtained from
#'   the imputed datasets into a single final model. A candidate variable is
#'   included only if it is selected in at least \code{pool_threshold} (a value
#'   in (0, 1)) proportion of the imputed datasets; coefficients of all other
#'   variables are set to zero. A value of \code{0} corresponds to
#'   estimate-averaging, while values \code{> 0} correspond to
#'   selection-frequency thresholding.
#' @param center       One of \code{c("auto", "off", "force")}. Controls
#'   centering of \code{X} within each imputed dataset.
#'   With \code{"auto"} (recommended), centering is applied only if the training
#'   matrix is not already centered. With \code{"force"}, centering is always
#'   applied. With \code{"off"}, centering is skipped. If \code{X_list_val} is
#'   provided, validation sets are centered using the means from the
#'   corresponding training set.
#'
#' @return A list with elements:
#' \itemize{
#'   \item \code{INT}: intercept(s). A scalar if \code{pool = TRUE}, otherwise
#'     a length-M vector.
#'   \item \code{BETA}: coefficient estimates. A length-p vector if
#'     \code{pool = TRUE}, otherwise an M \eqn{\times} p matrix.
#'   \item \code{CV_error}: vector of validation errors (if validation data
#'     were provided), otherwise \code{NULL}.
#' }
#'
#' @details
#' This function supports \emph{MIBoost}, which enforces uniform variable
#' selection across multiply imputed datasets. For full methodology, see the
#' references below.
#'
#' @references
#' Buehlmann, P. and Hothorn, T. (2007). "Boosting Algorithms: Regularization,
#' Prediction and Model Fitting." \doi{10.1214/07-STS242} \cr
#' Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting
#' Machine." \doi{10.1214/aos/1013203451} \cr
#' van Buuren, S. and Groothuis-Oudshoorn, K. (2011). "mice: Multivariate
#' Imputation by Chained Equations in R." \doi{10.18637/jss.v045.i03} \cr
#' Kuchen, R. (2025). "MIBoost: A Gradient Boosting Algorithm for Variable
#' Selection After Multiple Imputation." \doi{10.48550/arXiv.2507.21807}
#'
#' @examplesIf requireNamespace("mice", quietly = TRUE) && requireNamespace("miceadds", quietly = TRUE)
#' \donttest{
#'   set.seed(123)
#'   utils::data(booami_sim)
#'
#'   M <- 2
#'   n <- nrow(booami_sim)
#'   x_cols <- grepl("^X\\d+$", names(booami_sim))
#'
#'   tr_idx <- sample(seq_len(n), floor(0.8 * n))
#'   dat_tr <- booami_sim[tr_idx, , drop = FALSE]
#'   dat_va <- booami_sim[-tr_idx, , drop = FALSE]
#'
#'   pm_tr <- mice::quickpred(dat_tr, method = "spearman",
#'                            mincor = 0.30, minpuc = 0.60)
#'
#'   imp_tr <- mice::mice(dat_tr, m = M, predictorMatrix = pm_tr,
#'                        maxit = 1, printFlag = FALSE)
#'   imp_va <- mice::mice.mids(imp_tr, newdata = dat_va, maxit = 1, printFlag = FALSE)
#'
#'   X_list      <- vector("list", M)
#'   y_list      <- vector("list", M)
#'   X_list_val  <- vector("list", M)
#'   y_list_val  <- vector("list", M)
#'   for (m in seq_len(M)) {
#'     tr_m <- mice::complete(imp_tr, m)
#'     va_m <- mice::complete(imp_va, m)
#'     X_list[[m]]     <- data.matrix(tr_m[, x_cols, drop = FALSE])
#'     y_list[[m]]     <- tr_m$y
#'     X_list_val[[m]] <- data.matrix(va_m[, x_cols, drop = FALSE])
#'     y_list_val[[m]] <- va_m$y
#'   }
#'
#'   fit <- impu_boost(
#'     X_list, y_list,
#'     X_list_val = X_list_val, y_list_val = y_list_val,
#'     ny = 0.1, mstop = 50, type = "gaussian",
#'     MIBoost = TRUE, pool = TRUE, center = "auto"
#'   )
#'
#'   which.min(fit$CV_error)
#'   head(fit$BETA)
#'   fit$INT
#' }
#'
#' @seealso \code{\link{simulate_booami_data}},
#'   \code{\link{cv_boost_raw}}, \code{\link{cv_boost_imputed}}
#' @export
impu_boost <- function(
    X_list, y_list,
    X_list_val = NULL, y_list_val = NULL,
    ny = 0.1, mstop = 250, type = c("gaussian","logistic"),
    MIBoost = TRUE, pool = TRUE, pool_threshold = 0,
    center = "auto"
) {
  ## ---- argument checks ------------------------------------------------------
  type <- match.arg(type)
  stopifnot(is.list(X_list), is.list(y_list), length(X_list) == length(y_list))
  M <- length(X_list)
  stopifnot(M >= 1L)

  # enforce scalar logicals (prevents 'length > 1' in if())
  stopifnot(is.logical(MIBoost), length(MIBoost) == 1L, !is.na(MIBoost))
  stopifnot(is.logical(pool),    length(pool)    == 1L, !is.na(pool))
  MIBoost <- isTRUE(MIBoost)
  pool    <- isTRUE(pool)

  center <- match.arg(center, c("auto","off","force"))

  has_val <- !is.null(X_list_val) && !is.null(y_list_val)
  if (has_val) {
    stopifnot(is.list(X_list_val), is.list(y_list_val),
              length(X_list_val) == M, length(y_list_val) == M)
  }

  ## ---- helpers --------------------------------------------------------------
  as01 <- function(y) {
    if (is.factor(y) || is.character(y)) return(as.numeric(as.character(y)))
    as.numeric(y)
  }

  center_matrix <- function(X, means = NULL, do_center = TRUE) {
    if (!do_center) {
      return(list(Xc = data.matrix(X), means = rep(0, ncol(X))))
    }
    X <- data.matrix(X)
    if (is.null(means)) {
      means <- colMeans(X, na.rm = FALSE)
    }
    list(Xc = sweep(X, 2L, means, FUN = "-"), means = means)
  }

  # logistic working response step for a single predictor
  glm1_binomial_coef <- function(x, y, f) {
    # y in {0,1}, f = linear predictor
    p <- stats::plogis(f)
    w <- p * (1 - p)
    # guard: if all weights ~ 0, no update
    if (all(w <= .Machine$double.eps)) return(0)
    z <- f + (y - p) / pmax(w, .Machine$double.eps)
    # weighted least squares for single x (with implicit intercept handled separately)
    # coef = argmin sum w * (z - (a + b*x))^2; we only return slope b here
    wx <- w * x
    wz <- w * z
    sx2 <- sum(wx * x)
    sxz <- sum(wx * z)
    sx  <- sum(wx)
    sz  <- sum(wz)
    n   <- sum(w)
    # Solve for slope in 2x2 normal equations:
    # [ sum(w)   sum(wx) ] [a] = [ sum(wz) ]
    # [ sum(wx)  sum(w x^2) ] [b] = [ sum(w x z) ]
    # We only need b:
    denom <- (n * sx2 - sx * sx)
    if (abs(denom) <= .Machine$double.eps) return(0)
    b <- (n * sxz - sx * sz) / denom
    b
  }

  ## ---- coerce inputs, basic sizes ------------------------------------------
  # convert to matrices; keep column names
  X_list  <- lapply(X_list, function(X) data.matrix(X))
  p       <- ncol(X_list[[1L]])
  stopifnot(p >= 1L)
  colnms  <- colnames(X_list[[1L]])

  # y handling
  if (type == "logistic") {
    y_list <- lapply(y_list, function(y) as01(y))
  } else {
    y_list <- lapply(y_list, function(y) as.numeric(y))
  }

  if (has_val) {
    X_list_val <- lapply(X_list_val, function(X) data.matrix(X))
    if (type == "logistic") {
      y_list_val <- lapply(y_list_val, function(y) as01(y))
    } else {
      y_list_val <- lapply(y_list_val, function(y) as.numeric(y))
    }
  }

  ## ---- centering ------------------------------------------------------------
  # Rule:
  # - "off": do not center
  # - "force": center all training Xs
  # - "auto": center if training columns are not already mean-zero
  center_do <- switch(center,
                      "off"   = FALSE,
                      "force" = TRUE,
                      "auto"  = {
                        # detect on first dataset
                        m0 <- colMeans(X_list[[1L]])
                        any(abs(m0) > 1e-12)
                      })

  center_means_list <- vector("list", M)
  for (m in seq_len(M)) {
    cm <- center_matrix(X_list[[m]], do_center = center_do)
    X_list[[m]] <- cm$Xc
    center_means_list[[m]] <- cm$means
    if (has_val) {
      X_list_val[[m]] <- center_matrix(X_list_val[[m]], means = cm$means, do_center = center_do)$Xc
    }
  }
  # convenience pooled means (only meaningful when 'pool = TRUE')
  center_means <- if (center_do) {
    # average of imputation-specific means (same length p)
    Reduce("+", center_means_list) / M
  } else NULL

  ## ---- initial intercepts and state -----------------------------------------
  # working predictors f (one per imputation)
  f_tr  <- vector("list", M)
  f_va  <- if (has_val) vector("list", M) else NULL

  # initialize intercept(s)
  if (type == "gaussian") {
    INT  <- vapply(seq_len(M), function(m) mean(y_list[[m]]), numeric(1))
  } else { # logistic
    mu   <- vapply(seq_len(M), function(m) {
      ym <- y_list[[m]]
      # stabilize mean into (0,1)
      mval <- mean(ym)
      mval <- min(max(mval, 1e-6), 1 - 1e-6)
      mval
    }, numeric(1))
    INT  <- stats::qlogis(mu)
  }
  for (m in seq_len(M)) {
    f_tr[[m]] <- rep(INT[m], length(y_list[[m]]))
    if (has_val) f_va[[m]] <- rep(INT[m], length(y_list_val[[m]]))
  }

  # coefficients
  if (pool) {
    BETA <- numeric(p)
    names(BETA) <- colnms
  } else {
    BETA <- matrix(0, nrow = M, ncol = p,
                   dimnames = list(NULL, colnms))
  }

  # bookkeeping for selection frequency (per-dataset)
  if (!MIBoost && !pool && p > 0) {
    sel_freq <- matrix(0L, nrow = M, ncol = p)
  }

  ## ---- loss helpers ----------------------------------------------------------
  gaussian_rss <- function(y, f) sum((y - f)^2)
  logistic_dev <- function(y, f) {
    p <- stats::plogis(f)
    eps <- 1e-12
    sum(-2 * (y * log(p + eps) + (1 - y) * log(1 - p + eps)))
  }

  fold_loss <- function(use_val = FALSE) {
    if (type == "gaussian") {
      if (use_val) {
        mean(vapply(seq_len(M), function(m) gaussian_rss(y_list_val[[m]], f_va[[m]]), numeric(1)))
      } else {
        mean(vapply(seq_len(M), function(m) gaussian_rss(y_list[[m]], f_tr[[m]]), numeric(1)))
      }
    } else {
      if (use_val) {
        mean(vapply(seq_len(M), function(m) logistic_dev(y_list_val[[m]], f_va[[m]]), numeric(1)))
      } else {
        mean(vapply(seq_len(M), function(m) logistic_dev(y_list[[m]], f_tr[[m]]), numeric(1)))
      }
    }
  }

  ## ---- main boosting loop ----------------------------------------------------
  CV_error <- if (has_val) numeric(mstop) else NULL

  for (t in seq_len(mstop)) {
    # ----- choose coordinate r* -----
    if (MIBoost) {
      # evaluate aggregated loss over imputations for each variable r
      agg_loss <- rep(NA_real_, p)

      for (r in seq_len(p)) {
        loss_sum <- 0
        for (m in seq_len(M)) {
          x <- X_list[[m]][, r]
          if (type == "gaussian") {
            # best 1D least squares step on residuals
            res <- y_list[[m]] - f_tr[[m]]
            denom <- sum(x * x)
            if (denom <= .Machine$double.eps) next
            alpha <- sum(x * res) / denom
            f_new <- f_tr[[m]] + ny * alpha * x
            loss_sum <- loss_sum + gaussian_rss(y_list[[m]], f_new)
          } else {
            b <- glm1_binomial_coef(x, y_list[[m]], f_tr[[m]])
            f_new <- f_tr[[m]] + ny * b * x
            loss_sum <- loss_sum + logistic_dev(y_list[[m]], f_new)
          }
        }
        agg_loss[r] <- loss_sum
      }
      r_star <- which.min(agg_loss)

      # ----- update along r* for all imputations -----
      for (m in seq_len(M)) {
        x <- X_list[[m]][, r_star]
        if (type == "gaussian") {
          res   <- y_list[[m]] - f_tr[[m]]
          denom <- sum(x * x)
          if (denom > .Machine$double.eps) {
            alpha <- sum(x * res) / denom
            f_tr[[m]] <- f_tr[[m]] + ny * alpha * x
            if (has_val) f_va[[m]] <- f_va[[m]] + ny * alpha * X_list_val[[m]][, r_star]
            if (pool) BETA[r_star] <- BETA[r_star] + ny * alpha
          }
        } else {
          b <- glm1_binomial_coef(x, y_list[[m]], f_tr[[m]])
          f_tr[[m]] <- f_tr[[m]] + ny * b * x
          if (has_val) f_va[[m]] <- f_va[[m]] + ny * b * X_list_val[[m]][, r_star]
          if (pool) BETA[r_star] <- BETA[r_star] + ny * b
        }
      }
      if (!pool) {
        # store per-imputation betas when not pooling
        for (m in seq_len(M)) {
          if (type == "gaussian") {
            res   <- y_list[[m]] - (f_tr[[m]] - ny * 0) # no need to recompute; we add increment
          }
          # recompute increment for record
          x <- X_list[[m]][, r_star]
          if (type == "gaussian") {
            res   <- y_list[[m]] - (f_tr[[m]] - ny * 0) # placeholder; see below
            # For bookkeeping, get alpha on current residuals (approximate)
            denom <- sum(x * x)
            alp <- if (denom > .Machine$double.eps) sum(x * (y_list[[m]] - (f_tr[[m]] - ny * 0))) / denom else 0
            BETA[m, r_star] <- BETA[m, r_star] + ny * alp
          } else {
            b <- glm1_binomial_coef(x, y_list[[m]], f_tr[[m]] - ny * 0)
            BETA[m, r_star] <- BETA[m, r_star] + ny * b
          }
        }
      }
    } else {
      # per-dataset independent coordinate descent
      for (m in seq_len(M)) {
        best_r <- NA_integer_
        best_l <- Inf
        for (r in seq_len(p)) {
          x <- X_list[[m]][, r]
          if (type == "gaussian") {
            res   <- y_list[[m]] - f_tr[[m]]
            denom <- sum(x * x)
            if (denom <= .Machine$double.eps) next
            alpha <- sum(x * res) / denom
            f_new <- f_tr[[m]] + ny * alpha * x
            lval  <- gaussian_rss(y_list[[m]], f_new)
          } else {
            b   <- glm1_binomial_coef(x, y_list[[m]], f_tr[[m]])
            f_new <- f_tr[[m]] + ny * b * x
            lval  <- logistic_dev(y_list[[m]], f_new)
          }
          if (lval < best_l) { best_l <- lval; best_r <- r }
        }
        # update along best_r
        r_star_m <- best_r
        x <- X_list[[m]][, r_star_m]
        if (type == "gaussian") {
          res   <- y_list[[m]] - f_tr[[m]]
          denom <- sum(x * x)
          if (denom > .Machine$double.eps) {
            alpha <- sum(x * res) / denom
            f_tr[[m]] <- f_tr[[m]] + ny * alpha * x
            if (has_val) f_va[[m]] <- f_va[[m]] + ny * alpha * X_list_val[[m]][, r_star_m]
            BETA[m, r_star_m] <- BETA[m, r_star_m] + ny * alpha
            sel_freq[m, r_star_m] <- sel_freq[m, r_star_m] + 1L
          }
        } else {
          b <- glm1_binomial_coef(x, y_list[[m]], f_tr[[m]])
          f_tr[[m]] <- f_tr[[m]] + ny * b * x
          if (has_val) f_va[[m]] <- f_va[[m]] + ny * b * X_list_val[[m]][, r_star_m]
          BETA[m, r_star_m] <- BETA[m, r_star_m] + ny * b
          sel_freq[m, r_star_m] <- sel_freq[m, r_star_m] + 1L
        }
      }
    }

    # track validation loss
    if (has_val) {
      CV_error[t] <- fold_loss(use_val = TRUE)
    }
  } # end boosting loop

  ## ---- assemble outputs ------------------------------------------------------
  if (has_val) {
    out <- list(CV_error = CV_error)
  } else {
    # Final coefficients / pooling
    if (pool) {
      # when MIBoost=FALSE but pool=TRUE, aggregate per-dataset betas
      if (!MIBoost) {
        # estimate averaging by default (pool_threshold == 0)
        if (!exists("sel_freq", inherits = FALSE)) {
          # if we didn't run the per-dataset path (e.g., MIBoost=TRUE), BETA is already pooled
          pooled_beta <- BETA
        } else {
          if (is.matrix(BETA)) {
            # average betas across imputations
            avg_beta <- colMeans(BETA, na.rm = TRUE)
            if (pool_threshold > 0) {
              freq <- colMeans(sel_freq > 0)
              keep <- freq >= pool_threshold
              avg_beta[!keep] <- 0
            }
            pooled_beta <- avg_beta
          } else {
            pooled_beta <- BETA
          }
        }
      } else {
        pooled_beta <- BETA
      }

      # pooled intercept: average of INT (works for both families reasonably)
      INT_pool <- mean(INT)

      out <- list(
        INT  = as.numeric(INT_pool),
        BETA = as.numeric(pooled_beta)
      )
      if (!is.null(center_means)) out$center_means <- center_means
    } else {
      # return per-imputation models
      out <- list(
        INT  = as.numeric(INT),
        BETA = BETA
      )
      if (!is.null(center_means_list)) out$center_means_list <- center_means_list
    }
  }

  out
}





