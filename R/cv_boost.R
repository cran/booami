#' Cross-validation for boosting after multiple imputation (pre-imputed inputs)
#'
#' Performs k-fold cross-validation for \code{\link{impu_boost}} to determine
#' the optimal value of \code{mstop} before fitting the final model on the
#' full dataset. This function should only be used when data have already
#' been imputed. In most cases, it is preferable to provide unimputed data
#' and use \code{\link{cv_boost_raw}} instead.
#'
#' To avoid data leakage, each CV fold should first be split into training and
#' validation subsets, after which imputation is performed. For the final model,
#' all data should be imputed independently.
#'
#' The recommended workflow is illustrated in the examples.
#'
#' @references
#' Kuchen, R. (2025). \emph{MIBoost: A Gradient Boosting Algorithm for Variable
#' Selection After Multiple Imputation}. arXiv:2507.21807.
#' \doi{10.48550/arXiv.2507.21807} \url{https://arxiv.org/abs/2507.21807}.
#'
#' @param X_train_list A list of length \eqn{k}. Element \code{i} is itself a
#'   list of length \eqn{M} containing the \eqn{n_{train} \times p} numeric
#'   design matrices for each imputed dataset in CV fold \code{i}.
#' @param y_train_list A list of length \eqn{k}. Element \code{i} is a list of
#'   length \eqn{M}, where each element is a length-\eqn{n_{train}} numeric
#'   response vector aligned with \code{X_train_list[[i]][[m]]}.
#' @param X_val_list A list of length \eqn{k}. Element \code{i} is a list of
#'   length \eqn{M} containing the \eqn{n_{val} \times p} numeric validation
#'   matrices matched to the corresponding imputed training dataset
#'   in fold \code{i}.
#' @param y_val_list A list of length \eqn{k}. Element \code{i} is a list of
#'   length \eqn{M}, where each element is a length-\eqn{n_{val}} continuous
#'   response vector aligned with \code{X_val_list[[i]][[m]]}.
#' @param X_full A list of length \eqn{M} with the \eqn{n \times p} numeric
#'   full-data design matrices (one per imputed dataset) used to fit
#'   the final model.
#' @param y_full A list of length \eqn{M}, where each element is a
#'   length-\eqn{n} continuous response vector corresponding to the
#'   imputed dataset in \code{X_full}.
#' @param ny Learning rate. Defaults to \code{0.1}.
#' @param mstop Maximum number of boosting iterations to evaluate during
#'   cross-validation. The selected \code{mstop} is the value that minimizes
#'   the mean CV error over \code{1:mstop}. Default is \code{250}.
#' @param type Type of loss function. One of:
#'   \code{"gaussian"} (mean squared error) for continuous responses,
#'   or \code{"logistic"} (binomial deviance) for binary responses.
#' @param MIBoost Logical. If \code{TRUE}, applies the MIBoost algorithm,
#'   which enforces uniform variable selection across all imputed datasets. If
#'   \code{FALSE}, variables are selected independently within each imputed
#'   dataset, and pooling is governed by \code{pool_threshold}.
#' @param pool Logical. If \code{TRUE}, models across the \eqn{M} imputed
#'   datasets are aggregated into a single final model. If \code{FALSE},
#'   \eqn{M} separate models are returned.
#' @param pool_threshold Only used when \code{MIBoost = FALSE} and \code{pool = TRUE}.
#'   Controls the pooling rule when aggregating the \eqn{M} models obtained from
#'   the imputed datasets into a single final model. A candidate variable is
#'   included only if it is selected in at least \code{pool_threshold} (a value
#'   in (0, 1]) proportion of the imputed datasets; coefficients of all other
#'   variables are set to zero. A value of \code{0} corresponds to
#'   estimate-averaging, while values \code{> 0} correspond to
#'   selection-frequency thresholding.
#' @param show_progress Logical; print fold-level progress and summary timings.
#'   Default \code{TRUE}.
#' @param center One of \code{c("auto", "off", "force")}. Controls centering of
#'   \code{X}.
#'   With \code{"auto"} (recommended), centering is applied only if the training
#'   data are not already centered (checked across imputations). With
#'   \code{"force"}, centering is always applied. With \code{"off"}, centering
#'   is skipped.
#'
#'   If centering is applied, a single grand mean vector \eqn{\mu_\star} is
#'   computed from the training imputations in the corresponding fold and then
#'   subtracted from \emph{all} imputed training and validation matrices in that
#'   fold (and analogously for the final model fit on \code{X_full}).
#'
#' @details
#' Centering affects only \code{X}; \code{y} is left unchanged. For
#' \code{type = "logistic"}, responses are treated as numeric \code{0/1}
#' via the logistic link. Validation loss is averaged over
#' imputations and then over folds.
#'
#' @return A list with:
#' \itemize{
#'   \item \code{CV_error}: numeric vector of length \code{mstop} with the mean
#'         cross-validated loss across folds (and imputations).
#'   \item \code{best_mstop}: integer index of the minimizing entry in
#'         \code{CV_error}.
#'   \item \code{final_model}: numeric vector of length \code{1 + p} containing
#'         the intercept followed by \eqn{p} coefficients of the final pooled
#'         model fitted at \code{best_mstop} on \code{X_full}/\code{y_full}.
#'   \item \code{center_means}: (optional) numeric vector of length \eqn{p}
#'         containing the centering means used for \code{X} (when available).
#' }
#'
#' @seealso \code{\link{impu_boost}}, \code{\link{cv_boost_raw}}
#'
#' @examplesIf requireNamespace("mice", quietly = TRUE) && requireNamespace("miceadds", quietly = TRUE)
#' \donttest{
#'   set.seed(123)
#'   utils::data(booami_sim)
#'   k <- 2; M <- 2
#'
#'   # Separate X and y; drop missing y (policy)
#'   X_all <- booami_sim[, 1:25, drop = FALSE]
#'   y_all <- booami_sim[, 26]
#'   keep <- !is.na(y_all)
#'   X_all <- X_all[keep, , drop = FALSE]
#'   y_all <- y_all[keep]
#'
#'   n <- nrow(X_all); p <- ncol(X_all)
#'   folds <- sample(rep(seq_len(k), length.out = n))
#'
#'   X_train_list <- vector("list", k)
#'   y_train_list <- vector("list", k)
#'   X_val_list   <- vector("list", k)
#'   y_val_list   <- vector("list", k)
#'
#'   for (cv in seq_len(k)) {
#'     tr <- folds != cv
#'     va <- !tr
#'     Xtr <- X_all[tr, , drop = FALSE]; ytr <- y_all[tr]
#'     Xva <- X_all[va, , drop = FALSE]; yva <- y_all[va]
#'
#'     # Impute X only (y is never used for imputation)
#'     pm_tr  <- mice::quickpred(Xtr, method = "spearman", mincor = 0.30, minpuc = 0.60)
#'     imp_tr <- mice::mice(Xtr, m = M, predictorMatrix = pm_tr, maxit = 1, printFlag = FALSE)
#'     imp_va <- mice::mice.mids(imp_tr, newdata = Xva, maxit = 1, printFlag = FALSE)
#'
#'     X_train_list[[cv]] <- vector("list", M)
#'     y_train_list[[cv]] <- vector("list", M)
#'     X_val_list[[cv]]   <- vector("list", M)
#'     y_val_list[[cv]]   <- vector("list", M)
#'
#'     for (m in seq_len(M)) {
#'       tr_m <- mice::complete(imp_tr, m)
#'       va_m <- mice::complete(imp_va, m)
#'       X_train_list[[cv]][[m]] <- data.matrix(tr_m)
#'       y_train_list[[cv]][[m]] <- ytr
#'       X_val_list[[cv]][[m]]   <- data.matrix(va_m)
#'       y_val_list[[cv]][[m]]   <- yva
#'     }
#'   }
#'
#'   # Full-data imputations (X only)
#'   pm_full  <- mice::quickpred(X_all, method = "spearman", mincor = 0.30, minpuc = 0.60)
#'   imp_full <- mice::mice(X_all, m = M, predictorMatrix = pm_full, maxit = 1, printFlag = FALSE)
#'   X_full <- lapply(seq_len(M), function(m) data.matrix(mice::complete(imp_full, m)))
#'   y_full <- lapply(seq_len(M), function(m) y_all)
#'
#'   res <- cv_boost_imputed(
#'     X_train_list, y_train_list,
#'     X_val_list,   y_val_list,
#'     X_full,       y_full,
#'     ny = 0.1, mstop = 50, type = "gaussian",
#'     MIBoost = TRUE, pool = TRUE, center = "auto",
#'     show_progress = FALSE
#'   )
#'   \dontshow{invisible(utils::capture.output(str(res)))}
#' }
#'
#' \donttest{
#'   set.seed(2025)
#'   utils::data(booami_sim)
#'   k <- 5; M <- 10
#'
#'   X_all <- booami_sim[, 1:25, drop = FALSE]
#'   y_all <- booami_sim[, 26]
#'   keep <- !is.na(y_all)
#'   X_all <- X_all[keep, , drop = FALSE]
#'   y_all <- y_all[keep]
#'
#'   n <- nrow(X_all); p <- ncol(X_all)
#'   folds <- sample(rep(seq_len(k), length.out = n))
#'
#'   X_train_list <- vector("list", k)
#'   y_train_list <- vector("list", k)
#'   X_val_list   <- vector("list", k)
#'   y_val_list   <- vector("list", k)
#'
#'   for (cv in seq_len(k)) {
#'     tr <- folds != cv; va <- !tr
#'     Xtr <- X_all[tr, , drop = FALSE]; ytr <- y_all[tr]
#'     Xva <- X_all[va, , drop = FALSE]; yva <- y_all[va]
#'
#'     pm_tr  <- mice::quickpred(Xtr, method = "spearman", mincor = 0.20, minpuc = 0.40)
#'     imp_tr <- mice::mice(Xtr, m = M, predictorMatrix = pm_tr, maxit = 5, printFlag = TRUE)
#'     imp_va <- mice::mice.mids(imp_tr, newdata = Xva, maxit = 1, printFlag = FALSE)
#'
#'     X_train_list[[cv]] <- vector("list", M)
#'     y_train_list[[cv]] <- vector("list", M)
#'     X_val_list[[cv]]   <- vector("list", M)
#'     y_val_list[[cv]]   <- vector("list", M)
#'
#'     for (m in seq_len(M)) {
#'       tr_m <- mice::complete(imp_tr, m)
#'       va_m <- mice::complete(imp_va, m)
#'       X_train_list[[cv]][[m]] <- data.matrix(tr_m)
#'       y_train_list[[cv]][[m]] <- ytr
#'       X_val_list[[cv]][[m]]   <- data.matrix(va_m)
#'       y_val_list[[cv]][[m]]   <- yva
#'     }
#'   }
#'
#'   pm_full  <- mice::quickpred(X_all, method = "spearman", mincor = 0.20, minpuc = 0.40)
#'   imp_full <- mice::mice(X_all, m = M, predictorMatrix = pm_full, maxit = 5, printFlag = TRUE)
#'   X_full <- lapply(seq_len(M), function(m) data.matrix(mice::complete(imp_full, m)))
#'   y_full <- lapply(seq_len(M), function(m) y_all)
#'
#'   res_heavy <- cv_boost_imputed(
#'     X_train_list, y_train_list,
#'     X_val_list,   y_val_list,
#'     X_full,       y_full,
#'     ny = 0.1, mstop = 250, type = "gaussian",
#'     MIBoost = TRUE, pool = TRUE, center = "auto",
#'     show_progress = TRUE
#'   )
#'   str(res_heavy)
#' }
#' @export
cv_boost_imputed <- function(
    X_train_list, y_train_list,
    X_val_list,   y_val_list,
    X_full, y_full,
    ny = 0.1, mstop = 250, type = c("gaussian","logistic"),
    MIBoost = TRUE, pool = TRUE, pool_threshold = 0,
    show_progress = TRUE,
    center = c("auto","off","force")
) {
  # match args (keep same order/choices as core)
  type   <- match.arg(type)
  center <- match.arg(center)

  # ---- light validation (fold-level) ----
  K <- length(y_train_list)
  stopifnot(
    length(X_train_list) == K,
    length(X_val_list)   == K,
    length(y_val_list)   == K,
    K >= 2L
  )
  if (!is.numeric(ny) || length(ny) != 1L || !is.finite(ny) || ny <= 0) {
    stop("`ny` must be a single positive numeric.")
  }
  if (!is.numeric(mstop) || length(mstop) != 1L || !is.finite(mstop) || mstop < 1) {
    stop("`mstop` must be a single positive integer.")
  }
  mstop <- as.integer(mstop)

  # ensure each fold has same number of imputations M and basic shapes look sane
  Ms <- integer(K)
  for (cv in seq_len(K)) {
    Ms[cv] <- length(X_train_list[[cv]])
    if (length(y_train_list[[cv]]) != Ms[cv] ||
        length(X_val_list[[cv]])   != Ms[cv] ||
        length(y_val_list[[cv]])   != Ms[cv]) {
      stop(sprintf("Fold %d: inconsistent number of imputations across X/y train/val.", cv))
    }
  }
  if (length(unique(Ms)) != 1L) stop("All folds must have the same number of imputations M.")
  M <- Ms[1L]

  # full data lists should also have M imputations
  if (length(X_full) != M || length(y_full) != M) {
    stop("`X_full` and `y_full` must each be lists of length M (same M as the folds).")
  }

  # basic logistic target check (silent coercion not performed)
  if (type == "logistic") {
    ok01 <- function(v) all(is.finite(v)) && all(v %in% c(0,1))
    if (!ok01(y_train_list[[1]][[1]]) || !ok01(y_val_list[[1]][[1]]) || !ok01(y_full[[1]])) {
      stop("For type='logistic', outcomes must be 0/1 (no NAs, no other values).")
    }
  }

  # ---- delegate to the core ----
  res <- .cv_boost_core(
    X_train_list = X_train_list, y_train_list = y_train_list,
    X_val_list   = X_val_list,   y_val_list   = y_val_list,
    X = X_full, y = y_full,
    ny = ny, mstop = mstop, type = type,
    MIBoost = MIBoost, pool = pool, pool_threshold = pool_threshold,
    show_progress = show_progress,
    center = center
  )


  # ---- friendly metadata & names for prediction -------------------
  # 1) Annotate family type for booami_predict() inference
  if (is.null(res$type)) res$type <- type

  # 2) Bubble up center_means if the core exposed them (preferred)
  #    a) directly on the returned object
  #    b) or nested in a `final_fit` list (if the core returns the impu_boost fit)
  if (is.null(res$center_means)) {
    if (!is.null(res$final_fit) && !is.null(res$final_fit$center_means)) {
      res$center_means <- res$final_fit$center_means
    }
  }
  # (No attempt here to "recompute" means from X_full - we only propagate what the core actually used.)

  # 3) Name final_model coefficients for name-based alignment in prediction
  if (!is.null(res$final_model)) {
    pred_names <- colnames(X_full[[1L]])
    if (!is.null(pred_names) &&
        length(res$final_model) == (length(pred_names) + 1L) &&
        is.null(names(res$final_model))) {
      names(res$final_model) <- c("(Intercept)", pred_names)
    }
  }

  class(res) <- c("booami_cv_imputed", "booami_cv", class(res))

  res
}



#' Cross-Validated Component-Wise Gradient Boosting with
#' Multiple Imputation Performed Inside Each Fold
#'
#' Performs k-fold cross-validation for \code{\link{impu_boost}} on data with
#' missing values. Within each fold, multiple imputation, centering, model
#' fitting, and validation are performed in a leakage-avoiding manner to select
#' the optimal number of boosting iterations (\code{mstop}). The final model is
#' then fitted to the multiply imputed datasets derived from the full dataset
#' at the selected stopping iteration.
#'
#' @details
#' Rows with missing outcomes \code{y} are removed before fold assignment.
#' Within each CV fold, the remaining data are first split into a training subset
#' and a validation subset. Multiple imputation is then performed on the
#' covariates \code{X} only (the outcome is never imputed and is not used as a
#' predictor in the imputation models). The training covariates are multiply
#' imputed \eqn{M} times using \pkg{mice}, producing \eqn{M} imputed training
#' datasets. The corresponding validation covariates are then imputed \eqn{M}
#' times using the imputation models learned from the training data (leakage-avoiding).
#'
#' If centering is applied, a single grand mean vector \eqn{\mu_\star} is
#' computed from the imputed training covariates in the corresponding fold and
#' subtracted from \emph{all} imputed training and validation covariate matrices
#' in that fold.
#'
#' \code{\link{impu_boost}} is run on the imputed training datasets for up to
#' \code{mstop} boosting iterations. At each iteration, prediction errors are
#' computed on the corresponding validation datasets and averaged across
#' imputations. This yields an aggregated error curve per fold, which is then
#' averaged across folds. The optimal stopping iteration is chosen as the
#' \code{mstop} value minimizing the mean CV error.
#'
#' Finally, the full covariate matrix \code{X} is multiply imputed \eqn{M} times.
#' If centering is applied, it uses a grand mean \eqn{\mu_\star} computed across
#' the \eqn{M} full-data imputations. \code{\link{impu_boost}} is applied to these
#' datasets for the selected number of boosting iterations to obtain the final model.
#'
#' \strong{Imputation control.} All key \pkg{mice} settings can be passed via
#' \code{impute_args} (a named list forwarded to \code{mice::mice()}) and/or
#' \code{impute_method} (a named character vector of per-variable methods).
#' Internally, the function builds a full default method vector from the actual
#' data given to \code{mice()}, then \emph{merges} any user-supplied entries
#' by name. \emph{The names in \code{impute_method} must exactly match the
#' column names in \code{X} (i.e., the data passed to \code{mice()}).} Partial
#' vectors are allowed; variables not listed fall back to defaults; unknown names
#' are ignored with a warning. The function sets and may override \code{data},
#' \code{method} (after merging overrides), \code{predictorMatrix}, and
#' \code{ignore} (to enforce train-only learning). Predictor matrices can be built
#' with \code{mice::quickpred()} (see \code{use_quickpred}, \code{quickpred_args})
#' or with \code{mice::make.predictorMatrix()}.
#'
#' @references
#' Kuchen, R. (2025). \emph{MIBoost: A Gradient Boosting Algorithm for Variable
#' Selection After Multiple Imputation}. arXiv:2507.21807.
#' \doi{10.48550/arXiv.2507.21807} \url{https://arxiv.org/abs/2507.21807}.
#'
#' @param X A data.frame or matrix of predictors of size \eqn{n \times p}
#'   containing missing values. Column names are preserved. Rows with missing
#'   outcomes \code{y} are removed before CV. If no missing values are present in
#'   \code{X} after removing missing-\code{y} rows, use
#'   \code{\link{cv_boost_imputed}} instead.
#' @param y A vector of length \eqn{n} with the outcome (numeric for
#'   \code{type = "gaussian"}; numeric \code{0/1} or a 2-level factor for
#'   \code{type = "logistic"}). Must align with \code{X} rows. Rows with missing
#'   \code{y} are removed before CV. The outcome is never imputed and is not used
#'   as a predictor in imputation models.
#' @param k Number of cross-validation folds. Default is \code{5}.
#' @param ny Learning rate. Defaults to \code{0.1}.
#' @param mstop Maximum number of boosting iterations to evaluate during
#'   cross-validation. The selected \code{mstop} is the value minimizing
#'   the mean CV error over \code{1:mstop}. Default is \code{250}.
#' @param type Type of loss function. One of:
#'   \code{"gaussian"} (mean squared error) for continuous responses,
#'   or \code{"logistic"} (binomial deviance) for binary responses.
#' @param MIBoost Logical. If \code{TRUE}, applies the MIBoost algorithm,
#'   which enforces uniform variable selection across all imputed datasets. If
#'   \code{FALSE}, variables are selected independently within each imputed
#'   dataset, and pooling is governed by \code{pool_threshold}.
#' @param pool Logical. If \code{TRUE}, models across the \eqn{M} imputed
#'   datasets are aggregated into a single final model. If \code{FALSE},
#'   \eqn{M} separate models are returned.
#' @param pool_threshold Only used when \code{MIBoost = FALSE} and
#'   \code{pool = TRUE}. Controls the pooling rule when aggregating the
#'   \eqn{M} models obtained from the imputed datasets into a single final model.
#'   A candidate variable is included only if it is selected in at least
#'   \code{pool_threshold} (a value in (0, 1) proportion of the imputed
#'   datasets; coefficients of all other variables are set to zero. A value of
#'   \code{0} corresponds to estimate-averaging, while values \code{> 0}
#'   correspond to selection-frequency thresholding.
#' @param impute_args A named list of arguments forwarded to \code{mice::mice()}
#'   both inside CV and on the full dataset (e.g., \code{m}, \code{maxit},
#'   \code{seed}, \code{printFlag}, etc.). Internally, \code{data},
#'   \code{predictorMatrix}, and \code{ignore} are set by the function and will
#'   override any values supplied here. If \code{m} is missing, a default of
#'   \code{10} is used.
#' @param impute_method Optional \emph{named} character vector passed to
#'   \code{mice::mice(method = ...)} to control per-variable methods for
#'   covariates \code{X} (e.g., \code{"pmm"}, \code{"logreg"}). This may be a
#'   \emph{partial} vector: entries are merged by name into a full default method
#'   vector derived from \code{X}; unmatched names are ignored with a warning.
#'   If \code{NULL} (default), numeric covariates use \code{"pmm"}.
#' @param use_quickpred Logical. If \code{TRUE} (default), build the
#'   \code{predictorMatrix} via \code{mice::quickpred()} on the training covariates
#'   within each fold; otherwise use \code{mice::make.predictorMatrix()}.
#' @param quickpred_args A named list of arguments forwarded to
#'   \code{mice::quickpred()} (e.g., \code{mincor}, \code{minpuc},
#'   \code{method}, \code{include}, \code{exclude}). Ignored when
#'   \code{use_quickpred = FALSE}.
#' @param seed Base random seed for fold assignment. If \code{impute_args$seed}
#'   is \emph{not} supplied, this value also seeds imputation; otherwise the
#'   user-specified \code{impute_args$seed} is respected and deterministically
#'   offset per fold. RNG state is restored on exit. Default \code{123}.
#' @param show_progress Logical. If \code{TRUE} (default), print progress for
#'   the imputation and boosting phases, plus a summary at completion.
#' @param return_full_imputations Logical. If \code{TRUE}, attach the list of
#'   full-data imputations used for the final fit as
#'   \code{$full_imputations = list(X = <list length m>, y = <list length m>)}.
#'   Default is \code{FALSE}.
#' @param center One of \code{c("auto", "off", "force")}. Controls centering of
#'   \code{X}. With \code{"auto"} (recommended), centering is applied only if the
#'   training data are not already centered (checked across imputations). With
#'   \code{"force"}, centering is always applied. With \code{"off"}, centering is
#'   skipped.
#'
#'   If centering is applied, a single grand mean vector \eqn{\mu_\star} is
#'   computed from the imputed training covariates in the corresponding fold and
#'   subtracted from all imputed training and validation matrices in that fold
#'   (and analogously for the final model fit on the full-data imputations).
#'
#' @return A list with:
#' \itemize{
#'   \item \code{CV_error}: numeric vector (length \code{mstop}) of mean CV loss.
#'   \item \code{best_mstop}: integer index minimizing \code{CV_error}.
#'   \item \code{final_model}: numeric vector of length \code{1 + p} with the
#'         intercept and pooled coefficients of the final fit on full-data
#'         imputations at \code{best_mstop}.
#'   \item \code{full_imputations}: (optional) when \code{return_full_imputations=TRUE},
#'         a list \code{list(X = <list length m>, y = <list length m>)} containing
#'         the full-data imputations used for the final model.
#'   \item \code{folds}: integer vector of length \eqn{n} giving the CV fold id
#'         for each observation (1..k).
#' }
#'
#' @seealso \code{\link{impu_boost}}, \code{\link{cv_boost_imputed}}, \pkg{mice}
#'
#' @examplesIf requireNamespace("mice", quietly = TRUE) && requireNamespace("miceadds", quietly = TRUE)
#' \donttest{
#'   utils::data(booami_sim)
#'   X <- booami_sim[, 1:25]
#'   y <- booami_sim[, 26]
#'
#'   res <- cv_boost_raw(
#'     X = X, y = y,
#'     k = 2, seed = 123,
#'     impute_args    = list(m = 2, maxit = 1, printFlag = FALSE, seed = 1),
#'     quickpred_args = list(mincor = 0.30, minpuc = 0.60),
#'     mstop = 50,
#'     show_progress = FALSE
#'   )
#'
#'   # Partial custom imputation method override (X variables only)
#'   meth <- c(X1 = "pmm")
#'   res2 <- cv_boost_raw(
#'     X = X, y = y,
#'     k = 2, seed = 123,
#'     impute_args    = list(m = 2, maxit = 1, printFlag = FALSE, seed = 456),
#'     quickpred_args = list(mincor = 0.30, minpuc = 0.60),
#'     mstop = 50,
#'     impute_method  = meth,
#'     show_progress = FALSE
#'   )
#'   \dontshow{invisible(utils::capture.output(str(res2)))}
#' }
#'
#' @export
cv_boost_raw <- function(
    X, y,
    k = 5, ny = 0.1, mstop = 250, type = c("gaussian","logistic"),
    MIBoost = TRUE, pool = TRUE, pool_threshold = 0,
    impute_args = list(m = 10, maxit = 5, printFlag = FALSE),
    impute_method = NULL,
    use_quickpred = TRUE,
    quickpred_args = list(mincor = 0.1, minpuc = 0.5, method = NULL, include = NULL, exclude = NULL),
    seed = 123,
    show_progress = TRUE,
    return_full_imputations = FALSE,
    center = "auto"
) {
  type   <- match.arg(type)
  center <- match.arg(center, c("auto","off","force"))

  if (!is.data.frame(X) && !is.matrix(X))
    stop("X must be a data.frame or matrix.")
  X <- as.data.frame(X, check.names = FALSE)

  if (length(y) != nrow(X))
    stop("Length of y must match nrow(X).")

  # --- drop rows with missing y (so y is never imputed, and cannot be used) ---
  keep <- !is.na(y)
  if (!any(keep)) stop("All rows have missing y; nothing to fit.")
  if (any(!keep) && isTRUE(show_progress)) {
    cat(sprintf("Dropping %d rows with missing y...\n", sum(!keep)))
  }
  X <- X[keep, , drop = FALSE]
  y <- y[keep]

  # after dropping missing y, ensure y is finite
  if (any(!is.finite(as.numeric(y)))) stop("Non-finite values in y after dropping NAs.")

  # logistic guard: must be 0/1 with no NAs
  if (type == "logistic") {
    y_num <- as.numeric(y)
    if (!all(y_num %in% c(0, 1))) {
      # also allow factor levels "0"/"1" etc.
      if (is.factor(y) || is.character(y)) {
        y_num2 <- suppressWarnings(as.numeric(as.character(y)))
        if (!all(is.finite(y_num2)) || !all(y_num2 %in% c(0, 1))) {
          stop("For type='logistic', y must be coded 0/1 (no NAs).")
        }
        y <- y_num2
      } else {
        stop("For type='logistic', y must be coded 0/1 (no NAs).")
      }
    } else {
      y <- y_num
    }
  } else {
    y <- as.numeric(y)
  }

  # Now we only care whether X has missingness
  if (!anyNA(X)) {
    stop("After removing missing y, X has no missing values. Use cv_boost_imputed() (or a complete-data variant).")
  }

  if (!("ignore" %in% names(formals(mice::mice))))
    stop("Update 'mice': need 'ignore' in mice().")

  n <- nrow(X)

  # --- folds: reproducible WITHOUT altering the user's RNG state
  folds_id <- withr::with_preserve_seed({
    seq_rep <- rep(seq_len(k), length.out = n)
    if (is.null(seed)) {
      sample(seq_rep)
    } else {
      withr::with_seed(seed, sample(seq_rep))
    }
  })

  if (!is.logical(pool) || length(pool) != 1L) {
    stop("'pool' must be a single TRUE/FALSE.")
  }

  seed_base_mice <- if (!is.null(impute_args$seed)) impute_args$seed else seed
  get_m <- function() if (!is.null(impute_args$m)) impute_args$m else 10L

  # methods builder (NO y column here)
  build_methods_X <- function(datX) {
    meth <- mice::make.method(datX)
    is_num <- vapply(datX, is.numeric, logical(1L))
    meth[is_num] <- "pmm"

    # allow user override for X columns
    if (!is.null(impute_method)) {
      nm <- intersect(names(impute_method), names(meth))
      if (length(nm) == 0L) {
        warning("None of the names in 'impute_method' matched the X columns; overrides ignored.")
      } else {
        meth[nm] <- impute_method[nm]
      }
    }
    meth
  }

  m <- get_m()
  if (isTRUE(show_progress)) cat(sprintf("MI-in-CV (no y in imputation): %d folds  x  %d imputations...\n", k, m))

  X_train_list <- vector("list", k); y_train_list <- vector("list", k)
  X_val_list   <- vector("list", k); y_val_list   <- vector("list", k)

  for (i in seq_len(k)) {
    tr_idx <- folds_id != i; va_idx <- !tr_idx
    Xtr <- X[tr_idx, , drop = FALSE]; ytr <- y[tr_idx]
    Xva <- X[va_idx, , drop = FALSE]; yva <- y[va_idx]

    # IMPORTANT: impute only X (no y column at all)
    dat_tr  <- Xtr
    dat_va  <- Xva
    dat_all <- rbind(dat_tr, dat_va)
    ntr <- nrow(dat_tr)

    if (isTRUE(use_quickpred)) {
      qp <- quickpred_args
      qp <- qp[!vapply(qp, is.null, logical(1))]
      pm <- do.call(mice::quickpred, c(list(data = dat_tr), qp))
    } else {
      pm <- mice::make.predictorMatrix(dat_tr)
    }

    meth <- build_methods_X(dat_all)

    ignore_vec <- c(rep(FALSE, ntr), rep(TRUE, nrow(dat_va)))
    args <- utils::modifyList(impute_args, list(
      data   = dat_all,
      method = meth,
      predictorMatrix = pm,
      ignore = ignore_vec
    ), keep.null = TRUE)

    if (is.null(args$m)) args$m <- m
    if (is.null(args$seed)) {
      args$seed <- seed_base_mice + i
    } else {
      args$seed <- args$seed + i
    }

    if (isTRUE(show_progress)) cat(sprintf("Fold %d: mice on X only (m=%d)...\n", i, args$m))
    mids <- do.call(mice::mice, args)

    X_train_list[[i]] <- vector("list", args$m)
    y_train_list[[i]] <- vector("list", args$m)
    X_val_list[[i]]   <- vector("list", args$m)
    y_val_list[[i]]   <- vector("list", args$m)

    for (j in seq_len(args$m)) {
      comp <- mice::complete(mids, action = j)
      comp_tr <- comp[seq_len(ntr), , drop = FALSE]
      comp_va <- comp[(ntr + 1):nrow(comp), , drop = FALSE]

      X_train_list[[i]][[j]] <- data.matrix(comp_tr)
      X_val_list[[i]][[j]]   <- data.matrix(comp_va)

      # y is NOT imputed; just replicate observed y per imputation
      y_train_list[[i]][[j]] <- ytr
      y_val_list[[i]][[j]]   <- yva
    }
  }

  # --- Full-data imputations (again: X only, no y in mice) ---
  if (isTRUE(show_progress)) cat("Full-data imputations (X only)...\n")
  dat_full <- X

  if (isTRUE(use_quickpred)) {
    qp <- quickpred_args
    qp <- qp[!vapply(qp, is.null, logical(1))]
    pm_full <- do.call(mice::quickpred, c(list(data = dat_full), qp))
  } else {
    pm_full <- mice::make.predictorMatrix(dat_full)
  }

  meth_full <- build_methods_X(dat_full)

  args_full <- utils::modifyList(impute_args, list(
    data = dat_full,
    method = meth_full,
    predictorMatrix = pm_full
  ), keep.null = TRUE)

  if (is.null(args_full$m)) args_full$m <- m
  if (is.null(args_full$seed)) {
    args_full$seed <- seed_base_mice + 10^6
  } else {
    args_full$seed <- args_full$seed + 10^6
  }

  mids_full <- do.call(mice::mice, args_full)

  X_list <- vector("list", args_full$m)
  y_list <- vector("list", args_full$m)
  for (j in seq_len(args_full$m)) {
    comp_full <- mice::complete(mids_full, j)
    X_list[[j]] <- data.matrix(comp_full)
    y_list[[j]] <- y  # replicated observed y
  }

  if (isTRUE(show_progress))
    cat(sprintf("Boosting: k=%d, m=%d; mstop=%d...\n", k, m, mstop))

  res <- .cv_boost_core(
    X_train_list, y_train_list,
    X_val_list,   y_val_list,
    X_list,  y_list,
    ny, mstop, type, MIBoost, pool, pool_threshold,
    show_progress = show_progress,
    center = center
  )

  class(res) <- c("booami_cv_raw", "booami_cv", class(res))

  if (is.null(res$type)) res$type <- type
  if (is.null(res$center_means) && !is.null(res$final_fit) && !is.null(res$final_fit$center_means)) {
    res$center_means <- res$final_fit$center_means
  }
  if (is.null(res$center_means_list) && !is.null(res$final_fit) && !is.null(res$final_fit$center_means_list)) {
    res$center_means_list <- res$final_fit$center_means_list
  }

  pred_names <- colnames(X_list[[1L]])
  if (!is.null(res$final_model)) {
    if (!is.null(pred_names) &&
        length(res$final_model) == (length(pred_names) + 1L) &&
        is.null(names(res$final_model))) {
      names(res$final_model) <- c("(Intercept)", pred_names)
    }
  } else if (!is.null(res$final_models) && length(res$final_models)) {
    for (m_i in seq_along(res$final_models)) {
      fm <- res$final_models[[m_i]]
      if (!is.null(pred_names) &&
          length(fm) == (length(pred_names) + 1L) &&
          is.null(names(fm))) {
        names(fm) <- c("(Intercept)", pred_names)
        res$final_models[[m_i]] <- fm
      }
    }
  }

  if (isTRUE(return_full_imputations)) {
    res$full_imputations <- list(X = X_list, y = y_list)
  }
  res$folds <- folds_id
  res
}












