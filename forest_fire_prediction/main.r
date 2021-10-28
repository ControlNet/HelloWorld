options(warn = -1, repr.plot.width=15, repr.plot.height=15)

library(GGally)
library(ggstatsplot)
library(reshape2)
library(caTools)
library(gridExtra)
library(class)
library(mltools)
library(data.table)
library(FNN)
library(tree)
library(MASS)
library(MLmetrics)
library(dplyr)
library(ggsci)
library(scales)
library(glmnet)
library(boot)

## set Material colors
get_material_color <- function(color) pal_material(color)(10)[4]
blue <- get_material_color("blue")
purple <- get_material_color("purple")
red <- get_material_color("red")
indigo <- get_material_color("indigo")
green <- get_material_color("green")
gray <- get_material_color("grey")
white <- "white"

### Load data
fire_data <- read.csv("forestfires.csv")
categorical_vars <- c("month", "day")
fire_data$month <- fire_data$month %>% as.factor
fire_data$day <- fire_data$day %>% as.factor
numerical_vars <- names(fire_data) %>% setdiff(categorical_vars) %>% setdiff("area")
str(fire_data)

summary(fire_data)

ggpairs(fire_data[numerical_vars])

ggcorrmat(fire_data, colors = c(blue, "white", green))

fire_data %>%
  select(-month, -day) %>%
  melt(id.vars = NULL) %>%
  ggplot +
  geom_boxplot(mapping = aes(x = "", y = value), fill = blue) +
  facet_wrap(~variable, scales = "free_y", ncol = 11) +
  ggtitle("The boxplot for each column")

fire_data %>% ggplot() +
  geom_point(aes(x = factor(1), y = area, fill = area), shape = 21, size = 3, color = "black", alpha=0.5, position = "jitter") +
  scale_fill_gradientn(colors = c("black", purple, red, red, red)) +
  xlab("area") + ylab("size") +
  ggtitle("The data points of the area")

fire_data %>%
  select(X, Y, area) %>%
  ggpairs(title = "The relationship between coordinates and area")

fire_data %>%
  select(X, Y, FFMC, DMC, DC, ISI, temp, RH, wind, area) %>%
  ggduo(c("X", "Y", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind"), "area",
        title = "The scatter with numerical features and area", mapping = NULL)

fire_data %>%
  mutate(logarea = log(area)) %>%
  select(X, Y, FFMC, DMC, DC, ISI, temp, RH, wind, logarea) %>%
  filter(logarea != -Inf) %>%
  ggduo(c("X", "Y", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind"), "logarea",
        title = "The scatter with coordinates and log(area)", mapping = NULL)

# transform the month as int
month_int <- sapply(fire_data$month, function(x) list("jan" = 1, "feb" = 2, "mar" = 3,
  "apr" = 4, "may" = 5, "jun" = 6, "jul" = 7, "aug" = 8, "sep" = 9, "oct" = 10,
  "nov" = 11, "dec" = 12)[[x]])
# plot
fire_data %>% select(month, area) %>% mutate(month_int = month_int) %>%
  group_by(month_int) %>%
  summarise(mean_area = mean(area)) %>% ggplot +
  geom_line(aes(x = month_int, y = mean_area), size = 1) +
  geom_point(aes(x = month_int, y = mean_area), size = 3, color = red) +
  xlab("month") + ylab("mean of the area") +
  scale_x_continuous(breaks = 1:12) +
  ggtitle("The trend of mean of area for months")

# transform the day of week as int
day_int <- sapply(fire_data$day, function(x) list("mon" = 1, "tue" = 2, "wed" = 3,
  "thu" = 4, "fri" = 5, "sat" = 6, "sun" = 7)[[x]])
# plot
fire_data %>% select(day, area) %>% mutate(day_int = day_int) %>% group_by(day_int) %>%
  summarise(mean_area = mean(area)) %>% ggplot +
  geom_line(aes(x = day_int, y = mean_area), size = 1) +
  geom_point(aes(x = day_int, y = mean_area), size = 3, color = red) +
  xlab("day") + ylab("mean of the area") +
  scale_x_continuous(breaks = 1:7) +
  ggtitle("The trend of mean of area for days")

fire_data %>% mutate(day_int = day_int) %>% mutate(month_int = month_int) %>%
  select(month_int, day_int, FFMC, DMC, DC, ISI, temp, RH, wind, area) %>%
  ggduo(columnsX = c("month_int", "day_int"),
        columnsY = c("FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "area"),
        title = "The scatter with numerical features and area", mapping = NULL)

fire_data %>% select(X, Y, area) %>% mutate(X = as.factor(X), Y = as.factor(Y)) %>%
  group_by(X, Y) %>% summarise(mean_area = mean(area)) %>% ggplot +
  geom_tile(aes(X, Y, fill = mean_area), color = gray) +
  scale_fill_gradientn(colors = c(white, purple, red))



# generate combination features
generate_comb_features <- function(data) {
  data_comb <- data %>% as.data.frame
  for (i in 1:(length(numerical_vars) - 1)) {
    for (j in (i+1):length(numerical_vars)) {
      numerical_var1 <- numerical_vars[i]
      numerical_var2 <- numerical_vars[j]
      new_name <- paste(numerical_var1, numerical_var2, sep = ":")
      data_comb[,new_name] <- data_comb[,numerical_var1] * data_comb[,numerical_var2]
    }
  }
  data.frame(data_comb)
}

# generate polynomial features
generate_polynomial_features <- function(data, power) {
  data_poly <- data %>% as.data.frame
  for (numerical_var in numerical_vars) {
    new_name <- paste0(numerical_var, ".", power)
    data_poly[,new_name] <- data_poly[,numerical_var] ^ power
  }
  data.frame(data_poly)
}

set.seed(42)
train_mask <- sample(1:nrow(fire_data_processed),
                     size = 0.8 * nrow(fire_data_processed), replace = FALSE)
train_data <- fire_data_processed[train_mask,]
test_data <- fire_data_processed[-train_mask,]
# apply the features generation
train_data_comb <- train_data %>% generate_comb_features %>% generate_polynomial_features(power = 3)
test_data_comb <- test_data %>% generate_comb_features %>% generate_polynomial_features(power = 3)
# plot chart for the amounts of train data and test data
data_sizes <- data.frame(group=c("train", "test"), value=c(nrow(train_data), nrow(test_data)))
g1 <- data_sizes %>% ggplot(aes(x="", y=value, fill=group)) + geom_bar(stat = "identity") +
  coord_polar("y") + ggtitle("Number of rows") + scale_fill_manual(values = c(red, blue))
g2 <- data_sizes %>% ggplot(aes(x=group, y=value, fill=group)) + geom_bar(stat = "identity") +
  scale_fill_manual(values = c(red, blue))
grid.arrange(g1, g2, nrow = 1)

fire_count <- fire_data_processed %>% count(fire)
fire_count %>% ggplot(aes(x = fire, y = n, fill = fire)) + geom_bar(stat = "identity") +
  scale_fill_manual(values = c(red, blue)) +
  ggtitle("Bar chart for fire") +
  labs(y = "numbers")

r2 <- function(y_test, y_pred)
        MLmetrics::R2_Score(y_true = y_test, y_pred = y_pred)
mse <- function(y_test, y_pred)
        MLmetrics::MSE(y_true = y_test %>% unlist, y_pred = y_pred)
rmse <- function(y_test, y_pred)
        MLmetrics::RMSE(y_true = y_test %>% unlist, y_pred = y_pred)
mae <- function(y_test, y_pred)
        MLmetrics::MAE(y_true = y_test %>% unlist, y_pred = y_pred)

accuracy <- function(y_test, y_pred)
        MLmetrics::Accuracy(y_true = y_test, y_pred = y_pred)
f1 <- function(y_test, y_pred)
        MLmetrics::F1_Score(y_true = y_test %>% unlist, y_pred = y_pred %>% unlist)
# confusion matrix is not return a scalar, only for show
confusion_matrix <- function(y_test, y_pred)
        table(y_test %>% unlist, y_pred)

AbstractModel <- setRefClass("AbstractModel",
  fields = list(train = "data.frame", need_comb_data = "logical",
                use_regularizer = "logical"),
  methods = list(
    initialize = function() {},
    fit = function(train, ...) {},
    predict = function(test, ...) {},
    get_default_formula = function(formula) {},
    # the argument evaluate_func can be r2, mse, accuracy, f1,
    # which will be defined in Model Comparison section
    evaluate = function(y_test, y_pred, evaluate_func) evaluate_func(y_test, y_pred),
    pred_evaluate = function(test, evaluate_func) {}
))

AbstractRegression <- setRefClass("AbstractRegression", contains = "AbstractModel",
  fields = list(is_logy = "logical", log_bias = "numeric"),
  methods = list(
    initialize = function(is_logy = FALSE, log_bias = 0, ...) {
      .self$is_logy <- is_logy
      .self$log_bias <- log_bias
    },
    logy_revert = function(pred) {
      if (.self$is_logy) pred %>% exp %>% (function(x) x - .self$log_bias)
      else pred
    },
    get_default_formula = function(formula) {
      if (is.null(formula)) {
        if (.self$is_logy) log(area + .self$log_bias) ~ . - fire
        else area ~ . - fire
      }
    },
    pred_evaluate = function(test, evaluate_func) {
      y_test <- test[,"area"]
      y_pred <- .self$predict(test)
      evaluate_func(y_test %>% unlist, y_pred)
    }
))

AbstractClassifier <- setRefClass("AbstractClassifier",
  contains = "AbstractModel", methods = list(
    get_default_formula = function(formula) {
      if (is.null(formula)) fire ~ . - area
      else formula
    },
    pred_evaluate = function(test, evaluate_func) {
      y_test <- test[,"fire"]
      y_pred <- .self$predict(test)
      evaluate_func(y_test %>% unlist, y_pred)
    }
))

grid_search_1d <- function(xs, best_func, pred_evaluate_func) {
  # calculate the result by given pred_evaluate function
  results <- sapply(xs, pred_evaluate_func)
  # get the best function, max for r2, min for mse
  best_result <- best_func(results)
  best_x <- xs[which(results == best_result)[1]]
  print(paste0("Best hyperparameter value is ", best_x))
  print(paste0("Best value is ", best_result))
  # return the geom_path object for visualization
  list(results =  data.frame(x = xs, result = results),
       x = best_x,
       best_result = best_result
  )
}

grid_search_2d <- function(xs, ys, best_func, pred_evaluate_func) {
  # initialize a matrix for storing results
  results <- matrix(nrow = length(xs) * length(ys), ncol = 3) %>%
          `colnames<-`(c("x", "y", "value"))
  # loop all posible hyperparameter to calculate each result
  index <- 1
  for (i in 1:length(xs)) {
    for (j in 1:length(ys)) {
      results[index, ] <- c(xs[i], ys[j], pred_evaluate_func(xs[i], ys[j]))
      index <- index + 1
    }
  }
  results <- as.data.frame(results)
  # locate best hyperparameter
  best_result <- best_func(results$value)
  max_index <- which(results$value == best_result)[1]
  print(paste0("Best x: ", results$x[max_index]))
  print(paste0("Best y: ", results$y[max_index]))
  max_x_index <- which(xs == results$x[max_index])
  max_y_index <- which(ys == results$y[max_index])
  # return the geom_tile and annotation objects for visualization
  list(tile = geom_tile(data = results,
                        mapping = aes(x = as.factor(x), y = as.factor(y), fill = value)),
    annotation = annotate("text", x = max_x_index, y = max_y_index, size = 3,
                          label = paste0("best=", round(best_result, 1))),
    x = results$x[max_index],
    y = results$y[max_index],
    best_result = best_result)
}

# a function for evaluate the model automatically
evaluate_model <- function(model, evaluate_func) {
  if (model$need_comb_data) model$pred_evaluate(test_data_comb, evaluate_func)
  else model$pred_evaluate(test_data, evaluate_func)
}

# search best regularizer for linear regression by cross validation
# calculate the mse without regularization as control group
cvlm <- cv.glm(rbind(train_data, test_data),
               glm(area ~ . - fire, data = rbind(train_data, test_data),
                   family = "gaussian"),
               K = 10)
original_cvlm_mse <- cvlm[["delta"]][1]
cvlm <- cv.glm(rbind(train_data_comb, test_data_comb),
               glm(area ~ . - fire, data = rbind(train_data_comb, test_data_comb),
                   family = "gaussian"),
               K = 10)
comb_cvlm_mse <- cvlm[["delta"]][1]

# set potential alpha values
alphas <- seq(0, 1, 0.02)

# the elastic net for original data (without combined and polynomial features)
original_glmnet <- alphas %>% lapply(FUN = function(alpha) {
  cv.glmnet(rbind(train_data_comb,test_data_comb) %>%
                    select(!c("area", "fire")) %>% data.matrix,
            rbind(train_data_comb,test_data_comb)[, "area"] %>% data.matrix,
            alpha = alpha, family = "gaussian", nfolds = 10)
})
# the elastic net for data with combined and polynomial features
comb_glmnet <- alphas %>% lapply(FUN = function(alpha) {
  cv.glmnet(rbind(train_data,test_data) %>%
                    select(!c("area", "fire")) %>% data.matrix,
            rbind(train_data,test_data)[, "area"] %>% data.matrix,
            alpha = alpha, family = "gaussian", nfolds = 10)
})
# calculate the mse
original_mse <- sapply(original_glmnet, function(x) x$cvm %>% min)
comb_mse <- sapply(comb_glmnet, function(x) x$cvm %>% min)

# combine the data for visualization
regularization_mse_data <- data.frame(alpha = alphas, original = original_mse,
                                      comb = comb_mse, regularizer = TRUE)
no_regularization_mse_data <- data.frame(alpha = alphas, original = original_cvlm_mse,
                                         comb = comb_cvlm_mse, regularizer = FALSE)
melted_regularization_mse <- rbind(regularization_mse_data, no_regularization_mse_data) %>%
  melt(id.vars = c("alpha", "regularizer"))
# plot
ggplot(melted_regularization_mse) +
  geom_line(aes(x = alpha, y = value, color = variable, linetype = regularizer)) +
  labs(color = "data", linetype = "regularization", y = "mse") +
  scale_linetype_manual(values=c(2,1)) +
  ggtitle("The MSE of different regularizations")

# choose best regularizer for both comb_data and original data
best_original_glmnet <- original_glmnet[[which.min(original_mse)]]
regularizer_for_original <- list(lambda = best_original_glmnet$lambda.min,
                                 alpha = alphas[which.min(original_mse)])

best_comb_glmnet <- comb_glmnet[[which.min(comb_mse)]]
regularizer_for_comb <- list(lambda = best_comb_glmnet$lambda.min,
                             alpha = alphas[which.min(comb_mse)])

# print the result
sprintf("The best regularizer for original_data is: Lambda = %.3f, Alpha = %.2f",
        regularizer_for_original$lambda, regularizer_for_original$alpha)
sprintf("The best regularizer for comb_data is: Lambda = %.3f, Alpha = %.2f",
        regularizer_for_comb$lambda, regularizer_for_comb$alpha)

# linear regression
LinearRegression <- setRefClass("LinearRegression", contains = "AbstractRegression",
                                fields = c("model", "x_train", "y_train"), methods = list(
  fit = function(train, comb_data, regularizer = NULL, formula = NULL) {
    # build and fit the model by given train data
    # record the input parameters
    .self$train <- train
    .self$need_comb_data <- comb_data
    formula <- .self$get_default_formula(formula)
    # build model
    .self$model <- {
      # if no regularizer, build a ordinary lm model
      if (is.null(regularizer)) lm(formula = formula, data = .self$train)
      else {
        # if there is a regularizer, build a glmnet model
        .self$use_regularizer <- TRUE
        .self$x_train <- model.matrix(formula, data = .self$train)
        .self$y_train <- .self$train[, "area"] %>% unlist
        glmnet(x = .self$x_train, y = .self$y_train, alpha = 0.5,
               lambda = regularizer$lambda)
      }
    }
    .self
  },
  predict = function(test) {
    # predict the value by given test data
    if (length(.self$use_regularizer) == 0)
      .self$model %>% predict.lm(test) %>% (.self$logy_revert)
    else
      .self$model %>% predict.glmnet(model.matrix(.self$get_default_formula(NULL),
                                                  data = test))
  },
  step = function(k = 2) {
    # this method is for stepwise feature selection
    .self$model <- stats::step(.self$model, trace = 0, k = k)
    .self
  }
))

# hyperparameter grid search
# for original data
lr_grid_search_result1 <- grid_search_1d(exp(seq(0, 5, 0.2)), min, function(x)
    LinearRegression(TRUE, x)$fit(train_data, FALSE)$pred_evaluate(test_data, mse))
# for comb_data
lr_grid_search_result2 <- grid_search_1d(exp(seq(0, 5, 0.2)), min, function(x)
    LinearRegression(TRUE, x)$fit(train_data_comb, TRUE)$pred_evaluate(test_data_comb, mse))
# for comb_data with feature selection
lr_grid_search_result3 <- grid_search_1d(exp(seq(0, 5, 0.2)), min, function(x)
    LinearRegression(TRUE, x)$fit(train_data_comb, TRUE)$step()$pred_evaluate(test_data_comb, mse))
# record the number of parameters
lr_grid_search_len <- length(lr_grid_search_result1$results$x)
# combine the results for visualization for logged area
lr_grid_search_results <- data.frame(
  x = lr_grid_search_result1$results$x,
  original = lr_grid_search_result1$results$result,
  comb = lr_grid_search_result2$results$result,
  step = lr_grid_search_result3$results$result,
  logged_y = rep(TRUE, lr_grid_search_len)
)
# calculate the results for area which is not logged as comparison
lr_original_mse_no_log <- LinearRegression(FALSE, 0)$fit(train_data, FALSE)$
        pred_evaluate(test_data, mse)
lr_comb_mse_no_log <- LinearRegression(FALSE, 0)$fit(train_data_comb, TRUE)$
        pred_evaluate(test_data_comb, mse)
lr_step_mse_no_log <- LinearRegression(FALSE, 0)$fit(train_data_comb, TRUE)$
        step()$pred_evaluate(test_data_comb, mse)

lr_grid_search_results_no_log <- data.frame(
  x = lr_grid_search_result1$results$x,
  original = rep(lr_original_mse_no_log, lr_grid_search_len),
  comb = rep(lr_comb_mse_no_log, lr_grid_search_len),
  step = rep(lr_step_mse_no_log, lr_grid_search_len),
  logged_y = rep(FALSE, lr_grid_search_len)
)
# melt the results together for visualization
lr_grid_search_results_melt <-
  rbind(lr_grid_search_results, lr_grid_search_results_no_log) %>%
  melt(id.vars = c("x", "logged_y"))

ggplot(lr_grid_search_results_melt) +
  geom_line(mapping = aes(x = x, y = value, color = variable, linetype = logged_y)) +
  labs(color = "data", linetype = "is y logged", x = "log bias", y = "mse") +
  scale_linetype_manual(values=c(2,1)) +
  scale_color_manual(values = c(blue, red, "black")) +
  ggtitle("The grid search for linear regression")

select_best_grid_result <- function(grid_search_results, best_func = min) {
  grid_search_results_values <- grid_search_results %>%
          sapply(function(x) x$best_result) %>% unlist
  which(grid_search_results_values == best_func(grid_search_results_values))
}

lr_grid_search_results <- list(lr_grid_search_result1,
                               lr_grid_search_result2,
                               lr_grid_search_result3)
lr_best_parameter_index <- select_best_grid_result(lr_grid_search_results, min)
linear_regression <- if (lr_best_parameter_index == 1) {
  LinearRegression(TRUE, lr_grid_search_result1$x)$fit(train_data, FALSE)
} else if (lr_best_parameter_index == 2) {
  LinearRegression(TRUE, lr_grid_search_result2$x)$fit(train_data_comb, TRUE)
} else if (lr_best_parameter_index == 3) {
  LinearRegression(TRUE, lr_grid_search_result3$x)$fit(train_data_comb, TRUE)$step()
}

linear_regression %>% evaluate_model(mse)

# KNN Regressor
KNNRegression <- setRefClass("KNNRegressor", contains = "AbstractRegression",
  fields = list(k = "numeric", x_train = "data.frame", y_train = "numeric"),
  methods = list(
    initialize = function(k, is_logy = FALSE, log_bias = 0) {
      .self$k <- k
      .self$is_logy <- is_logy
      .self$log_bias <- log_bias
    },
    fit = function(train, comb_data) {
      .self$need_comb_data <- comb_data
      .self$train <- train
      .self$x_train <- train %>% select(!c("area", "fire"))
      .self$y_train <- train[, "area"] %>% unlist
      if (.self$is_logy) {
        .self$y_train <- .self$y_train %>% (function(x) log(x + .self$log_bias))
      }
      .self
    },
    predict = function(test) {
      x_test <- test %>% select(!c("area", "fire"))
      knn.reg(train = .self$x_train, y = .self$y_train,
              test = x_test, k = .self$k)$pred %>% (.self$logy_revert)
    }
))

# grid search of KNN with original features
knn_reg_grid_search_result1 <- grid_search_2d(xs = seq(3, 91, 2), ys = exp(seq(0, 10, 0.2)),
                                              min, function(x, y) {
  KNNRegression(x, TRUE, y)$fit(train_data, FALSE)$pred_evaluate(test_data, mse)
})

ggplot() + knn_reg_grid_search_result1$tile + knn_reg_grid_search_result1$annotation +
  scale_fill_gradientn(colors = c(gray, blue, purple, red)) +
  labs(x = "K", y = "log bias", fill = "mse") +
  theme(axis.text = element_text(size = 6)) +
  ggtitle("Grid search for KNN regression on original data")

# grid search of KNN with combination features
knn_reg_grid_search_result2 <- grid_search_2d(xs = seq(3, 91, 2), ys = exp(seq(0, 10, 0.2)),
                                              min, function(x, y) {
  KNNRegression(x, TRUE, y)$fit(train_data_comb, TRUE)$pred_evaluate(test_data_comb, mse)
})

ggplot() + knn_reg_grid_search_result2$tile + knn_reg_grid_search_result2$annotation +
  scale_fill_gradientn(colors = c(gray, blue, purple, red)) +
  labs(x = "K", y = "log bias", fill = "mse") +
  theme(axis.text = element_text(size=6)) +
  ggtitle("Grid search for KNN regression on comb_data")

# get best knn regression
knn_reg_grid_search_results <- list(knn_reg_grid_search_result1,
                                    knn_reg_grid_search_result2)
knn_reg_best_parameter_index <- select_best_grid_result(knn_reg_grid_search_results, min)

knn_regression <- if (knn_reg_best_parameter_index == 1) {
  KNNRegression(knn_reg_grid_search_result1$x, TRUE, knn_reg_grid_search_result1$y)$
          fit(train_data, FALSE)
} else if (knn_reg_best_parameter_index == 2) {
  KNNRegression(knn_reg_grid_search_result2$x, TRUE, knn_reg_grid_search_result2$y)$
          fit(train_data_comb, TRUE)
}
knn_regression %>% evaluate_model(mse)

# Decision Tree Regression
DecisionTreeRegression <- setRefClass("DecisionTreeRegression",
  contains = "AbstractRegression",
  fields = c("model"),
  methods = list(
    fit = function(train, comb_data, formula = NULL) {
      .self$need_comb_data <- comb_data
      .self$train <- train
      formula <- .self$get_default_formula(formula)
      .self$model <- tree(formula = formula, data = .self$train)
      .self
    },
    predict = function(test) {
      .self$model %>% stats::predict(test) %>% (.self$logy_revert)
    }
))

# grid search
decision_tree_reg_grid_search_result1 <- grid_search_1d(exp(seq(-1, 5, 0.1)), min, function(x)
  DecisionTreeRegression(TRUE, x)$fit(train_data, FALSE)$pred_evaluate(test_data, mse))
decision_tree_reg_grid_search_result2 <- grid_search_1d(exp(seq(-1, 5, 0.1)), min, function(x)
  DecisionTreeRegression(TRUE, x)$fit(train_data_comb, TRUE)$pred_evaluate(test_data_comb, mse))

decision_tree_reg_grid_search_len <-
        length(decision_tree_reg_grid_search_result1$results$x)

decision_tree_reg_grid_search_results <- data.frame(
  x = decision_tree_reg_grid_search_result1$results$x,
  original = decision_tree_reg_grid_search_result1$results$result,
  comb = decision_tree_reg_grid_search_result2$results$result,
  logged_y = rep(TRUE, decision_tree_reg_grid_search_len)
)
# calculate the results for area which is not logged as comparison
decision_tree_regression_original_mse_no_log <- DecisionTreeRegression(FALSE, 0)$
        fit(train_data, FALSE)$pred_evaluate(test_data, mse)
decision_tree_regression_comb_mse_no_log <- DecisionTreeRegression(FALSE, 0)$
        fit(train_data_comb, TRUE)$pred_evaluate(test_data_comb, mse)

decision_tree_regression_grid_search_results_no_log <- data.frame(
  x = decision_tree_reg_grid_search_result1$results$x,
  original = rep(decision_tree_regression_original_mse_no_log,
                 decision_tree_reg_grid_search_len),
  comb = rep(decision_tree_regression_comb_mse_no_log,
             decision_tree_reg_grid_search_len),
  logged_y = rep(FALSE, decision_tree_reg_grid_search_len)
)

# melt the results together for visualization
decision_tree_regression_grid_search_results_melt <-
  rbind(decision_tree_reg_grid_search_results,
        decision_tree_regression_grid_search_results_no_log) %>%
  melt(id.vars = c("x", "logged_y"))

ggplot(decision_tree_regression_grid_search_results_melt) +
  geom_line(mapping = aes(x = x, y = value, color = variable, linetype = logged_y)) +
  labs(color = "data", linetype = "is y logged", x = "log bias", y = "mse") +
  scale_linetype_manual(values = c(2,1)) +
  scale_color_manual(values = c(blue, red)) +
  ggtitle("Grid search for decision tree regression")

# Build best decision tree
decision_tree_reg_grid_search_results <- list(decision_tree_reg_grid_search_result1,
                                              decision_tree_reg_grid_search_result2)
decision_tree_reg_best_parameter_index <-
        select_best_grid_result(decision_tree_reg_grid_search_results, min)
decision_tree_regression <- if (decision_tree_reg_best_parameter_index == 1) {
  DecisionTreeRegression(TRUE, decision_tree_reg_grid_search_result1$x)$
          fit(train_data, FALSE)
} else if (decision_tree_reg_best_parameter_index == 2) {
  DecisionTreeRegression(TRUE, decision_tree_reg_grid_search_result2$x)$
          fit(train_data_comb, TRUE)
}
decision_tree_regression %>% evaluate_model(mse)

# Logistic Regression
LogisticRegression <- setRefClass("LogisticRegression",
  contains = "AbstractClassifier",
  fields = list(model = "glm", threshold = "numeric"),
  methods = list(
    initialize = function(threshold = 0.5) {
      .self$threshold <- threshold
    },
    fit = function(train, comb_data, formula = NULL) {
      .self$need_comb_data <- comb_data
      .self$train <- train
      formula <- .self$get_default_formula(formula)
      .self$model <- glm(formula = formula, family = binomial, data = .self$train)
      .self
    },
    predict = function(test) {
      prob <- .self$model %>% predict.glm(newdata = test, type = "response")
      prob > .self$threshold
    },
    set_threshold = function(threshold) {
      .self$threshold <- threshold
      .self
    },
    step = function(k = 2) {
      .self$model <- stepAIC(.self$model, k = k, direction = "backward", trace = 0)
      .self
    }
))

# Logistic regression with original dataset, combination feature dataset,
# and the dataset with stepwise sub-selection
logistic_regression1 <- LogisticRegression()$fit(train_data, FALSE)
logistic_regression2 <- LogisticRegression()$fit(train_data_comb, TRUE)
logistic_regression3 <- LogisticRegression()$fit(train_data_comb, TRUE)$step()

logistic_reg_grid_search_result1 <- grid_search_1d(seq(0, 1, 0.02), max, function(x)
    logistic_regression1$set_threshold(x)$pred_evaluate(test_data, accuracy))
logistic_reg_grid_search_result2 <- grid_search_1d(seq(0, 1, 0.02), max, function(x)
    logistic_regression2$set_threshold(x)$pred_evaluate(test_data_comb, accuracy))
logistic_reg_grid_search_result3 <- grid_search_1d(seq(0, 1, 0.02), max, function(x)
    logistic_regression3$set_threshold(x)$pred_evaluate(test_data_comb, accuracy))

logistic_reg_grid_search_len <- length(seq(0, 1, 0.02))

logistic_reg_grid_search_results_melt <- data.frame(
  x = logistic_reg_grid_search_result1$results$x,
  original = logistic_reg_grid_search_result1$results$result,
  comb = logistic_reg_grid_search_result2$results$result,
  step = logistic_reg_grid_search_result3$results$result
) %>% melt(id.vars = "x")

ggplot(logistic_reg_grid_search_results_melt) +
  geom_line(aes(x = x, y = value, color = variable)) +
  scale_color_manual(values = c(blue, red, "black")) +
  labs(color = "data", x = "threshold", y = "accuracy") +
  ggtitle("Grid search for logistic regression")

# select the best logistic regression
logistic_reg_grid_search_results <- list(logistic_reg_grid_search_result1,
                                         logistic_reg_grid_search_result2,
                                         logistic_reg_grid_search_result3)
logistic_reg_reg_best_parameter_index <- select_best_grid_result(logistic_reg_grid_search_results, max)

logistic_regression <- if (logistic_reg_reg_best_parameter_index == 1) {
  logistic_regression1$set_threshold(logistic_reg_grid_search_result1$x)
} else if (logistic_reg_reg_best_parameter_index == 2) {
  logistic_regression2$set_threshold(logistic_reg_grid_search_result2$x)
} else if (logistic_reg_reg_best_parameter_index == 3) {
  logistic_regression3$set_threshold(logistic_reg_grid_search_result3$x)
}

logistic_regression %>% evaluate_model(accuracy)

# KNN Classifier
KNNClassifier <- setRefClass("KNNClassifier", contains = "AbstractClassifier",
                             fields = list(k = "numeric", x_train = "data.frame", y_train = "logical"),
                             methods = list(
  initialize = function(k) {
    .self$k <- k
  },
  fit = function(train, comb_data) {
    .self$need_comb_data <- comb_data
    .self$x_train <- train %>% select(!c("area", "fire"))
    .self$y_train <- train[, "fire"] %>% unlist
    .self
  },
  predict = function(test) {
    x_test <- test %>% select(!c("area", "fire"))
    FNN::knn(train = .self$x_train, cl = .self$y_train, test = x_test, k = .self$k)
  }
))

# KNN classifier with original dataset and combination feature dataset
knn_cla_grid_search_result1 <- grid_search_1d(seq(1, 100, 1), max, function(x)
    KNNClassifier(x)$fit(train_data, FALSE)$pred_evaluate(test_data, accuracy))

knn_cla_grid_search_result2 <- grid_search_1d(seq(1, 100, 1), max, function(x)
    KNNClassifier(x)$fit(train_data_comb, TRUE)$pred_evaluate(test_data_comb, accuracy))

knn_cla_grid_search_len <- length(seq(1, 100, 1))

knn_cla_grid_search_results_melt <- data.frame(
  x = knn_cla_grid_search_result1$results$x,
  original = knn_cla_grid_search_result1$results$result,
  comb = knn_cla_grid_search_result2$results$result
) %>% melt(id.vars = "x")

ggplot(knn_cla_grid_search_results_melt) +
  geom_line(aes(x = x, y = value, color = variable)) +
  scale_color_manual(values = c(blue, red)) +
  labs(color = "data", x = "K", y = "accuracy") +
  ggtitle("Grid search for KNN classifier")

# grid search of KNN with original features
knn_cla_grid_search_results <- list(knn_cla_grid_search_result1, knn_cla_grid_search_result2)
knn_cla_best_parameter_index <- select_best_grid_result(knn_cla_grid_search_results, max)

knn_classifier <- if (knn_cla_best_parameter_index == 1) {
  KNNClassifier(knn_cla_grid_search_result1$x)$fit(train_data, FALSE)
} else if (knn_cla_best_parameter_index == 2) {
  KNNClassifier(knn_cla_grid_search_result2$x)$fit(train_data_comb, TRUE)
}

knn_classifier %>% evaluate_model(accuracy)

# Decision tree classifier
DecisionTreeClassifier <- setRefClass("DecisionTreeClassifier", contains = "AbstractClassifier",
                                      fields = c("model"), methods = list(
  fit = function(train, comb_data, formula = NULL) {
    .self$need_comb_data <- comb_data
    formula <- .self$get_default_formula(formula)
    .self$model <- tree(formula = formula, data = train)
    .self
  },
  predict = function(test) {
    prob <- .self$model %>% stats::predict(newdata = test)
    prob > 0.5
  }
))

# Decision tree classifier with original dataset and combination feature dataset
decision_tree_cla_results <- c(
  DecisionTreeClassifier()$fit(train_data, FALSE)$pred_evaluate(test_data, accuracy),
  DecisionTreeClassifier()$fit(train_data_comb, TRUE)$pred_evaluate(test_data_comb, accuracy)
)
decision_tree_cla_results

decision_tree_cla_best_parameter_index <-
        which(decision_tree_cla_results == max(decision_tree_cla_results))
decision_tree_classifier <- if (decision_tree_cla_best_parameter_index == 1) {
  DecisionTreeClassifier()$fit(train_data, FALSE)
} else if (decision_tree_cla_best_parameter_index == 2) {
  DecisionTreeClassifier()$fit(train_data_comb, TRUE)
}
decision_tree_classifier %>% evaluate_model(accuracy)

CompoundModel <- setRefClass("CompoundModel", contains = "AbstractRegression",
  fields = list(classifier = "AbstractClassifier", regressor = "AbstractRegression"),
  methods = list(
    initialize = function(classifier, regressor) {
      .self$classifier <- classifier
      .self$regressor <- regressor
    },

    predict = function(test, test_comb) {
      # predict the data is fire or not fire
      fire_pred <- {
        if (.self$classifier$need_comb_data) .self$classifier$predict(test_comb)
        else .self$classifier$predict(test)
      }
      # the classifier returns TRUE or FALSE, then set FALSE as 0
      predict_by_regressor <- function(row) {
        # if it fires (area > 0), use regressor to predict the area
        if (fire_pred[row] == as.factor(TRUE)) {
          if (.self$regressor$need_comb_data) .self$regressor$predict(test_comb[row,])
          else .self$regressor$predict(test[row,])
        }
        # else predict the fire area as 0
        else 0
      }
      # use relu to set all negative prediction as 0
      relu <- function(pred) max(0, pred)
      # predict
      y_pred <- 1:nrow(test) %>% sapply(function(row) row %>% predict_by_regressor %>% relu)
      y_pred
    },

    pred_evaluate = function(test, test_comb, evaluate_func) {
      y_test <- test[,"area"]
      y_pred <- .self$predict(test, test_comb)
      evaluate_func(y_test %>% unlist, y_pred)
    }
))

# prepare the list for grid search
regressors <- list(linear_regression, knn_regression, decision_tree_regression)
classifiers <- list(logistic_regression, knn_classifier, decision_tree_classifier)
# grid search of compound model with different regressors and classifiers
grid_search_for_compound_model <- function(regressors_in, classifiers_in,
                                           best_func = min, evaluate_func = mse) {
  results <- data.frame(regressors = character(), classifiers = character(), value = double())
  for (i in 1:length(regressors_in)) {
    reg <- regressors_in[[i]]
    for (j in 1:length(classifiers_in)) {
      cla <- classifiers_in[[j]]
      new_row <- list(regressors = i, classifiers = j,
                      value = CompoundModel(cla, reg)$
                                pred_evaluate(test_data, test_data_comb, evaluate_func))
      results <- rbind(results, new_row)
    }
  }
  best_result <- best_func(results$value)
  best_index <- which(results$value == best_result)[1]
  best_x_index <- results$regressors[best_index]
  best_y_index <- results$classifiers[best_index]
  results <- results %>%
    mutate(reg_name = sapply(regressors, function(x) class(regressors_in[[x]]))) %>%
    mutate(cla_name = sapply(classifiers, function(x) class(classifiers_in[[x]])))
  print(paste0("Best regressor: ", regressors_in[[best_x_index]] %>% class))
  print(paste0("Best classifier: ", classifiers_in[[best_y_index]] %>% class))

  list(tile = geom_tile(data = results, mapping = aes(x = regressors, y = classifiers,
                                                      fill = value)),
    annotation = annotate("text", x = best_x_index, y = best_y_index,
                          label = paste0("best=", best_result)),
    text = geom_text(data = results, aes(x = regressors, y = classifiers,
                                         label = round(value, 5))),
    regressors = regressors_in[best_x_index],
    classifiers = classifiers_in[best_y_index],
    best_result = best_result,
    results = results)
}

results <- grid_search_for_compound_model(regressors, classifiers)

ggplot() + results$tile + results$text +
  scale_x_discrete(limits = sapply(regressors, class)) +
  scale_y_discrete(limits = sapply(classifiers, class)) +
  scale_fill_gradientn(colors = c(blue, gray)) +
  labs(fill = "mse") +
  ggtitle("MSE value for compound models")

# choose best compound model
compound_model <- CompoundModel(logistic_regression, knn_regression)
compound_model$pred_evaluate(test_data, test_data_comb, evaluate_func = mse)

regressors_errors <- c(regressors %>% sapply(evaluate_model, evaluate_func = mse),
                       compound_model$pred_evaluate(test_data, test_data_comb,
                                                    evaluate_func = mse))

ggplot(data = data.frame(reg_name = c(sapply(regressors, class), "CompoundModel"),
                         error = regressors_errors)) +
  geom_bar(mapping = aes(x = reg_name, y = error, fill = error),
           stat = "identity", color = "black") +
  scale_fill_gradientn(colors = c(blue, gray)) +
  labs(x = "models", y = "mse", fill = "mse") +
  ggtitle("The MSE values of potential models")

# choose top 3 models as required
models <- c(linear_regression, knn_regression, compound_model)
model_names <- models %>% sapply(class)
model_rmse <- c(sapply(models[-3], evaluate_model, evaluate_func = rmse),
               compound_model$pred_evaluate(test_data, test_data_comb, rmse))
ggplot(mapping = aes(x = model_names, y = model_rmse, fill = model_rmse)) +
  geom_bar(stat = "identity", color = "black") +
  scale_fill_gradientn(colors = c(blue, gray)) +
  labs(x = "models", y = "rmse", fill = "rmse") +
  ggtitle("The RMSE of selected models")

model_mae <- c(sapply(models[-3], evaluate_model, evaluate_func = mae),
               compound_model$pred_evaluate(test_data, test_data_comb, mae))
ggplot(mapping = aes(x = model_names, y = model_mae, fill = model_mae)) +
  geom_bar(stat = "identity", color = "black") +
  scale_fill_gradientn(colors = c(blue, gray)) +
  labs(x = "models", y = "mae", fill = "mae") +
  ggtitle("The MAE of selected models")

model_r2 <- c(sapply(models[-3], evaluate_model, evaluate_func = r2),
               compound_model$pred_evaluate(test_data, test_data_comb, r2))
ggplot(mapping = aes(x = model_names, y = model_r2, fill = model_r2)) +
  geom_bar(stat = "identity", color = "black") +
  scale_fill_gradientn(colors = c(gray, blue)) +
  labs(x = "models", y = "r2", fill = "r2") +
  ggtitle("The R-Squared of selected models")

ggplot(data.frame(name = model_names, rmse = model_rmse, mae = model_mae)) +
  geom_point(aes(x = mae, y = rmse, color = name), size = 3) +
  labs(color = "model") +
  coord_fixed() +
  ggtitle("The MAE and RMSE scores of models")

model_summary <- compound_model$classifier$model %>% summary
model_summary

important_features <- model_summary$coefficients[
        model_summary$coefficients[,"Pr(>|z|)"] < 0.01,]
important_features

# positive correlation
important_features[important_features[,"Estimate"] > 0, ]

# negative correlation
important_features[important_features[,"Estimate"] < 0, ]
