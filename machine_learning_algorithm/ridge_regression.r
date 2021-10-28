options(warn=-1)
library(dplyr)
library(ggplot2)
library(reshape2)

set.seed(5201)

# reading the data
read_data <- function(fname, sc) {
  data <- read.csv(file = fname, head = TRUE, sep = ",")
  nr = dim(data)[1]
  nc = dim(data)[2]
  x = data[1:nr, 1:(nc - 1)]
  y = data[1:nr, nc]
  if (isTRUE(sc)) {
    x = scale(x)
    y = scale(y)
  }
  return(list("x" = x, "y" = y))
}

# auxiliary function to calculate labels based on the estimated coefficients
predict_func <- function(Phi, w) {  
  return(Phi %*% w)
}

# auxiliary function to calculate the objective function for the training
train_obj_func <- function(Phi, w, label, lambda) {
  # the L2 regulariser is already included in the objective function for training
  return(mean((predict_func(Phi, w) - label)^2) + .5 * lambda * w %*% w)
}

# auxiliary function to compute the error of the model
get_errors <- function(train_data, test_data, W) {
  n_weights = dim(W)[1]
  errors = matrix(, nrow = n_weights, ncol = 2)
  for (tau in 1:n_weights) {
    errors[tau, 1] = train_obj_func(train_data$x, W[tau,], train_data$y, 0)
    errors[tau, 2] = train_obj_func(test_data$x, W[tau,], test_data$y, 0)
  }
  return(errors)
}

##--- Stochastic Gradient Descent --------------------------------------------
sgd_train <- function(train_x, train_y, lambda, eta, epsilon, max_epoch) {

  train_len = dim(train_x)[1]
  tau_max = max_epoch * train_len

  W <- matrix(, nrow = tau_max, ncol = ncol(train_x))
  W[1,] <- runif(ncol(train_x))

  tau = 1 # counter
  obj_func_val <- matrix(, nrow = tau_max, ncol = 1)
  obj_func_val[tau, 1] = train_obj_func(train_x, W[tau,], train_y, lambda)

  while (tau <= tau_max) {

    # check termination criteria
    if (obj_func_val[tau, 1] <= epsilon) { break }

    # shuffle data:
    train_index <- sample(1:train_len, train_len, replace = FALSE)

    # loop over each datapoint
    for (i in train_index) {
      # increment the counter
      tau <- tau + 1
      if (tau > tau_max) { break }

      # make the weight update
      y_pred <- predict_func(train_x[i,], W[tau - 1,])
      W[tau,] <- sgd_update_weight(W[tau - 1,], train_x[i,], train_y[i], y_pred, lambda, eta)

      # keep track of the objective funtion
      obj_func_val[tau, 1] = train_obj_func(train_x, W[tau,], train_y, lambda) 
    }
  }
  # resulting values for the training objective function as well as the weights
  return(list('vals' = obj_func_val, 'W' = W))
}

train_data <- read_data("train3.csv", TRUE)
x_train <- train_data$x
y_train <- train_data$y

test_data <- read_data("test3.csv", TRUE)
x_test <- test_data$x
y_test <- test_data$y

# updating the weight vector
sgd_update_weight <- function(W_prev, x, y_true, y_pred, lambda, eta) {
  residual <- y_true - y_pred
  grad = -residual %*% x + lambda * W_prev
  return(W_prev - eta * grad)
}

max_epoch <- 20
lambdas <- seq(0, 10, 0.4)
epsilon <- 0  # error tolerance for termination
eta <- 0.01  # learning rate

cal_errors_from_lambda <- function(lambda) {
  train_res <- sgd_train(x_train, y_train, lambda, eta, epsilon, max_epoch)
  last_weight <- train_res$W[nrow(train_res$W),] %>% as.matrix %>% t
  errors <- get_errors(train_data, test_data, last_weight) 
  # errors contains the train error and test error  
  errors  
}

errors <- lapply(lambdas, cal_errors_from_lambda) %>% 
            Reduce(f = rbind) %>% 
            cbind(lambdas, .) %>% 
            `colnames<-`(c("lambda", "train", "test")) %>% as.data.frame
errors %>% head

errors_melt <- errors %>% melt(id.vars = c("lambda")) %>% mutate(x = log(lambda))

ggplot(data=errors_melt) + 
  geom_line(aes(x=x, y=value, color=variable)) +
  xlab("log(lambda)") +
  ylab("error") + 
  ggtitle("The error rate trend for log(lambda)")

lambdas[which.min(errors$test)]
