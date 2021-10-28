library(dplyr)
library(ggplot2)
library(reshape2)

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

# read the train and test data
train_data <- read_data("train1.csv", FALSE)
x_train <- train_data$x
y_train <- train_data$y
test_data <- read_data("test1.csv", FALSE)
x_test <- test_data$x
y_test <- test_data$y

KNNRegressor <- setRefClass("KNNRegressor",
                            fields = list(k = "numeric", x_train = "data.frame", y_train = "data.frame"),
                            methods = list(
                                
                              initialize = function(k) {
                                .self$k <- k
                              },

                              fit = function(x_train, y_train) {
                                .self$x_train <- x_train
                                .self$y_train <- y_train
                                .self
                              },

                              predict = function(x_test) {
                                # define a inner function for applying each row
                                predict_for_row <- function(x_test_row) {
                                  # calculate the distance for each test data
                                  distance <- .self$x_train %>% 
                                    apply(1, function(x_train_row) {
                                    # Manhattan distance function
                                    x_train_row["dist"] <- sum(abs(x_train_row - x_test_row))
                                    x_train_row
                                    }) %>% 
                                    t %>% 
                                    as.data.frame %>% .["dist"]
                                    
                                  # find the K nearest neighbours' labels
                                  nearest_indexes <- order(distance$dist)[1:k]
                                  train_labels <- .self$y_train[nearest_indexes,]
                                  # predict the test labels with the mean of nearest neighbours
                                  y_pred_row <- mean(train_labels)
                                  y_pred_row
                                }
                                
                                y_pred <- apply(x_test, 1, predict_for_row)
                                y_pred
                              }
                            )
)

knn <- function(train.data, train.label, test.data, K=3) {
    knn_regressor <- KNNRegressor(K)$fit(as.data.frame(train.data), as.data.frame(train.label))
    knn_regressor$predict(as.data.frame(test.data))
}

# Mean of squared error
error_func <- function(test.pred, test.label) {
    mean((test.pred - test.label) ^ 2)
}
# calculate the errors for given K and data
cal_error_for_k <- function(k, x, y) {
    y_pred <- knn(x_train, y_train, x, k)
    error_func(y_pred, y)
}

# define the partial functions for calculating the errors for train and test data
cal_train_error <- function(k) cal_error_for_k(k, x=x_train, y=y_train)
cal_test_error <- function(k) cal_error_for_k(k, x=x_test, y=y_test)
k <- 1:30

# apply the function to get the errors
train_error <- sapply(k, cal_train_error)
test_error <- sapply(k, cal_test_error)
error <- data.frame(k=1:30, train = train_error, test = test_error)
# melt for visualization
error_melt <- melt(error, id="k")

# plot
ggplot(data=error_melt %>% mutate(x = 1/k), aes(x=x, y=value, color=variable)) +
    geom_line() + ggtitle('Test and Train Error') + labs(x="1/K", y="Error")

which.min(error$test)
