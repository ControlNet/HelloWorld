options(warn=-1)
library(dplyr)
library(reshape2)
library(ggplot2)

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
                                # define a inner function `majority`  
                                majority <- function(x) mean(x)
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
                                  y_pred_row <- majority(train_labels)
                                  y_pred_row
                                }
                                
                                y_pred <- apply(x_test, 1, predict_for_row)
                                y_pred
                              }
                            )
)

knn <- function(train.data, train.label, test.data, K) {
  knn_regressor <- KNNRegressor(K)$fit(as.data.frame(train.data), as.data.frame(train.label))
  knn_regressor$predict(as.data.frame(test.data))
}

error_func <- function(test.pred, test.label) {
  mean((test.pred - test.label) ^ 2)
}

train_data <- read.csv("train2.csv")
test_data <- read_data("test2.csv", FALSE)
x_test <- test_data$x
y_test <- test_data$y
head(train_data)

Bootstrap <- setRefClass("Bootstrap",
                         fields = list(
                           original_dataset = "data.frame",
                           original_size = "numeric",
                           sample_size = "numeric"),
                         methods = list(
                           initialize = function(original_dataset, sample_size) {
                             .self$original_dataset <- original_dataset
                             # get the original size by getting the row of original dataset
                             .self$original_size <- nrow(original_dataset)
                             .self$sample_size <- sample_size
                           },

                           sample = function(times) {
                             # for each time, generate a bootstrapping indexes and concat by rows
                             indexes <- Reduce(rbind, lapply(1:times, function(t) base::sample(x = .self$original_size,
                                                       size = .self$sample_size, replace = TRUE)))
                             # from indexes get data from original dataset
                             result <- apply(indexes, 1, function(indexes) .self$original_dataset[indexes,])
                             result
                           },

                           sample_once = function() {
                             .self$sample(1)[[1]]
                           }
                         )
)

boot <- function(original_dataset, sample_size, times) {
    Bootstrap(original_dataset, sample_size)$sample(times)
}

# time consumption: about 5 minutes
# get the boot data with specified size and times
boot_data <- boot(train_data, sample_size = 20, times = 50)

# define a function for calculate errors from data and K
cal_error <- function(data, k) {
  pred <- knn(train.data = data[,-5], train.label = data[,5], test.data = x_test, K = k)
  error_func(pred, y_test)  
}

# calculate the errors when K is in 1:15
errors <- Reduce(rbind, lapply(1:15, function(i) {
  sapply(boot_data, cal_error, k=i)
}))

# reformat the names of columns and rows
rownames(errors) <- 1:15
colnames(errors) <- NULL
errors

# melt the errors for plot
errors_melt <- errors %>% t %>% melt

# plot
ggplot() + geom_boxplot(data = errors_melt, mapping = aes(x=factor(Var2), y=value), outlier.shape = NA) + 
  labs(x = "K", y = "Test Error") + ggtitle('Test Error vs. K (Box Plot)')

options(warn=-1)

# time consumption: about 5 minutes
# define the sizes
sizes <- seq(5, 75, 5)

# calculate the errors from different sizes
errors_with_sizes <- Reduce(rbind, lapply(sizes, function(size) {
  # for each size, calculate the errors
  boot_data <- boot(train_data, sample_size = size, times = 50)  
  sapply(boot_data, function(data) cal_error(data, k=5))
}))

# reformat the names of errors matrix
rownames(errors_with_sizes) <- sizes
colnames(errors_with_sizes) <- NULL    
errors_with_sizes

# melt for visualization
errors_with_sizes_melt <- errors_with_sizes %>% as.matrix %>% melt

# plot
ggplot() + geom_boxplot(data = errors_with_sizes_melt, mapping = aes(x=factor(Var1), y=value), outlier.shape = NA) +
  labs(x = "Size", y = "Test Error") + ggtitle('Test Error vs. Size (Box Plot)')

options(warn=-1)
