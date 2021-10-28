options(warn=-1)
library(dplyr)
library(reshape)
library(ggplot2)
library(h2o)

localH2O =  h2o.init(nthreads = -1, port = 54321, max_mem_size = '16G', startH2O = TRUE)

error.rate <- function(Y1, T1){
  if (nrow(Y1)!=nrow(T1)){
    stop('error.rate: size of true lables and predicted labels mismatch')
  }
  return (sum(T1!=Y1)/nrow(T1))
}

labeled.frame <- h2o.importFile(path = 'train8_labeled.csv' ,sep=',') 
unlabeled.frame <- h2o.importFile(path = 'train8_unlabeled.csv' ,sep=',') 
test.frame <- h2o.importFile(path = 'test8.csv' ,sep=',') 

labeled.frame[,1] <- as.factor(labeled.frame$label)
unlabeled.frame[,1] <- NA
train.frame <- h2o.rbind(labeled.frame[,-1], unlabeled.frame[,-1])
test.frame[,1] <- as.factor(test.frame$label)

# set neurons
neurons <- seq(20, 400, 20)

# a function to build a autoencoder based on a given neuron number
build.ae <- function(neuron) {
    h2o.deeplearning(
        x = 1:ncol(train.frame),
        training_frame = train.frame,
        hidden = c(neuron),
        epochs = 15,
        activation = "Tanh",
        autoencoder = TRUE, 
        seed = 5201, 
        rate_decay = 0.99
    )
}

# build models and calculate reconstruction errors
ae.models <- neurons %>% lapply(build.ae)
ae.errors <- ae.models %>% sapply(h2o.mse)

ggplot(data.frame(neuron = neurons, error = ae.errors)) + 
    geom_line(aes(x = neuron, y = error)) + 
    labs(x = "Neuron", y = "Reconstruction Error") + 
    ggtitle("The Reconstruction Error of Autoencoders")

# build nn for original features
build.nn <- function(neuron) {
    h2o.deeplearning(
        x = 2:ncol(labeled.frame),
        y = 1,
        training_frame = labeled.frame,
        hidden = c(neuron),
        epochs = 5,
        rate = 0.001,
        activation = "Tanh",
        autoencoder = FALSE, 
        seed = 42, 
        rate_decay = 0.95,
        l1 = 0.01,
        l2 = 0.1
    )
}

# record the models
nn.models <- neurons %>% lapply(build.nn)
# calculate test errors
nn.errors <- nn.models %>% sapply(function(nn) {
    1 - mean(predict(nn, test.frame)$predict == test.frame$label)
})

# concat the hidden features to original features

concat.hidden.features <- function(frame) {
    1:length(neurons) %>% lapply(function(i) {
        feature <- h2o.deepfeatures(ae.models[[i]], frame, layer=1)
        h2o.cbind(frame, feature)
    })
}

labeled.augmented.frames <- concat.hidden.features(labeled.frame)
test.augmented.frames <- concat.hidden.features(test.frame)

# build augmented self-taught networks
build.augmented.nn <- function(index) {
    # choose the augmented data
    labeled.augmented.frame <- labeled.augmented.frames[[index]]
    neuron <- neurons[index]
    h2o.deeplearning(
        x = 2:ncol(labeled.augmented.frame),
        y = 1,
        training_frame = labeled.augmented.frame,
        hidden = c(neuron),
        epochs = 5,
        rate = 0.001,
        activation = "Tanh",
        autoencoder = FALSE, 
        seed = 42, 
        rate_decay = 0.95,
        l1 = 0.01,
        l2 = 0.1
    )
}

# record the models
augmented.nn.models <- 1:length(neurons) %>% lapply(build.augmented.nn)
# calculate test errors
augmented.nn.errors <- 1:length(neurons) %>% sapply(function(index) {
    nn = augmented.nn.models[[index]]
    test.augmented.frame = test.augmented.frames[[index]]
    1 - mean(predict(nn, test.augmented.frame)$predict == test.frame$label)
})

data.frame(neuron = neurons, original = nn.errors, augmented = augmented.nn.errors) %>% 
    melt(id.vars = "neuron") %>%
    ggplot + 
    geom_line(aes(x = neuron, y = value, color = variable)) + 
    labs(x = "Neuron", y = "Test Error", color = "Neural Network") + 
    ggtitle("The Test Error of Neural Networks")
