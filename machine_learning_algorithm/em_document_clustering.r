library(dplyr)
library(tm)

eps=1e-10
  
# reading the data
read.data <- function(file.name, sample.size=1000, seed=100, pre.proc=TRUE, spr.ratio= 0.90) {
  
  # Read the data
  text <- readLines(file.name)
  # select a subset of data if sample.size > 0
  if (sample.size>0){
    set.seed(seed)
    text <- text[sample(length(text), sample.size)]
  }
  ## the terms before the first '\t' are the lables (the newsgroup names) and all the remaining text after '\t' are the actual documents
  docs <- strsplit(text, '\t')
  # store the labels for evaluation
  labels <- unlist(lapply(docs, function(x) x[1]))
  # store the unlabeled texts    
  # docs <- data.frame(unlist(lapply(docs, function(x) x[2])))
  uid <- paste0("doc_", formatC(1:length(text), width = 4, format = "d", flag = "0"))
  docs <- data.frame(doc_id = uid, text = unlist(lapply(docs, function(x) x[2])))
  

  # create a corpus
  docs <- DataframeSource(docs)
  corp <- Corpus(docs)
  
  # Preprocessing:
  if (pre.proc){
    corp <- tm_map(corp, removeWords, stopwords("english")) # remove stop words (the most common word in a language that can be find in any document)
    corp <- tm_map(corp, removePunctuation) # remove pnctuation
    corp <- tm_map(corp, stemDocument) # perform stemming (reducing inflected and derived words to their root form)
    corp <- tm_map(corp, removeNumbers) # remove all numbers
    corp <- tm_map(corp, stripWhitespace) # remove redundant spaces 
  }  
  # Create a matrix which its rows are the documents and colomns are the words. 
  dtm <- DocumentTermMatrix(corp)
  ## reduce the sparcity of out dtm
  dtm <- removeSparseTerms(dtm, spr.ratio)
  ## convert dtm to a matrix
  word.doc.mat <- as.matrix(dtm)
  
  # Return the result
  return (list("docs" = docs, "word.doc.mat"= word.doc.mat, "labels" = labels))
}

logSum <- function(v) {
   m = max(v)
   return (m + log(sum(exp(v-m))))
}

initial.param <- function(vocab_size, K=4, seed=123456){
  rho <- matrix(1/K,nrow = K, ncol=1)                    # assume all clusters have the same size (we will update this later on)
  mu <- matrix(runif(K*vocab_size),nrow = K, ncol = vocab_size)    # initiate Mu 
  mu <- prop.table(mu, margin = 2)               # normalization to ensure that sum of each row is 1
  return (list("rho" = rho, "mu"= mu))
}
                                                        
train_obj <- function(model, counts) { 
  N <- dim(counts)[1] # number of documents
  K <- dim(model$mu)[1]
   
  nloglike = 0
  for (n in 1:N){
    lprob <- matrix(0,ncol = 1, nrow=K)
    for (k in 1:K){
      lprob[k,1] = sum(counts[n,] * log(model$mu[k,] + eps)) 
    }
    nloglike <- nloglike - logSum(lprob + log(model$rho))
  }
  
  return (nloglike)
}

cluster.viz <- function(doc.word.mat, color.vector, title=' '){
  p.comp <- prcomp(doc.word.mat, scale. = TRUE, center = TRUE)
  plot(p.comp$x, col=color.vector, pch=1,  main=title)
}

E.step <- function(gamma, model, counts, mode = "soft"){
  # Model Parameter Setting
  N <- dim(counts)[1] # number of documents
  K <- dim(model$mu)[1] 

  # E step:    
  for (n in 1:N){
    for (k in 1:K){
      ## calculate the posterior based on the estimated mu and rho in the "log space"
      gamma[n,k] <- log(model$rho[k,1] + eps) + sum(counts[n,] * log(model$mu[k,] + eps))
    }
    # normalisation to sum to 1 in the log space
    logZ = logSum(gamma[n,])
    gamma[n,] = gamma[n,] - logZ
  }
  # converting back from the log space 
  gamma <- exp(gamma)
    
  # implement the hard E step  
  if (mode == "hard") {
    max_ind <- gamma == apply(gamma, 1, max)
    gamma[max_ind] <- 1 - (K - 1) * eps
    gamma[!max_ind] <- eps  
  }  
  return (gamma)
}

M.step <- function(gamma, model, counts){
  # Model Parameter Setting
  N <- dim(counts)[1]   # number of documents
  W <- dim(counts)[2]   # number of words i.e. vocabulary size
  K <- dim(model$mu)[1] # number of clusters
    
  # M step: Student needs to write this part for soft/hard EM

  # calculate rho(Aka. phi)(N_k for all clusters)
  model$rho <- matrix(colSums(gamma), ncol=1) / N
  for (k in 1:K) {
    # calculate mu
    model$mu[k,] <- ((gamma[,k] * counts) %>% colSums) / ((gamma[,k] * counts) %>% sum)
  }
    
  # Return the result
  return (model)
}

EM <- function(counts, K=4, max.epoch=10, seed=123456, mode = "soft"){

  # Model Parameter Setting
  N <- dim(counts)[1] # number of documents
  W <- dim(counts)[2] # number of unique words (in all documents)
  
  # Initialization
  model <- initial.param(W, K=K, seed=seed)
  gamma <- matrix(0, nrow = N, ncol = K)

  print(train_obj(model,counts))
  # Build the model
  for(epoch in 1:max.epoch){
    # E Step
    gamma <- E.step(gamma, model, counts, mode)  
    # M Step
    model <- M.step(gamma, model, counts)
    print(train_obj(model,counts)) 
  }
    
  # Return Model
  return(list("model"=model,"gamma"=gamma))
}

EM.main = function(mode, K, epochs=5, seed = 5201, real.label.visualize = FALSE) {
    set.seed(seed)
    
    data <- read.data(file.name='train6.txt', sample.size=0, seed=seed, pre.proc=TRUE, spr.ratio= .99)

    # word-document frequency matrix 
    counts <- data$word.doc.mat

    # calling the EM algorithm on the data
    res <- EM(counts, K=K, max.epoch=epochs, mode = mode, seed = seed)
    
    # visualization
    ## find the culster with the maximum probability (since we have soft assignment here)
    label.hat <- apply(res$gamma, 1, which.max)
    ## normalize the count matrix for better visualization
    counts <- counts %>% t %>% scale %>% t # only use when the dimensionality of the data (number of words) is large enough
    
    ## visualize the stimated clusters
    cluster.viz(counts, label.hat, paste0('Estimated Clusters (', mode,' EM)'))

    ## visualize the real clusters
    if (real.label.visualize) cluster.viz(counts, factor(data$label), 'Real Clusters')
}

EM.main("soft", 4, epochs=10, seed = NULL, real.label.visualize = FALSE)
EM.main("hard", 4, epochs=10, seed = NULL, real.label.visualize = TRUE)
