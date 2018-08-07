library(data.table)
library(magrittr)
library(ggplot2)
library(scales)
library(stringr)
library(quanteda)
library(gridExtra)
library(tm)
library(caret)
library(doSNOW)

dtrain <- fread('~/Desktop/train.tsv', showProgress = FALSE, nrows = 11000)
View(dtrain)
dtrain$item_description <- removeNumbers(dtrain$item_description)
dtrain$item_description <- removePunctuation(dtrain$item_description)
dtrain$item_description <- iconv(dtrain$item_description, "UTF-8", "ASCII", sub = "")
dtrain$item_description <- stripWhitespace(dtrain$item_description)
dtrain$desc_length <- nchar(dtrain$item_description)
dtrain[item_description == "No description yet", desc_length := 0]
dtrain[brand_name == "", brand_name := "Home made"]

dtrain$brand_name <- removeNumbers(dtrain$brand_name)
dtrain$brand_name <- removePunctuation(dtrain$brand_name)
dtrain$brand_name <- iconv(dtrain$brand_name, "UTF-8", "ASCII", sub="")
dtrain$brandName_length <- nchar(dtrain$brand_name)
dtrain[, c("level_1_cat", "level_2_cat") := tstrsplit(dtrain$category_name, split = "/", keep = c(1,2))]
dtrain$level_1_cat <- iconv(dtrain$level_1_cat, "UTF-8", "ASCII", sub = "")
dtrain$level_2_cat <- iconv(dtrain$level_2_cat, "UTF-8", "ASCII", sub = "")
dtrain$CountLevel1 <- nchar(dtrain$level_1_cat)
dtrain$CountLevel2 <- nchar(dtrain$level_2_cat)
dtrain[category_name == "", CountLevel1 := 0]
dtrain[category_name == "", CountLevel2 := 0]
View(dtrain)
names(dtrain) <- make.names(names(dtrain))

dtrainOrigin <- head(dtrain, 10000)
dtrainOrigin <- as.data.frame(dtrainOrigin)
dtestOrigin <- tail(dtrain, 1000)
dtestOrigin <- as.data.frame(dtestOrigin)

#USING TOKENIZATION
dtrainOrigin.tokens<-tokens_remove(tokens(dtrainOrigin$item_description, what = "word", remove_numbers = TRUE, remove_punct = TRUE, remove_symbol = TRUE, remove_hyphens = TRUE), stopwords("english"))

dtrainOrigin.tokens <- tokens_wordstem(dtrainOrigin.tokens, language = "english")

#dtrainOrigin.tokens <- tolower(dtrainOrigin.tokens)
dtrainOrigin.tokens.dfm <- dfm(dtrainOrigin.tokens, tolower = TRUE)
dtrainOrigin.tokens.matrix <- as.matrix(dtrainOrigin.tokens.dfm)
dtrainOrigin.tokens.df <- cbind(item_condition = dtrainOrigin$item_condition_id,shipping = dtrainOrigin$shipping, brandName_length = dtrainOrigin$brandName_length, as.data.frame(dtrainOrigin.tokens.dfm))
names(dtrainOrigin.tokens.df) <- make.names(names(dtrainOrigin.tokens.df))
#Term Frequency
term.frequency <- function(row) {
  row/sum(row)
}
#Inverse document frequency
inverse.doc.frequency <- function(col) {
  corpus.size <- length(col)
  doc.count <- length(which(col > 0))
  log10(corpus.size/doc.count)
}
#TF-IDF CALCULATION
tf.id <- function(tf,idf) {
  tf*idf
}
#Normalize the dataframe
dtrainOrigin.tokens.df <- apply(dtrainOrigin.tokens.matrix,1,term.frequency)
#Penlaize the dataframe
dtrainOrigin.tokens.idf <- apply(dtrainOrigin.tokens.matrix,2,inverse.doc.frequency)
#Rationaized the dataframe
dtrainOrigin.tokens.tfid <- apply(dtrainOrigin.tokens.df,2,tf.id,idf = dtrainOrigin.tokens.idf)
dtrainOrigin.tokens.tfid <- as.data.frame(dtrainOrigin.tokens.tfid)
dim(dtrainOrigin.tokens.tfid)
#Transpse thr tfid
dtrainOrigin.tokens.tfid <- t(dtrainOrigin.tokens.tfid)
#Checking for incompletion
incomplete.cases <- which(!complete.cases(dtrainOrigin.tokens.tfid))
#Fixing the value 0 if the attritubute is incomplete
#dtrainOrigin.tokens.tfid[incomplete.cases] <- rep(0.0,ncol(dtrainOrigin.tokens.tfid))
#dtrainOrigin.tokens.tfid[incomplete.cases] <- 0.0
dtrainOrigin.tokens.tfid <- na.omit(dtrainOrigin.tokens.tfid)
dim(dtrainOrigin.tokens.tfid)
#---TOKENIZATION END UP----#
#---N-GRAM---#
#dtrainOrigin.tokens.2Gram <- tokens_ngrams(dtrainOrigin.tokens, n = 1:2)
#dtrainOrigin.tokens.2Gram.dfm <- dfm(dtrainOrigin.tokens.2Gram, tolower = TRUE)
#dtrainOrigin.tokens.2Gram.matrix <- as.matrix(dtrainOrigin.tokens.2Gram.dfm)
#Garbage Collection
gc()
#APPLING TF_IDF
#dtrainOrigin.tokens.2Gram.dfm <- apply(dtrainOrigin.tokens.2Gram.matrix,1,term.frequency)
#dtrainOrigin.tokens.2Gram.idf <- apply(dtrainOrigin.tokens.2Gram.matrix,2,inverse.doc.frequency)
#dtrainOrigin.tokens.2Gram.tfid <- apply(dtrainOrigin.tokens.2Gram.dfm,2,tf.id,idf = dtrainOrigin.tokens.2Gram.idf)
#incomplete.cases2 <- which(!complete.cases(dtrainOrigin.tokens.2Gram.tfid))
#dtrainOrigin.tokens.2Gram.idf[incomplete.cases2] <- rep(0.0,ncol(dtrainOrigin.tokens.2Gram.tfid))
#dtrainOrigin.tokens.2Gram.tfid.df <- cbind(item_condition_id = dtrainOrigin$item_condition_id, price = dtrainOrigin$price, shipping = dtrainOrigin$shipping, brandName_length = dtrainOrigin$brandName_length, as.data.frame(dtrainOrigin.tokens.2Gram.tfid))
#names(dtrainOrigin.tokens.2Gram.tfid.df) <- make.names(names(dtrainOrigin.tokens.2Gram.tfid.df))
#gc()
library(irlba)
library(rsvd)
start.time <- Sys.time()
#train.irlba <- irlba(t(dtrainOrigin.tokens.tfid), nv = 7, maxit = 2)
#1.26hours
train.rsvd <- rsvd(t(dtrainOrigin.tokens.tfid), nv = 7)
total.time <- Sys.time() - start.time
total.time
#dtrainOrigin$price <- dtrainOrigin$price[-nrow(dtrainOrigin),]
#train.rsvd$v[5000] <- 0
trainO <- cbind(price = dtrainOrigin$price, train.rsvd$v)
trainO <- as.data.frame(trainO)
names(trainO)<- make.names(names(trainO))
#preparing testing data
dtestOrigin.tokens <-tokens_remove(tokens(dtestOrigin$item_description, what = "word", remove_numbers = TRUE, remove_punct = TRUE, remove_symbol = TRUE, remove_hyphens = TRUE), stopwords("english"))
dtestOrigin.tokens <- tokens_wordstem(dtestOrigin.tokens, language = "english")
dtestOrigin.tokens.dfm <- dfm(dtestOrigin.tokens, tolower = TRUE)
dtestOrigin.tokens.dfm <- dfm_select(dtestOrigin.tokens.dfm,dtrainOrigin.tokens.dfm)
dtestOrigin.tokens.dfm.matrix <- as.matrix(dtestOrigin.tokens.dfm)
#Normalize the test dataframe
dtestOrigin.tokens.df <- apply(dtestOrigin.tokens.dfm.matrix,1,term.frequency)
#Penlaize the test dataframe
#dtestOrigin.tokens.idf <- apply(dtestOrigin.tokens.dfm.matrix,2,inverse.doc.frequency)
#Rationaized the test dataframe
dtestOrigin.tokens.tfid <- apply(dtestOrigin.tokens.df,2,tf.id,idf = dtrainOrigin.tokens.idf)
dtestOrigin.tokens.tfid <- as.data.frame(dtestOrigin.tokens.tfid)
dim(dtestOrigin.tokens.tfid)
#Transpose the testdataframe
dtestOrigin.tokens.tfid <- t(dtestOrigin.tokens.tfid)
#Removing incomplete cases from test dataframe
dtestOrigin.tokens.tfid <- na.omit(dtestOrigin.tokens.tfid)
dim(dtestOrigin.tokens.tfid)
#Garbage Collection
gc()
#performing svd on testdataframe
start.time <- Sys.time()
test.rsvd <- rsvd(t(dtestOrigin.tokens.tfid), nv = 7)
total.time <- Sys.time() - start.time
total.time
testO <- as.data.frame(test.rsvd$v)
names(testO) <- make.names(names(testO))
set.seed(123)
cv.folds <- createMultiFolds(dtrainOrigin$price, k =10 , times = 3)
cv.cntrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, index = cv.folds)
library(caret)
View(trainO)
#ANFIS
library(frbs)
cl <- makeCluster(4, type = "SOCK")
registerDoSNOW(cl)
start.time <- Sys.time()
ANFIS.model <- train(price ~ ., data = trainO, method = "ANFIS")
stopCluster(cl)
total.time <- Sys.time() -start.time
total.time
ANFIS.model

#BAYESIAN RIDGE REGRESSION
#library(monomvn)
cl <- makeCluster(4, type = "SOCK")
registerDoSNOW(cl)
start.time <- Sys.time()
bayesian_ridge.regression.model <- train(price ~ ., data = trainO, method = "blassoAveraged", trControl = cv.cntrl, tuneLength = 7)
stopCluster(cl)
total.time <- Sys.time() -start.time
total.time
bayesian_ridge.regression.model

#BOODTED LINEAR MODEL
#library(bst)
#library(plyr)
cl <- makeCluster(4, type = "SOCK")
registerDoSNOW(cl)
start.time <- Sys.time()
BoostedLinearModel.model <- train(price ~ ., data = trainO, method = "BstLm", trControl = cv.cntrl, tuneLength = 7)
stopCluster(cl)
total.time <- Sys.time() -start.time
total.time
BoostedLinearModel.model

#Boosted Tree
cl <- makeCluster(4, type = "SOCK")
registerDoSNOW(cl)
start.time <- Sys.time()
BoostedTree.model <- train(price ~ ., data = trainO, method = "bstTree", trControl = cv.cntrl, tuneLength = 7)
stopCluster(cl)
total.time <- Sys.time() -start.time
total.time
BoostedTree.model

#Elasticnet
#library(elasticnet)
cl <- makeCluster(4, type = "SOCK")
registerDoSNOW(cl)
start.time <- Sys.time()
Elasticnet.model <- train(price ~ ., data = trainO, method = "enet", trControl = cv.cntrl, tuneLength = 7)
stopCluster(cl)
total.time <- Sys.time() -start.time
total.time
Elasticnet.model

#ExtremeGradientBoosting(Linear)
#library(xgboost)
cl <- makeCluster(4, type = "SOCK")
registerDoSNOW(cl)
start.time <- Sys.time()
ExtremeGradientBoosting.model <- train(price ~ ., data = trainO, method = "xgbLinear", trControl = cv.cntrl, tuneLength = 7)
stopCluster(cl)
total.time <- Sys.time() -start.time
total.time
ExtremeGradientBoosting.model

#Knn
cl <- makeCluster(4, type = "SOCK")
registerDoSNOW(cl)
start.time <- Sys.time()
Knn.model <- train(price ~ ., data = trainO, method = "knn", trControl = cv.cntrl, tuneLength = 7)
stopCluster(cl)
total.time <- Sys.time() -start.time
total.time
Knn.model

#Least Angle Regression
cl <- makeCluster(4, type = "SOCK")
registerDoSNOW(cl)
start.time <- Sys.time()
LeastAngleRegression.model <- train(price ~ ., data = trainO, method = "lars", trControl = cv.cntrl, tuneLength = 7)
stopCluster(cl)
total.time <- Sys.time() -start.time
total.time
LeastAngleRegression.model

#Linear Regression
cl <- makeCluster(4, type = "SOCK")
registerDoSNOW(cl)
start.time <- Sys.time()
LinearRegression.model <- train(price ~ ., data = trainO, method = "lm", trControl = cv.cntrl, tuneLength = 7)
stopCluster(cl)
total.time <- Sys.time() -start.time
total.time
LinearRegression.model

#Essemble model using H2o package
library(h2o)
library(h2oEnsemble)
h2o.init(nthreads = 1)
h2o.removeAll()
dtrain <- h2o.importFile("/home/aniket/Desktop/file/trainO1.csv")
dtest <- h2o.importFile("/home/aniket/Desktop/file/testO1.csv")
y <- "price"
x <- setdiff(names(dtrain), y)
learner <- c("h2o.glm.wrapper", "h2o.randomForest.wrapper", "h2o.gbm.wrapper", "h2o.deeplearning.wrapper")
metalearner <- "h2o.deeplearning.wrapper"
start.time <- Sys.time()
emsemble.fit <- h2o.ensemble(x = x,y = y, training_frame = dtrain, learner = learner, metalearner = metalearner, cvControl = list(V = 5))
total.time <- Sys.time() - start.time
total.time

perf <- h2o.ensemble_performance(emsemble.fit, newdata = dtest)




