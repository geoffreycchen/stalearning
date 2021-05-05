rm(list=ls())
cat("\014")
print(getwd())
getwd()
setwd("/Volumes/W/Statistical Learning/project data")

pitches <- read.csv("pitches.csv", nrows = 600) # added 'nrows' argument to only read in 60K rows for speed
str(pitches) # dim: 2867154 x 40
# Need to find a faster way to load dataset
# Maybe filter down number of observations somehow. 
# Read more into link: (https://inbo.github.io/tutorials/tutorials/r_large_data_files_handling/)
# 3 out of 40 variables are CATEGORICAL
# 37 out of 40 variables are NUMERICAL
dim(pitches)

pitches$on_1b = factor(pitches$on_1b, levels=c("0","1"))
pitches$on_2b = factor(pitches$on_2b)
pitches$type = factor(pitches$type)
pitches$on_3b = factor(pitches$on_3b)
pitches$code = factor(pitches$code)
pitches$pitch_type = factor(pitches$pitch_type)


str(pitches)
summary(pitches)
# 14189 is max NULL values in a single column
# Still have sufficient dataset to work with (> 400 observations)

atbats <- read.csv("atbats.csv")
str(atbats) # dim: 740389 x 11
games.15_18 <- read.csv("games.csv")
str(games.15_18) # dim: 9718 x 17
player.names <- read.csv("player_names.csv")
str(player.names) # dim: 2218 x 3

# Joining pitches dataframe with atbats dataframe
# g_id column identifies the season for each game, first 4-digits are the year 
pitches_wgame <- merge(pitches, atbats, by.x = "ab_id", by.y = "ab_id")

# Joining player names from player.name df to atbats df
# Combine hitter's name, pitcher's name and outcome 
# of atbat in single df
atbats_Hnames <- merge(atbats, player.names, by.x ="batter_id", by.y ="id")
atbats.pAndb <- merge(atbats_Hnames, player.names, by.x = "pitcher_id", by.y = "id")
head(pitches_wgame)


# Started building out tree model...to be continued



library(tree)
library(glmnet)
library(ISLR)
library(randomForest)
library(ggplot2)
library(gridExtra)
library(timeR)
set.seed (1)

# setting up designed matrix

X = model.matrix(end_speed~., pitches_wgame)[,-4]  #transforms any qualitative variables into dummy variables
y = X[,4]

n  =   nrow(X)
p = ncol(X)
n.train = floor(0.8*n)
n.test = n-n.train

M = 100 # doing for 100 times 

Rsq.test.ls     =     rep(0,M)  #ls = lasso
Rsq.train.ls    =     rep(0,M)
Rsq.test.en     =     rep(0,M)  #en = elastic net
Rsq.train.en    =     rep(0,M)
Rsq.test.rid    =     rep(0,M)  #rid = ridge
Rsq.train.rid   =     rep(0,M)
Rsq.test.rf     =     rep(0,M)  #rf = rondam forrest
Rsq.train.rf    =     rep(0,M)

residual.test.ls     = matrix(0, nrow = n.test,  ncol = M)
residual.train.ls    = matrix(0, nrow = n.train, ncol = M)
residual.test.en     = matrix(0, nrow = n.test,  ncol = M)
residual.train.en    = matrix(0, nrow = n.train, ncol = M)
residual.test.rid    = matrix(0, nrow = n.test,  ncol = M)
residual.train.rid   = matrix(0, nrow = n.train, ncol = M)
residual.test.rf     = matrix(0, nrow = n.test,  ncol = M)
residual.train.rf    = matrix(0, nrow = n.train, ncol = M)


my_timer <- createTimer(precision = "ms")


# fit lasso and calculate and record the train and test R squares 
my_timer$start("lasso")
for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]

  a=1 # lasso
  cv.fit.ls        =     cv.glmnet(X.train, y.train, intercept = T, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train,intercept = T, alpha = a, lambda = cv.fit.ls$lambda.min)
  y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.ls[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.ls[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)  
  
  residual.test.ls[,m]  = y.test - y.test.hat
  residual.train.ls[,m] = y.train - y.train.hat
  cat(sprintf("m =%3.f| Rsq.test.ls = %.5f | Rsq.train.ls = %.5f| \n", m, Rsq.test.ls[m], Rsq.train.ls[m]))
}
my_timer$stop("lasso")

?glmnet()

# fit elastic-net and calculate and record the train and test R squares 
my_timer$start("elastic-net")
for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
 
  a=0.5 # elastic-net 0<a<1
  cv.fit.en        =     cv.glmnet(X.train, y.train, intercept = T, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train,intercept = T, alpha = a, lambda = cv.fit.en$lambda.min)
  y.train.hat      =     predict(fit, newx = X.train, type = "response")     # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       =     predict(fit, newx = X.test, type = "response")      # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.en[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.en[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)  
  
  residual.test.en[,m]  = y.test - y.test.hat
  residual.train.en[,m] = y.train - y.train.hat
  cat(sprintf("m =%3.f| Rsq.test.en=%.5f| Rsq.train.en=%.5f| \n", m, Rsq.test.en[m], Rsq.train.en[m]))
}
my_timer$stop("elastic-net")


# fit ridge and calculate and record the train and test R squares 
my_timer$start("ridge")
for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]

  a=0.0 # ridge
  cv.fit.rid       =     cv.glmnet(X.train, y.train, intercept = T, alpha = a, nfolds = 10)
  fit.rid          =     glmnet(X.train, y.train,intercept = T, alpha = a, lambda = cv.fit.rid$lambda.min)
  y.train.hat      =     predict(fit.rid, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       =     predict(fit.rid, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.rid[m]  =     1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.rid[m] =     1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)  
  
  residual.test.rid[,m]  = y.test - y.test.hat
  residual.train.rid[,m] = y.train - y.train.hat
  cat(sprintf("m =%3.f| Rsq.test.rid=%.5f| Rsq.train.rid=%.10f| \n", m, Rsq.test.rid[m], Rsq.train.rid[m]))
}
my_timer$stop("ridge")


# fit random forrest and calculate and record the train and test R squares 
my_timer$start("random forrest")
for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
  rf.fit           =     randomForest(X.train,y.train, mtry = floor(sqrt(p)), importance=TRUE)
  y.train.hat      =     predict(rf.fit, newdata = X.train) # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       =     predict(rf.fit, newdata = X.test) # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.rf[m]   =     1-mean((y.test - y.test.hat)^2) / mean((y.test - mean(y.test))^2)
  Rsq.train.rf[m]  =     1-mean((y.train - y.train.hat)^2) / mean((y.train - mean(y.train))^2) 
  
  residual.test.rf[,m]  = y.test - y.test.hat
  residual.train.rf[,m] = y.train - y.train.hat
  

  cat(sprintf("m =%3.f| Rsq.test.rf=%.5f| Rsq.train.rf=%.5f| \n", m, Rsq.test.rf[m], Rsq.train.rf[m]))
  }
my_timer$stop("random forrest")


# cross validation curve
par(mfrow=c(3,1))
plot(cv.fit.ls, main ="lasso")
plot(cv.fit.en, main ="elastic net")
plot(cv.fit.rid, main ="ridge")






# r-squared boxplot 
par(mfrow=c(1,2))
boxplot(Rsq.test.ls,Rsq.test.en,Rsq.test.rid,Rsq.test.rf, main ="test", col = "orange", border = "brown",
        notch = F,names = c("lasso", "enlastic-net", "ridge", "random forrest"), outline = F)

boxplot(Rsq.train.ls,Rsq.train.en,Rsq.train.rid, Rsq.train.rf, main ="train", col = "orange", border = "brown",notch = F,
        names = c("lasso", "enlastic-net", "ridge", "random forrest"), outline = F)


# 90% of r-squared
quantile(Rsq.test.ls,probs = c(.05,.95))
quantile(Rsq.test.rid,probs = c(.05,.95))
quantile(Rsq.test.en,probs = c(.05,.95))
quantile(Rsq.test.rf,probs = c(.05,.95))



# residual boxplot 
par(mfrow=c(1,2))

boxplot(residual.test.ls[,1], residual.test.en[,1], residual.test.rid[,1], residual.test.rf[,1], main ="test", col = "orange", border = "brown",
        notch = F,names = c("lasso", "enlastic-net", "ridge", "random forrest"), outline = F)

boxplot(residual.train.ls[,1], residual.train.en[,1], residual.train.rid[,1], residual.train.rf[,1], main ="train", col = "orange", border = "brown",
        notch = F,names = c("lasso", "enlastic-net", "ridge", "random forrest"), outline = F)



# ----------------------------------------------------------------------------------------------------
bootstrapSamples  =     100
beta.ls.bs        =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.en.bs        =     matrix(0, nrow = p, ncol = bootstrapSamples)       
beta.rid.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.rf.bs        =     matrix(0, nrow = p, ncol = bootstrapSamples)  



for (m in 1:bootstrapSamples){
  bs_indexes       =     sample(n, replace=T)
  X.bs             =     X[bs_indexes, ]
  y.bs             =     y[bs_indexes]
  
  # fit bs en
  a                =     0.5 
  cv.fit           =     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit.en           =     glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)  
  beta.en.bs[,m]   =     as.vector(fit.en$beta)
  
  # fit bs lasso
  a                =     1 
  cv.fit           =     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit.ls           =     glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)  
  beta.ls.bs[,m]   =     as.vector(fit.ls$beta)
  
  # fit boostra rid
  a                =     0 
  cv.fit           =     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit.rid          =     glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)  
  beta.rid.bs[,m]  =     as.vector(fit.rid$beta)
  
  cat(sprintf("Bootstrap Sample %3.f \n", m))
}

# need to change the bootstrapped standard errors calculaton to the upper and lower bounds


ls.bs.sd    = apply(beta.ls.bs, 1, "sd")
en.bs.sd    = apply(beta.en.bs, 1, "sd")
rid.bs.sd   = apply(beta.rid.bs, 1, "sd")
rf.bs.sd    = apply(beta.rf.bs, 1, "sd")




#--------------------------------------------------------------------------------------------------------------------------------

# fit en to the whole data
a = 0.5 # elastic-net
my_timer$start("elastic-net   whole data")
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.en           =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)
my_timer$stop("elastic-net   whole data")

# fit lasso to the whole data
a = 1 # lasso
my_timer$start("lasso   whole data")
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.ls           =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)
my_timer$start("lasso   whole data")

# fit rid to the whole data
a = 0 # ridge
my_timer$start("ridge   whole data")
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.rid          =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)
my_timer$start("ridge   whole data")

# fit random forest to the whole data
my_timer$start("random forest   whole data")
rf.fit           =     randomForest(X, y, mtry = floor(sqrt(p)),intercept = FALSE, importance=TRUE)
beta.rf.bs[,m]   =     as.vector(rf.fit$beta)
my_timer$start("random forest   whole data")

importance(rf.fit)
varImpPlot(rf.fit)

betaS.en             =     data.frame(c(1:p), as.vector(fit.en$beta), 2*en.bs.sd)
colnames(betaS.en)   =     c( "feature", "value", "err")

betaS.ls             =     data.frame(c(1:p), as.vector(fit.ls$beta), 2*ls.bs.sd)
colnames(betaS.ls)   =     c( "feature", "value", "err")

betaS.rid             =     data.frame(c(1:p), as.vector(fit.rid$beta), 2*rid.bs.sd)
colnames(betaS.rid)   =     c( "feature", "value", "err")


lsPlot =  ggplot(betaS.ls, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="orange", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.1) +
  ggtitle("lasso")

enPlot =  ggplot(betaS.en,main="Main title", aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="orange", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle("elastic net")

ridPlot =  ggplot(betaS.rid, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="orange", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle("ridge")

rfPlot = varImpPlot(rf.fit, main = "random forrest", col = "brown", border = "gray")

grid.arrange(lsPlot, enPlot, ridPlot, rfPlot, nrow = 4)

# change the order of factor levels by specifying the order explicitly.
betaS.ls$feature      =  factor(betaS.ls$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])
betaS.en$feature      =  factor(betaS.en$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])
betaS.rid$feature     =  factor(betaS.rid$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])

lsPlot =  ggplot(betaS.ls, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="orange", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.1) +
  ggtitle("lasso")

enPlot =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="orange", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.1) +
  ggtitle("elastic net")

ridPlot =  ggplot(betaS.rid, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="orange", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.1) +
  ggtitle("ridge")

grid.arrange(lsPlot, enPlot, ridPlot,nrow = 3)









