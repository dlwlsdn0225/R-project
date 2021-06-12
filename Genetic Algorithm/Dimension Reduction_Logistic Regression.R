library(glmnet)
library(GA)
library(psych)
library(moments)
library(corrplot)
library(moments)
library(psych)
library(corrplot)
library(ggplot2)
library(dplyr)
library(ggplot)
library(tidyverse)
#--------------Logistic Regression
#performance Evaluation Function
perf_eval <- function(cm){
  #True Positive Rate: TPR
  TPR <- cm[2,2]/sum(cm[2,])
  #Precision
  PRE <- cm[2,2]/sum(cm[,2])
  #True Negative Rate: TNR
  TNR <- cm[1,1]/sum(cm[1,])
  #Simple Accuracy
  ACC <- (cm[1,1]+cm[2,2])/sum(cm)
  #Balanced Correction Rate
  BCR <- sqrt(TPR*TNR)
  #F1- Measure
  F1 <- 2*TPR*PRE/(TPR+PRE)
  return(c(TPR, PRE, TNR, ACC, BCR, F1))
}
#performance Evaluation Function
perf_table <- matrix(0, nrow=5, ncol=7)
perf_table
rownames(perf_table) <- c("All", "Forward","Backward","Stepwise","GA")
colnames(perf_table) <- c("AUROC(trn)","time(trn)","AUROC(Val)","Accuarcy(val)","BCR(val)","F1-Measure(val)","No.Variables")

#AUROC function 만들어주기 
# model=full_model, diabetes_tst=dataset
auroc <- function(model,dataset){
  model_prob <-predict(model, type="response", newdata=dataset)
  ROC1 <- data.frame(model_prob, dataset$diabetes_target)
  ROC2 <- arrange(ROC1, desc(model_prob), dataset$diabetes_target)
  colnames(ROC2) <- c("P(diabetes","diabetes")
  TPR1 <- length(which(ROC2$diabetes==1))
  FPR1 <- length(which(ROC2$diabetes==0))
  TPR_FPR <- cbind(0,0)
  colnames(TPR_FPR) <- c("TPR","FPR")
  TPR2 = 0 
  FPR2 = 0
  for(i in 1:nrow(ROC2)){
    if(ROC2[i,2]==1){
      TPR2 <- TPR2+1
    }else{
      FPR2 <- FPR2+1
    }
    TPR_tmp <- TPR2/TPR1
    FPR_tmp <- FPR2/FPR1
    TPR_FPR_tmp <- data.frame(TPR_tmp, FPR_tmp)
    colnames(TPR_FPR_tmp) <- c("TPR","FPR")
    TPR_FPR <- rbind(TPR_FPR, TPR_FPR_tmp)
  }
  initial <- c("0","0")
  ROC3 <- rbind(initial, ROC2)
  ROC_data <- data.frame(ROC3, TPR_FPR)
  colnames(ROC_data)<-c("P(diabetes)","diabetes","TPR(Sensitivity)","FPR(1-Specificity)")
  AUROC <- TPR_FPR %>%
    arrange(FPR) %>%
    mutate(area_rectangle = (lead(FPR)-FPR)*pmin(TPR,lead(TPR)),
           area_triangle = 0.5 * (lead(FPR)-FPR)*abs(TPR-lead(TPR))) %>%
    summarise(area = sum(area_rectangle + area_triangle, na.rm = TRUE))
  return(AUROC[,1])
}

#Load dataset
diabetes<- read.csv("diabetes.csv")
ndata <- nrow(diabetes)
nvar <- ncol(diabetes)
diabetes_input <- cbind(diabetes[-9])
target_idx <- 9
diabetes_target <- as.factor(diabetes[,target_idx])
diabetes_data <- data.frame(diabetes_target, diabetes_input)
diabetes_input_scaled <- scale(diabetes_input, center=TRUE, scale=TRUE)
diabetes_data
#데이터 분할해주기 
set.seed(12345)
trn_idx <- sample(1:nrow(diabetes_data), round(0.7*nrow(diabetes_data)))
diabetes_trn <- diabetes_data[trn_idx,] 
diabetes_tst <- diabetes_data[-trn_idx,]
#-------------All variables
full_model <- glm(diabetes_target ~., family=binomial, diabetes_trn)
summary(full_model)
full_model_coeff <- as.matrix(full_model$coefficients,9,1)
full_model_coeff
# ------train data set (ALL variable)
full_model_prob <- predict(full_model, type="response", newdata=diabetes_trn)
full_model_prey <- rep(0, nrow(diabetes_trn))
full_model_prey[which(full_model_prob >=0.5)] <- 1
full_model_cm <- table(diabetes_trn$diabetes_target, full_model_prey)
full_model_cm
perf_table[1,2] <- NA
#Make prediction
auroc_full <- auroc(full_model, diabetes_trn)
perf_table[1,1] <- auroc_full
perf_table
# ------validation data (ALL variable)
full_model_tst_prob <- predict(full_model, type="response", newdata=diabetes_tst)
full_model_tst_prey <- rep(0, nrow(diabetes_tst))
full_model_tst_prey[which(full_model_tst_prob >= 0.5)] <- 1
full_model_tst_cm <- table(diabetes_tst$diabetes_target, full_model_tst_prey)
full_model_tst_cm
auroc_full_tst <- auroc(full_model, diabetes_tst)
perf_table[1,3] <- auroc_full_tst
perf_table[1,7] <- nrow(full_model_coeff)
perf_table[1,(4:6)] <- perf_eval(full_model_tst_cm)[4:6]
perf_table


#---------------Forward Selection
tmp_x <- paste(colnames(diabetes_trn)[-1], collapse = "+")
tmp_xy <- paste("diabetes_target ~ ", tmp_x, collapse= "")
as.formula(tmp_xy)
#forward logistic regression

#-----------train data셋 에 대해 
start_forward <- proc.time()
forward_model <- step(glm(diabetes_target ~ 1, family = binomial, diabetes_trn), scope=list(upper=as.formula(tmp_xy),lower=diabetes_target~1), direction="forward",trace=1)
end_forward <- proc.time()
time_forward <- start_forward- end_forward
perf_table[2,2] <- abs(time_forward[3])
summary(forward_model)
forward_model_coeff <- as.matrix(forward_model$coefficients,9,1)
forward_model_coeff
perf_table[2,7] <- nrow(forward_model_coeff)
auroc_forward <- auroc(forward_model, diabetes_trn)
perf_table[2,1] <- auroc_forward
perf_table
#예측
forward_model_prob <- predict(forward_model, type="response",newdata=diabetes_trn)
forward_model_prey <- rep(0, nrow(diabetes_trn))
forward_model_prey[which(forward_model_prob >= 0.5)] <- 1
forward_model_cm <- table(diabetes_trn$diabetes_target, forward_model_prey)
forward_model_cm

#-----------validation data set 에 대해서 
auroc_forward_tst <- auroc(forward_model, diabetes_tst)
perf_table[2,3] <- auroc_forward_tst
#예측 
forward_model_tst_prob <- predict(forward_model, type="response", newdata=diabetes_tst)
forward_model_tst_prey <- rep(0, nrow(diabetes_tst))
forward_model_tst_prey[which(forward_model_tst_prob >= 0.5)] <- 1
forward_model_tst_cm <- table(diabetes_tst$diabetes_target, forward_model_tst_prey)
forward_model_tst_cm
#평가 
perf_table[2,(4:6)] <- perf_eval(forward_model_tst_cm)[4:6]
perf_table





#---------------Backward Selection
#-------- train data set 에 대해서 
start_backward <- proc.time()
backward_model <- step(glm(full_model, family = binomial, diabetes_trn), scope=list(upper=as.formula(tmp_xy),lower=diabetes_target~1), direction="backward",trace=1)
end_backward <- proc.time()
time_backward <- start_backward-end_backward
perf_table[3,2] <- abs(time_backward[3])
summary(backward_model)
backward_model_coeff <- as.matrix(backward_model$coefficients,9,1)
backward_model_coeff
perf_table[3,7] <- nrow(backward_model_coeff)
auroc_backward <- auroc(backward_model, diabetes_trn)
perf_table[3,1] <- auroc_backward
#예측
backward_model_prob <- predict(backward_model, type="response",newdata=diabetes_trn)
backward_model_prey <- rep(0, nrow(diabetes_trn))
backward_model_prey[which(backward_model_prob >= 0.5)] <- 1
backward_model_cm <- table(diabetes_trn$diabetes_target, backward_model_prey)
backward_model_cm

#-----------validation data set 에 대해서 
auroc_backward_tst <- auroc(backward_model, diabetes_tst)
perf_table[3,3] <- auroc_backward_tst
#예측 
backward_model_tst_prob <- predict(backward_model, type="response", newdata=diabetes_tst)
backward_model_tst_prey <- rep(0, nrow(diabetes_tst))
backward_model_tst_prey[which(backward_model_tst_prob >= 0.5)] <- 1
backward_model_tst_cm <- table(diabetes_tst$diabetes_target, backward_model_tst_prey)
backward_model_tst_cm
#평가 
perf_table[3,(4:6)] <- perf_eval(backward_model_tst_cm)[4:6]
perf_table



#------Stepwise Selection
#----------train data set에 대해서 
start_step <- proc.time()
step_model <- step(glm(diabetes_target ~ 1 , family = binomial, diabetes_trn), scope=list(upper=as.formula(tmp_xy),lower=diabetes_target~1), direction="both",trace=1)
end_step <- proc.time()
time_step <- start_step-end_step
perf_table[4,2] <- abs(time_step[3])
auroc_step <- auroc(step_model, diabetes_trn)
perf_table[4,1] <- auroc_step
summary(step_model)
step_model_coeff <- as.matrix(step_model$coefficients,9,1)
step_model_coeff
perf_table[4,7] <- nrow(step_model_coeff)
perf_table
#예측
step_model_prob <- predict(step_model, type="response",newdata=diabetes_trn)
step_model_prey <- rep(0, nrow(diabetes_trn))
step_model_prey[which(step_model_prob >= 0.5)] <- 1
step_model_cm <- table(diabetes_trn$diabetes_target, step_model_prey)
step_model_cm

#-----validation set 에 대해서 
auroc_step_tst <- auroc(step_model, diabetes_tst)
perf_table[4,3] <- auroc_step_tst
#예측 
step_model_tst_prob <- predict(step_model, type="response", newdata=diabetes_tst)
step_model_tst_prey <- rep(0, nrow(diabetes_tst))
step_model_tst_prey[which(step_model_tst_prob >= 0.5)] <- 1
step_model_tst_cm <- table(diabetes_tst$diabetes_target, step_model_tst_prey)
step_model_tst_cm
#평가 
perf_table[4,(4:6)] <- perf_eval(step_model_tst_cm)[4:6]
perf_table


#-------Genetic Algorithm
auroc2 <- function(model,dataset){
  model_prob <-predict(model, type="response", newdata=dataset)
  ROC1 <- data.frame(model_prob, dataset$y)
  ROC2 <- arrange(ROC1, desc(model_prob), dataset$y)
  colnames(ROC2) <- c("P(diabetes","diabetes")
  TPR1 <- length(which(ROC2$diabetes==1))
  FPR1 <- length(which(ROC2$diabetes==0))
  TPR_FPR <- cbind(0,0)
  colnames(TPR_FPR) <- c("TPR","FPR")
  TPR2 = 0 
  FPR2 = 0
  for(i in 1:nrow(ROC2)){
    if(ROC2[i,2]==1){
      TPR2 <- TPR2+1
    }else{
      FPR2 <- FPR2+1
    }
    TPR_tmp <- TPR2/TPR1
    FPR_tmp <- FPR2/FPR1
    TPR_FPR_tmp <- data.frame(TPR_tmp, FPR_tmp)
    colnames(TPR_FPR_tmp) <- c("TPR","FPR")
    TPR_FPR <- rbind(TPR_FPR, TPR_FPR_tmp)
  }
  initial <- c("0","0")
  ROC3 <- rbind(initial, ROC2)
  ROC_data <- data.frame(ROC3, TPR_FPR)
  colnames(ROC_data)<-c("P(diabetes)","diabetes","TPR(Sensitivity)","FPR(1-Specificity)")
  AUROC2 <- TPR_FPR %>%
    arrange(FPR) %>%
    mutate(area_rectangle = (lead(FPR)-FPR)*pmin(TPR,lead(TPR)),
           area_triangle = 0.5 * (lead(FPR)-FPR)*abs(TPR-lead(TPR))) %>%
    summarise(area = sum(area_rectangle + area_triangle, na.rm = TRUE))
  return(AUROC2[,1])
}
#fitness function AUROC for GA
fit_AUROC <- function(string){
  sel_var_idx <- which(string==1)
  sel_x <- x[,sel_var_idx]
  xy<- data.frame(sel_x,y)
  GA_lr <- glm(y~., family=binomial, xy)
  GA_auroc <- auroc2(GA_lr, xy)
  return(GA_auroc)
}

#----------train data set에 대해서 학습 
x <-as.matrix(diabetes_trn[,-1])
y <- diabetes_trn[,1]
start_GA <- proc.time()
GA_F1 <- ga(type="binary", fitness=fit_AUROC, nBits= ncol(x), names=colnames(x), popSize=50, pcrossover=0.5, pmutation=0.01, maxiter=100, elitism=2, seed=123)
end_GA <- proc.time()
time_GA <- start_GA- end_GA
perf_table[5,2] <- abs(time_GA[3])
perf_table
#최적의 변수들로 모델 train 시키기
best_var_idx <- which(GA_F1@solution==1)
GA_trn_data <- diabetes_trn[,c(best_var_idx,9)]
GA_trn_data
GA_model <- glm(diabetes_target~., family=binomial, GA_trn_data)
summary(GA_model)
GA_model_coeff <- as.matrix(GA_model$coefficients, 9,1)
GA_model_coeff
perf_table[5,7] <- nrow(GA_model_coeff)
auroc_GA<- auroc(GA_model,diabetes_trn)
perf_table[5,1] <-auroc_GA
#예측
GA_model_prob <- predict(GA_model, type="response",newdata=diabetes_trn)
GA_model_prey <- rep(0, nrow(diabetes_trn))
GA_model_prey[which(GA_model_prob >= 0.5)] <- 1
GA_model_cm <- table(diabetes_trn$diabetes_target, GA_model_prey)
GA_model_cm
#------validation data set에 대해서
GA_tst_data <- diabetes_tst[,c(best_var_idx,1)]
auroc_GA_tst<- auroc(GA_model,diabetes_tst)
perf_table[5,3] <-auroc_GA_tst
#예측
GA_model_tst_prob <- predict(GA_model, type="response",newdata=diabetes_tst)
GA_model_tst_prey <- rep(0, nrow(diabetes_tst))
GA_model_tst_prey[which(GA_model_tst_prob >= 0.5)] <- 1
GA_model_tst_cm <- table(diabetes_tst$diabetes_target, GA_model_tst_prey)
GA_model_tst_cm
#Performance evaluation
perf_table[5,(4:6)] <- perf_eval(GA_model_tst_cm)[4:6]
perf_table
GA_tst_data <- diabetes_tst[,c(best_var_idx,1)]


#hyper parameter 변경해주기  (popSize, pcrossover, maxiter)      "기본값" (population=50,pcrossover=0.5,pmutation=0.01,maxiter=50,elitism=2)
#최종 성능 평가를 위한 matrix 생성  
#하이퍼 파라미터 설정해주기 (population size, crossover-rate, mutation rate)
hyper_perf <- matrix(0, nrow=27, ncol=6)
colnames(hyper_perf)<-c("AUROC","Time","Accuracy","BCR","F1","No.Variables")

#-----------population을 변경, (pcrossover=0.5, maxiter=50)
x <-as.matrix(diabetes_trn[,-1])
y <- diabetes_trn[,1]
pop=10
i=1
while ((pop <= 50)&(i <=9)){
  start_time_pop <- proc.time()
  GA_pop <- ga(type="binary",fitness=fit_AUROC, nBits= ncol(x), names=colnames(x), popSize=pop, pcrossover=0.5, pmutation=0.01, maxiter=50, elitism=2, seed=123)
  end_time_pop <- proc.time()
  time_pop <- end_time_pop - start_time_pop
  hyper_perf[i,2] <- time_pop[3]
  best_pop_var_idx <- which(GA_pop@solution==1)
  GA_trn_pop_data <- diabetes_trn[,c(best_pop_var_idx,9)]
  GA_tst_pop_data <- diabetes_tst[,c(best_pop_var_idx,9)]
  GA_model_pop <- glm(diabetes_target ~., family=binomial,data=GA_trn_pop_data)
  GA_pop_coeff <- as.matrix(GA_model_pop$coefficients,9,1)
  auroc_pop <- auroc(GA_model_pop, GA_tst_pop_data)
  hyper_perf[i,1]<- auroc_pop
  hyper_perf[i,6] <-nrow(GA_pop_coeff)
  GA_pop_prob <- predict(GA_model_pop,type="response", newdata=GA_tst_pop_data)
  GA_pop_tst_prey <- rep(0, nrow(GA_tst_pop_data))
  GA_pop_tst_prey[which(GA_pop_prob>=0.5)] <-1
  GA_pop_cm <- table(GA_tst_pop_data$diabetes_target,GA_pop_tst_prey)
  hyper_perf[i,(3:5)] <- perf_eval(GA_pop_cm)[4:6]
  pop=pop+5
  i=i+1
}
hyper_perf

#-----------pcrossover을 변경, (population=50, maxiter=50)
x <-as.matrix(diabetes_trn[,-1])
y <- diabetes_trn[,1]
i=10
pcross=0.1
while ((pcross <=0.5)&(i <=18)){
  start_time_pcross <- proc.time()
  GA_pcross <- ga(type="binary",fitness=fit_AUROC, nBits= ncol(x), names=colnames(x), popSize=50, pcrossover=pcross, pmutation=0.01, maxiter=50, elitism=2, seed=123)
  end_time_pcross <- proc.time()
  time_pcross <- end_time_pcross - start_time_pcross
  hyper_perf[i,2] <- time_pcross[3]
  best_pcross_var_idx <- which(GA_pcross@solution==1)
  GA_trn_pcross_data <- diabetes_trn[,c(best_pcross_var_idx,9)]
  GA_tst_pcross_data <- diabetes_tst[,c(best_pcross_var_idx,9)]
  GA_model_pcross <- glm(diabetes_target ~., family=binomial,data=GA_trn_pcross_data)
  GA_pcross_coeff <- as.matrix(GA_model_pcross$coefficients,9,1)
  auroc_pcross <- auroc(GA_model_pcross, GA_tst_pcross_data)
  hyper_perf[i,1]<- auroc_pcross
  hyper_perf[i,6] <-nrow(GA_pcross_coeff)
  GA_pcross_prob <- predict(GA_model_pcross,type="response", newdata=GA_tst_pcross_data)
  GA_pcross_tst_prey <- rep(0, nrow(GA_tst_pcross_data))
  GA_pcross_tst_prey[which(GA_pcross_prob>=0.5)] <-1
  GA_pcross_cm <- table(GA_tst_pcross_data$diabetes_target,GA_pcross_tst_prey)
  hyper_perf[i,(3:5)] <- perf_eval(GA_pcross_cm)[4:6]
  pcross=pcross+0.05
  i=i+1
}

hyper_perf

#-----------maxiter을 변경, (population=50, pcrossover=0.5)
x <-as.matrix(diabetes_trn[,-1])
y <- diabetes_trn[,1]
iter=5
while ((iter <= 45)&(i <=27)){
  start_time_iter <- proc.time()
  GA_iter <- ga(type="binary",fitness=fit_AUROC, nBits= ncol(x), names=colnames(x), popSize=50, pcrossover=0.5, pmutation=0.01, maxiter=iter, elitism=2, seed=123)
  end_time_iter <- proc.time()
  time_iter <- end_time_iter - start_time_iter
  hyper_perf[i,2] <- time_iter[3]
  best_iter_var_idx <- which(GA_iter@solution==1)
  GA_trn_iter_data <- diabetes_trn[,c(best_iter_var_idx,9)]
  GA_tst_iter_data <- diabetes_tst[,c(best_iter_var_idx,9)]
  GA_model_iter <- glm(diabetes_target ~., family=binomial,data=GA_trn_iter_data)
  auroc_iter <- auroc(GA_model_iter, GA_tst_iter_data)
  hyper_perf[i,1]<- auroc_iter
  GA_iter_coeff <- as.matrix(GA_model_iter$coefficients,9,1)
  hyper_perf[i,6] <-nrow(GA_iter_coeff)
  GA_iter_prob <- predict(GA_model_iter,type="response", newdata=GA_tst_iter_data)
  GA_iter_tst_prey <- rep(0, nrow(GA_tst_iter_data))
  GA_iter_tst_prey[which(GA_iter_prob>=0.5)] <-1
  GA_iter_cm <- table(GA_tst_iter_data$diabetes_target,GA_iter_tst_prey)
  hyper_perf[i,(3:5)] <- perf_eval(GA_iter_cm)[4:6]
  iter=iter+5
  i=i+1
}
hyper_perf





