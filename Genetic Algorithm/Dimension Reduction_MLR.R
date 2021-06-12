#4단원 Dimensionality Reduction
library(glmnet)
library(GA)
library(psych)
library(moments)
library(corrplot)

# -------- Multiple Linear Regression 
nba<- read.csv("nba.csv")
nba
nrate <- nrow(nba)
nvar <- ncol(nba)
#의미 없는 변수들은 제거 
id_idx <-c(1,2,3,5,6)
nba_mlr_data <- cbind(nba[-c(id_idx)])
describe(nba_mlr_data)
#정규성을 띄지 않는 데이터 제거 (얘를 제거해주지 않으면 R2가 1이라서 variable selection 진행불가)
id_idx1 <- c(1,13,22,25,26)
nba_mlr_data1 <- cbind(nba_mlr_data[-c(id_idx1)])
nba_mlr_data1
#데이터 셋 나눠주기 및 preprocessing
nba_input_scale <- scale(nba_mlr_data1, center=TRUE, scale=TRUE)
nba_target <- nba$rankings
nba_input_scaled <- data.frame(nba_input_scale, nba_target)
set.seed(12345)
nba_trn_idx <- sample(1:nrate, round(0.7*nrate))
nba_trn <- nba_input_scaled[nba_trn_idx,]
nba_tst <- nba_input_scaled[-nba_trn_idx,]

"""
#차원 축소
id_idx2 <- c(7,8,10,11,13,15,16)
nba_mlr_data2 <- cbind(nba_mlr_data1[-c(id_idx2)])
nba_mlr_data2
"""
#MAE, MAPE, RMSE
perf_eval_reg <- function(tgt_y, pre_y){
  rmse <- sqrt(mean((tgt_y-pre_y)^2))
  mae <- mean(abs(tgt_y- pre_y))
  mape <- 100*mean(abs((tgt_y-pre_y)/tgt_y))
  return(c(rmse,mae,mape))
}
perf_mat <- matrix(0, nrow=5, ncol=6)
rownames(perf_mat) <- c("original","Forward Selection","Backward Selection","Stepwise Selection","Genetic Algorithm")
colnames(perf_mat) <-c("RMSE","MAE","MAPE","R-square","Time","No.Variables")
perf_mat

#------- All variables
#------------모든 변수를 사용하기 
full_model_nba <- lm(nba_target~ ., data=nba_trn)
summary(full_model_nba)
full_model_nba_coeff <- as.matrix(full_model_nba$coefficients)
full_model_nba_coeff
full_model_nba_prob <- predict(full_model_nba,newdata=nba_tst)
perf_mat[1,(1:3)] <- perf_eval_reg(nba_tst$nba_target, full_model_nba_prob)
perf_mat[1,4] <- summary(full_model_nba)$adj.r.square
perf_mat[1,5] <- NA
perf_mat[1,6] <- nrow(full_model_nba_coeff)
perf_mat


# ------ Forward Selection 
tmp_nba_x <- paste(colnames(nba_trn)[-23], collapse="+")
tmp_nba_xy <- paste("nba_target~", tmp_nba_x, collapse="")
as.formula(tmp_nba_xy)
forward_time_start <- proc.time()
forward_nba_model <- step(lm(nba_target~1, data=nba_trn), scope=list(upper=as.formula(tmp_nba_xy),lower=nba_target~1), direction="forward",trace=1)
forward_time_end <- proc.time()
forward_time <- forward_time_start -forward_time_end
perf_mat[2,5] <- abs(forward_time[3])
summary(forward_nba_model)
perf_mat[2,4] <- summary(forward_nba_model)$adj.r.square
forward_model_coeff <- as.matrix(forward_nba_model$coefficients,23,1)
forward_model_coeff
perf_mat[2,6] <- nrow(forward_model_coeff)


#prediction 만들기 
forward_nba_model_prob <- predict(forward_nba_model, type="response",newdata=nba_tst)
perf_mat[2,(1:3)] <- perf_eval_reg(nba_tst$nba_target, forward_nba_model_prob)

#----Backward Selection
backward_time_start <- proc.time()
backward_nba_model <- step(full_model_nba, scope=list(upper=as.formula(tmp_nba_xy),lower=nba_target~1), direction="backward",trace=1)
backward_time_end <- proc.time()
backward_time <- backward_time_start -backward_time_end
perf_mat[3,5] <- abs(backward_time[3])
summary(backward_nba_model)
perf_mat[3,4] <- summary(backward_nba_model)$adj.r.square
backward_nba_model_coeff <- as.matrix(backward_nba_model$coefficients,23,1)
backward_nba_model_coeff
perf_mat[3,6] <- nrow(backward_nba_model_coeff)
#prediction 만들기 
backward_nba_model_prob <- predict(backward_nba_model, type="response",newdata=nba_tst)
perf_mat[3,(1:3)] <- perf_eval_reg(nba_tst$nba_target, backward_nba_model_prob)
perf_mat


#-----Stepwise Selection
stepwise_time_start <- proc.time()
stepwise_nba_model <- step(lm(nba_target~1,data=nba_trn), scope=list(upper=as.formula(tmp_nba_xy),lower=nba_target~1), direction="both",trace=1)
stepwise_time_end <- proc.time()
stepwise_time <- stepwise_time_start- stepwise_time_end
perf_mat[4,5] <- abs(stepwise_time[3])
summary(stepwise_nba_model)
perf_mat[4,4] <- summary(stepwise_nba_model)$adj.r.square
stepwise_nba_model_coeff <- as.matrix(stepwise_nba_model$coefficients,23,1)
stepwise_nba_model_coeff
perf_mat[4,6] <- nrow(stepwise_nba_model_coeff)
#prediction 만들기 
stepwise_nba_model_prob <- predict(stepwise_nba_model, type="response",newdata=nba_tst)
perf_mat[4,(1:3)] <- perf_eval_reg(nba_tst$nba_target, stepwise_nba_model_prob)

#-----Genetic Algorithm
#fintess function
fit_R <- function(string){
  sel_var_idx <- which(string==1)
  sel_x <- x[,sel_var_idx]
  xy <- data.frame(sel_x,y)
  #train model
  GA_mlr<-lm(y~.,data=xy)
  rsquare <- (summary(GA_mlr)$adj.r.square)
  return(rsquare)
}
x <- as.matrix(nba_trn[,-23])
y <- nba_trn[,23]

#Genetic Algorithm을 통한 변수 선택
start_time_nba <- proc.time()
GA_nba <- ga(type="binary",fitness=fit_R, nBits= ncol(x), names=colnames(x), popSize=50, pcrossover=0.5, pmutation=0.01, maxiter=100, elitism=2, seed=123)
end_time_nba <- proc.time()
time_GA <- end_time_nba - start_time_nba
perf_mat[5,5] <- abs(time_GA[3])
summary(GA_nba)
GA_nba@solution[1,]
GA_nba@solution[2,]
GA_nba@solution[3,]
#GA_nba@solution (이 중 GP,W, L을 모두 선택한 2번 solution을 고른다)
best_nba_var_idx <- which(GA_nba@solution[2,]==1)
best_nba_var_idx
#최적의 변수들로 모델 train
GA_trn_nba_data <- nba_trn[ , c(best_nba_var_idx,23)]
GA_tst_nba_data <- nba_tst[ , c(best_nba_var_idx,23)]
GA_model_nba <- lm(nba_target~., data=GA_trn_nba_data)
summary(GA_model_nba)
perf_mat[5,4] <- summary(GA_model_nba)$adj.r.square
GA_model_nba_coeff <- as.matrix(GA_model_nba$coefficients,23,1)
GA_model_nba_coeff
perf_mat[5,6] <- nrow(GA_model_nba_coeff)
perf_mat
#예측하기 
GA_model_nba_prob <- predict(GA_model_nba, type="response", newdata=GA_tst_nba_data)
perf_mat[5,(1:3)] <- perf_eval_reg(nba_tst$nba_target, GA_model_nba_prob)
perf_mat


#하이퍼 파라미터 설정해주기 (population size, crossover-rate, mutation rate)
hyper_perf <- matrix(0, nrow=27, ncol=6)
colnames(hyper_perf)<-c("RMSE","MAE","MAPE","R-square","Time","No.Variables")
#population size variation (10부터100까지 증가, pcrossover=0.5, pmutation=0.01은 고정)
pop=10
i=1
time <- matrix(0,27,1)
while ((pop <= 100)&(i <=9)){
  start_time_nba <- proc.time()
  GA_nba_pop <- ga(type="binary",fitness=fit_R, nBits= ncol(x), names=colnames(x), popSize=pop, pcrossover=0.5, pmutation=0.01, maxiter=100, elitism=2, seed=123)
  end_time_nba <- proc.time()
  time_nba <- end_time_nba - start_time_nba
  hyper_perf[i,5] <- time_nba[3]
  best_pop_var_idx <- which(GA_nba_pop@solution[1,]==1)
  GA_trn_pop_data <- nba_trn[,c(best_pop_var_idx,23)]
  GA_tst_pop_data <- nba_tst[,c(best_pop_var_idx,23)]
  GA_model_pop <- lm(nba_target~., data=GA_trn_pop_data)
  hyper_perf[i,4] <- summary(GA_model_pop)$adj.r.square
  GA_pop_coeff <- as.matrix(GA_model_pop$coefficients,23,1)
  hyper_perf[i,6] <-nrow(GA_pop_coeff)
  GA_pop_prob <- predict(GA_model_pop,type="response", newdata=GA_tst_pop_data)
  hyper_perf[i,(1:3)] <- perf_eval_reg(nba_tst$nba_target, GA_pop_prob)
  pop=pop+10
  i=i+1
}

#pcrossover를 변형해주기 (pcrossover을 0.05부터 0.05씩 증가시킨다, (population=100과 pmutation=0.01은 고정))
pcross=0.05
while ((pcross<=0.45)&(i <=18)){
  start_time_pcross <- proc.time()
  GA_nba_pcross <- ga(type="binary",fitness=fit_R, nBits= ncol(x), names=colnames(x), popSize=100, pcrossover=pcross, pmutation=0.01, maxiter=100, elitism=2, seed=123)
  end_time_pcross <- proc.time()
  time_pcross <- end_time_pcross - start_time_pcross
  hyper_perf[i,5] <- time_pcross[3]
  best_pcross_var_idx <- which(GA_nba_pcross@solution[1,]==1)
  GA_trn_pcross_data <- nba_trn[,c(best_pcross_var_idx,23)]
  GA_tst_pcross_data <- nba_tst[,c(best_pcross_var_idx,23)]
  GA_model_pcross <- lm(nba_target~., data=GA_trn_pcross_data)
  hyper_perf[i,4] <- summary(GA_model_pcross)$adj.r.square
  GA_pcross_coeff <- as.matrix(GA_model_pcross$coefficients,23,1)
  hyper_perf[i,6] <-nrow(GA_pcross_coeff)
  GA_pcross_prob <- predict(GA_model_pcross,type="response", newdata=GA_tst_pcross_data)
  hyper_perf[i,(1:3)] <- perf_eval_reg(nba_tst$nba_target, GA_pcross_prob)
  pcross=pcross+0.05
  i=i+1
}


#pmutation를 변형해주기 (pmutation을 0.001에서 0.01까지 0.001씩 증가시킨다,(population=100, pcrossover=0.5로 고정))
pmut=0.001
while ((pmut <= 0.01)&(i <=27)){
  start_time_pmut <- proc.time()
  GA_nba_pmut <- ga(type="binary",fitness=fit_R, nBits= ncol(x), names=colnames(x), popSize=100, pcrossover=0.5, pmutation=pmut, maxiter=100, elitism=2, seed=123)
  end_time_pmut <- proc.time()
  time_pmut <- end_time_pmut - start_time_pmut
  hyper_perf[i,5] <- time_pmut[3]
  best_pmut_var_idx <- which(GA_nba_pmut@solution[1,]==1)
  GA_trn_pmut_data <- nba_trn[,c(best_pmut_var_idx,23)]
  GA_tst_pmut_data <- nba_tst[,c(best_pmut_var_idx,23)]
  GA_model_pmut <- lm(nba_target~., data=GA_trn_pmut_data)
  hyper_perf[i,4] <- summary(GA_model_pmut)$adj.r.square
  GA_pmut_coeff <- as.matrix(GA_model_pmut$coefficients,23,1)
  hyper_perf[i,6] <-nrow(GA_pmut_coeff)
  GA_pmut_prob <- predict(GA_model_pmut,type="response", newdata=GA_tst_pmut_data)
  hyper_perf[i,(1:3)] <- perf_eval_reg(nba_tst$nba_target, GA_pmut_prob)
  pmut=pmut+0.001
  i=i+1
}
hyper_perf











