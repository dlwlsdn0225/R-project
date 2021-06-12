
library(psych)
library(moments)
library(corrplot)
#데이터 불러오기 
nba<- read.csv("nba.csv")
nrate <- nrow(nba)
nvar <- ncol(nba)
#의미 없는 변수들은 제거 
id_idx <-c(1,2,3,5,6)
nba_mlr_data <- cbind(nba[-c(id_idx)])
describe(nba_mlr_data)
for(i in 1:nvar){
  boxplot(nba_mlr_data[,i], main=names(nba_mlr_data)[i])
}
boxplot(nba_mlr_data)
#정규성을 띄지 않는 데이터 제거
id_idx1 <- c(13,22,25,26)
nba_mlr_data1 <- cbind(nba_mlr_data[-c(id_idx1)])
#차원 축소
id_idx2 <- c(7,8,10,11,13,15,16)
nba_mlr_data2 <- cbind(nba_mlr_data1[-c(id_idx2)])
#correlation 계산
corr_nba <- cor(nba_mlr_data2)
round(corr_nba,2)
corrplot(corr_nba,is.corr=FALSE, method="number")
#scatter plot matrix
pairs(~GP+W+L+MIN+PTS+FG.+X3P.+FT.+REB+AST+TOV+STL+PF+FP+X..., data=nba_mlr_data2)

#데이터 셋 나눠주기 
set.seed(12345)
nba_trn_idx <- sample(1:nrate, round(0.7*nrate))
nba_trn_data <- nba_mlr_data2[nba_trn_idx,]
nba_val_data <- nba_mlr_data2[-nba_trn_idx,]
mlr_nba <- lm(rankings~ ., data=nba_trn_data)
mlr_nba
summary(mlr_nba)
plot(mlr_nba)
#MAE, MAPE, RMSA
perf_eval_reg <- function(tgt_y, pre_y){
  rmse <- sqrt(mean((tgt_y-pre_y)^2))
  mae <- mean(abs(tgt_y- pre_y))
  mape <- 100*mean(abs((tgt_y-pre_y)/tgt_y))
  return(c(rmse,mae,mape))
}
perf_mat <- matrix(0, nrow=2, ncol=3)
rownames(perf_mat) <- c("before","after")
colnames(perf_mat) <-c("RMSE","MAE","MAPE")
perf_mat

#predict함수 사용
mlr_nba_haty <- predict(mlr_nba, newdata=nba_val_data)
perf_mat[1,] <- perf_eval_reg(nba_val_data$rankings, mlr_nba_haty)
perf_mat

#변수 줄이기 
id_idx3 <- c(4,9,12,13,14)
nba_mlr_data3 <- cbind(nba_mlr_data2[-c(id_idx3)])
nba_trn_data1 <- nba_mlr_data3[nba_trn_idx,]
nba_val_data1<- nba_mlr_data3[-nba_trn_idx,]
mlr_nba1 <- lm(rankings~ ., data=nba_trn_data1)
mlr_nba1
summary(mlr_nba1)
mlr_nba_haty1 <- predict(mlr_nba1, newdata=nba_val_data1)
perf_mat[2,] <- perf_eval_reg(nba_val_data1$rankings, mlr_nba_haty1)
perf_mat

#신뢰구간 구하기 
mlr_nba1 <- lm(rankings~ ., data=nba_trn_data1)
confint(mlr_nba1, level=0.95)
#결과 출력
plot(nba_trn_data$rankings,fitted(mlr_nba))
abline(0,1,lty=3)
#정규성 평가 
nba_resid <- resid(mlr_nba)
m <- mean(nba_resid)
std <- sqrt(var(nba_resid))
hist(nba_resid, prob=TRUE, xlab='x-variable', main="normal curve over histogram")
curve(dnorm(x,mean=m,sd=std), col="darkblue", lwd=2, add=TRUE,yaxt="n")
skewness(nba_resid)
kurtosis(nba_resid)

mlr_nba_haty <- predict(mlr_nba, newdata=nba_val_data)
