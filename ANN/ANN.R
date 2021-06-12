#Artificial Neural Network
library(nnet)
library(ggplot2)
#------Q1. numeric이 아닌 변수들 확인 및 boxplot
earthq<- read.csv("earthq.csv")
str(earthq)
target_idx <- 40
earthq_target <- as.factor(earthq[,target_idx])
earthq1 <- cbind(earthq[,-c(target_idx)],earthq_target)
char_idx <- c(9:15,27)
char_earthq<- earthq1[,char_idx]
char_earthq

for (i in 1:length(char_earthq)){
  char_var <- char_earthq[,i]
  box <- ggplot(earthq1, aes(x=char_var))+geom_bar()
  print(box+ggtitle(char_var[i]))
}


#-----Q2. 1-hot encoding 이진형 명목형 변수들로 바꿔주기
char <- names(char_earthq)
dummy <- data.frame(0)
for( i in 1:length(char_earthq)){
  x <- class.ind(char_earthq[,i])
  for(j in 1:ncol(x)){
    colnames(x)[j] <- paste("dummy-",char[i],"-",colnames(x)[j])
  }
  dummy <- cbind(dummy, x)
}
dummy <- dummy[,-1]
dummy


#데이터 셋 분할해주기    trn <-150000, val <- 50000, tst <- 60601
#데이터 전처리
earthq_data <- cbind(earthq1[,-c(1,char_idx)],dummy)
str(earthq_data)
#target 
target_idx2<-31
earthq_data_target <-earthq_data[target_idx2]
earthq_data_target
#input 값들 설정
#이진형태
binary_idx <- c(8:18,20:30)
binary_input <- lapply(earthq_data[,binary_idx],factor)
#수치형
numeric_idx <- c(1:7,19)
numeric_data <-earthq_data[,numeric_idx]
numeric_input <- scale(numeric_data, center=TRUE, scale=TRUE)
#dummy 변수
dummy_input <- earthq_data[,-c(target_idx2,binary_idx,numeric_idx)]
#최종 dataframe 생성
earthq_input <- data.frame(numeric_input, binary_input, dummy_input)
earthq_final <- data.frame(earthq_input,earthq_data_target)
input_idx <- c(1:68)
target_idx <- 69
#데이터 분할
set.seed(12345)
trn_idx <- sample(1:nrow(earthq_final),150000)
val_idx <- sample(1:nrow(earthq_final),50000)
tst_idx <- sample(1:nrow(earthq_final),60601)
earthq_trn <- data.frame(earthq_final[trn_idx,])
earthq_val <- data.frame(earthq_final[val_idx,])
earthq_tst <- data.frame(earthq_final[tst_idx,])




#예측률평가 
#Performance eval_fold
perf_eval_multi <- function(cm){
  #Simple Accuracy
  ACC = sum(diag(cm))/sum(cm)
  #Balanced Correction Rate
  BCR = 1
  for(i in 1:dim(cm)[1]){
    BCR = BCR * (cm[i,i]/sum(cm[i,]))
  }
  BCR = BCR^(1/dim(cm)[1])
  return(c(ACC,BCR))
}

#평가 matrix 생성
perf_table <- matrix(0, nrow = 2, ncol = 4)
colnames(perf_table) <- c("ANN","ANN(GA)","Decision TREE","MLR")
rownames(perf_table) <- c("Accuracy","BCR")
perf_table

#---------Q1
#ANN 훈련
input_idx <- c(1:68)
target_idx <- 69
ann_trn_input <- earthq_trn[,input_idx]
ann_trn_target <- class.ind(earthq_trn[,-input_idx])
ann_val_input <- earthq_val[,input_idx]
ann_val_target <- class.ind(earthq_val[,-input_idx])
#best number of nodes
nH=seq(from=5, to=30, by=5)
maxit=seq(from=100, to=500, by=100)
val_perf <- matrix(0, 30, 4)
temp <-1
for( i in 1:length(nH)){
  for(j in 1:length(maxit)){
    cat("Training ANN: number of hidden nodes:", nH[i],",maxit:", maxit[j],"\n")
    eval_fold <- c()
    #Training model
    trn_input <- ann_trn_input
    trn_target <- ann_trn_target
    tmp_nnet <- nnet(trn_input, trn_target, size=nH[i], maxit=maxit[j],silent = TRUE,MaxNWts = 10000)
    #eval_fold
    val_input <- ann_val_input
    val_target <- ann_val_target
    eval_fold <- rbind(eval_fold, cbind(max.col(val_target),max.col(predict(tmp_nnet,val_input))))
    #Confusion matrix
    cfm <- matrix(0,nrow = 3, ncol = 3)
    cfm[1,1] <- length(which(eval_fold[,1] == 1 & eval_fold[,2] == 1))
    cfm[1,2] <- length(which(eval_fold[,1] == 1 & eval_fold[,2] == 2))
    cfm[1,3] <- length(which(eval_fold[,1] == 1 & eval_fold[,2] == 3))
    cfm[2,1] <- length(which(eval_fold[,1] == 2 & eval_fold[,2] == 1))
    cfm[2,2] <- length(which(eval_fold[,1] == 2 & eval_fold[,2] == 2))
    cfm[2,3] <- length(which(eval_fold[,1] == 2 & eval_fold[,2] == 3))
    cfm[3,1] <- length(which(eval_fold[,1] == 3 & eval_fold[,2] == 1))
    cfm[3,2] <- length(which(eval_fold[,1] == 3 & eval_fold[,2] == 2))
    cfm[3,3] <- length(which(eval_fold[,1] == 3 & eval_fold[,2] == 3))
    
    val_perf[temp,1] <-nH[i]
    val_perf[temp,2] <-maxit[j]
    val_perf[temp,3:4] <- t(perf_eval_multi(cfm))
    temp <- temp + 1
  }
}
val_perf
#Check best and worst combination of ANN
best <- val_perf[order(val_perf[,4],decreasing = TRUE),]
colnames(best) <- c("nH","Maxit","ACC","BCR")
worst <- val_perf[order(val_perf[,4],decreasing = FALSE),]
colnames(worst) <- c("nH","Maxit","ACC","BCR")
best
worst
#-------------rang 찾기 
best_nH <- best[1,1]
best_maxit <- best[1,2]
rang <- c(0.3,0.5,0.7)
val_perf2 <- matrix(0, 3, 3)
for( i in 1:length(rang)){
  cat("Training ANN: rang", rang[i],"\n")
  eval_fold <- c()
  #Training model
  trn_input <- ann_trn_input
  trn_target <- ann_trn_target
  tmp_nnet <- nnet(trn_input, trn_target, size = best_nH, maxit = best_maxit, rang = rang[i],silent = TRUE,MaxNWts = 10000)
  #eval_fold
  val_input <- ann_val_input
  val_target <- ann_val_target
  eval_fold <- rbind(eval_fold, cbind(max.col(val_target),max.col(predict(tmp_nnet,val_input))))
  #Confusion matrix
  cfm <- matrix(0,nrow = 3, ncol = 3)
  cfm[1,1] <- length(which(eval_fold[,1] == 1 & eval_fold[,2] == 1))
  cfm[1,2] <- length(which(eval_fold[,1] == 1 & eval_fold[,2] == 2))
  cfm[1,3] <- length(which(eval_fold[,1] == 1 & eval_fold[,2] == 3))
  cfm[2,1] <- length(which(eval_fold[,1] == 2 & eval_fold[,2] == 1))
  cfm[2,2] <- length(which(eval_fold[,1] == 2 & eval_fold[,2] == 2))
  cfm[2,3] <- length(which(eval_fold[,1] == 2 & eval_fold[,2] == 3))
  cfm[3,1] <- length(which(eval_fold[,1] == 3 & eval_fold[,2] == 1))
  cfm[3,2] <- length(which(eval_fold[,1] == 3 & eval_fold[,2] == 2))
  cfm[3,3] <- length(which(eval_fold[,1] == 3 & eval_fold[,2] == 3))
  
  val_perf2[i,1] <-rang[i]
  val_perf2[i,2:3] <- t(perf_eval_multi(cfm))
}
val_perf2
best_val_perf2 <- val_perf2[order(val_perf2[,3],decreasing = TRUE),]
colnames(best_val_perf2) <- c("rang","ACC","BCR")
best_val_perf2
best_rang <- best_val_perf2[1,1]
best_rang 


#---------Q5input_idx <- c(1:68)
#traindata 와 validation 결합 
ann_tv <- rbind(earthq_trn, earthq_val)
ann_tv_input <- ann_tv[,input_idx]
ann_tv_target<-class.ind(ann_tv[,target_idx])
#test data 생성
ann_tst_input <-earthq_tst[,input_idx]
ann_tst_target<-class.ind(earthq_tst[,target_idx])
val_perf3 <- matrix(0, nrow=10, ncol=3)
colnames(val_perf3) <- c("iteration","ACC","BCR")
for (i in 1:10){
  eval_fold <-c()
  #Train model
  tmp_nnet <- nnet(ann_tv_input, ann_tv_target, size = best_nH, maxit = best_maxit, rang = best_rang, silent = TRUE,MaxNWts = 10000)
  #Test model
  eval_fold <-rbind(eval_fold, cbind(max.col(ann_tst_target),max.col(predict(tmp_nnet,ann_tst_input))))
  #Confusion matrix
  cfm <- matrix(0,nrow = 3, ncol = 3)
  cfm[1,1] <- length(which(eval_fold[,1] == 1 & eval_fold[,2] == 1))
  cfm[1,2] <- length(which(eval_fold[,1] == 1 & eval_fold[,2] == 2))
  cfm[1,3] <- length(which(eval_fold[,1] == 1 & eval_fold[,2] == 3))
  cfm[2,1] <- length(which(eval_fold[,1] == 2 & eval_fold[,2] == 1))
  cfm[2,2] <- length(which(eval_fold[,1] == 2 & eval_fold[,2] == 2))
  cfm[2,3] <- length(which(eval_fold[,1] == 2 & eval_fold[,2] == 3))
  cfm[3,1] <- length(which(eval_fold[,1] == 3 & eval_fold[,2] == 1))
  cfm[3,2] <- length(which(eval_fold[,1] == 3 & eval_fold[,2] == 2))
  cfm[3,3] <- length(which(eval_fold[,1] == 3 & eval_fold[,2] == 3))
  val_perf3[i,1] <-i
  val_perf3[i,2:3] <- t(perf_eval_multi(cfm))
}
best_val_perf3 <- val_perf3[order(val_perf3[,3],decreasing = TRUE),]
colnames(best_val_perf3) <- c("iteartion","ACC","BCR")
best_val_perf3
perf_table[,1] <- best_val_perf3[1,2:3]

#-----------Q6 Genetic algorithm
#fit BCR
fit_BCR <- function(string){
  sel_var_idx <- which(string==1)
  sel_x <- x[,sel_var_idx]
  xy <-data.frame(sel_x, y)
  #train model
  eval.fold <- c()
  tmp_nnet <- nnet(sel_x,y, size=best_nH, maxit=best_maxit, rang=best_rang, silent=TRUE, MaxNWts=10000)
  eval_fold <-rbind(eval_fold, cbind(max.col(ann_tst_target),max.col(predict(tmp_nnet,ann_tst_input))))
  #Confusion matrix
  cfm <- matrix(0,nrow = 3, ncol = 3)
  cfm[1,1] <- length(which(eval_fold[,1] == 1 & eval_fold[,2] == 1))
  cfm[1,2] <- length(which(eval_fold[,1] == 1 & eval_fold[,2] == 2))
  cfm[1,3] <- length(which(eval_fold[,1] == 1 & eval_fold[,2] == 3))
  cfm[2,1] <- length(which(eval_fold[,1] == 2 & eval_fold[,2] == 1))
  cfm[2,2] <- length(which(eval_fold[,1] == 2 & eval_fold[,2] == 2))
  cfm[2,3] <- length(which(eval_fold[,1] == 2 & eval_fold[,2] == 3))
  cfm[3,1] <- length(which(eval_fold[,1] == 3 & eval_fold[,2] == 1))
  cfm[3,2] <- length(which(eval_fold[,1] == 3 & eval_fold[,2] == 2))
  cfm[3,3] <- length(which(eval_fold[,1] == 3 & eval_fold[,2] == 3))
  GA_perf <- perf_eval_multi(cfm)
  return(GA_perf[2])
}

#GA start
library(glmnet)
library(GA)
x <- as.matrix(earthq_trn[,input_idx])
y <- class.ind(earthq_trn[,-input_idx])
earthq_target <- earthq_final[,69]
val_perf_GA<- matrix(0, nrow=3, ncol=2)
colnames(val_perf_GA) <- c("iteration","best idx")
#첫번째 반복 
GA1 <- ga(type="binary",fitness=fit_BCR, nBits=ncol(x),popSize=20,pcrossover=0.5,pmutation=0.01,maxiter=10,elitism=2,seed=123)
best_var_idx <- which(GA@solution==1)
best_var_idx1 <- c(best_var_idx)
best_var_idx1
#두번째 반복
GA2 <- ga(type="binary",fitness=fit_BCR, nBits=ncol(x),popSize=20,pcrossover=0.5,pmutation=0.01,maxiter=10,elitism=2,seed=231)
best_var_idx2 <- which(GA2@solution==1)
best_var_idx2 <- c(best_var_idx2)
best_var_idx2
#세번째 반복
GA3 <- ga(type="binary",fitness=fit_BCR, nBits=ncol(x),popSize=20,pcrossover=0.5,pmutation=0.01,maxiter=10,elitism=2,seed=321)
best_var_idx3 <- which(GA3@solution==1)
best_var_idx3 <- c(best_var_idx3)
best_var_idx3
#검사
best_var_idx1
best_var_idx2
best_var_idx3


#-------Q7
#새로운 데이터 생성 
sel_idx <- c(1,10,12,13,19,22,25,26,27,30,31,36,37,41,50,59,61,65,69)
new_earthq_trn <- earthq_trn[,sel_idx]
new_earthq_val <- earthq_val[,sel_idx]
new_earthq_tst <- earthq_tst[,sel_idx]
var_idx2 <- c(1:18)
target_idx2 <-19
ann_trn_input_GA <- new_earthq_trn[,var_idx2]
ann_trn_target_GA <- class.ind(new_earthq_trn[,target_idx2])
ann_val_input_GA <- new_earthq_val[,var_idx2]
ann_val_target_GA <- class.ind(new_earthq_val[,target_idx2])
ann_tst_input_GA <- new_earthq_tst[,var_idx2]
ann_tst_target_GA <- class.ind(new_earthq_tst[,target_idx2])
#평가지표 생성
val_perf4 <- matrix(0, nrow=1, ncol=2)
colnames(val_perf4) <- c("ACC","BCR")
#Genetic Algorithm 변수들로 ann 구축
eval_fold <-c()
#Train model
tmp_nnet <- nnet(ann_trn_input_GA, ann_trn_target_GA, size = best_nH, maxit = best_maxit, rang = best_rang, silent = TRUE,MaxNWts = 10000)
#Test model
eval_fold <-rbind(eval_fold, cbind(max.col(ann_val_target_GA),max.col(predict(tmp_nnet,ann_val_input_GA))))
#Confusion matrix
cfm <- matrix(0,nrow = 3, ncol = 3)
cfm[1,1] <- length(which(eval_fold[,1] == 1 & eval_fold[,2] == 1))
cfm[1,2] <- length(which(eval_fold[,1] == 1 & eval_fold[,2] == 2))
cfm[1,3] <- length(which(eval_fold[,1] == 1 & eval_fold[,2] == 3))
cfm[2,1] <- length(which(eval_fold[,1] == 2 & eval_fold[,2] == 1))
cfm[2,2] <- length(which(eval_fold[,1] == 2 & eval_fold[,2] == 2))
cfm[2,3] <- length(which(eval_fold[,1] == 2 & eval_fold[,2] == 3))
cfm[3,1] <- length(which(eval_fold[,1] == 3 & eval_fold[,2] == 1))
cfm[3,2] <- length(which(eval_fold[,1] == 3 & eval_fold[,2] == 2))
cfm[3,3] <- length(which(eval_fold[,1] == 3 & eval_fold[,2] == 3))
val_perf4[i,1:2] <- t(perf_eval_multi(cfm))
GA_val_perf <- val_perf4
GA_val_perf
perf_table[,2] <- GA_val_perf
perf_table

#--------Q8
library(tree)
library(party)
min_criterion <- c(0.6,0.7,0.8)
min_split <- c(300,900,1500,3000)
max_depth <- c(5,15,30)
#데이터 결합 (train data)
tree_tv_trn <- rbind(new_earthq_trn,new_earthq_val)
#test data 설정
tree_tst <- new_earthq_tst
tree_tst_target <- class.ind(tree_tst[,19])
#decision tree start
CART_pre_search_result<-matrix(0,length(min_criterion)*length(min_split)*length(max_depth),6)
colnames(CART_pre_search_result) <- c("min_criterion","min_split","max_detph","Accuracy","BCR","number of leaf nodes")
CART_pre_search_result
#optimal parameter 찾기
iter_cnt=1
for(i in 1:length(min_criterion)){
  for(j in 1:length(min_split)){
    for (k in 1:length(max_depth)){
      tmp_control <- ctree_control(mincriterion=min_criterion[i],minsplit=min_split[j],maxdepth=max_depth[k])
      tmp_tree <- ctree(earthq_target~., data=tree_tv_trn, controls=tmp_control)
      tmp_tree_prediction <- predict(tmp_tree, newdata=tree_tst)
      tmp_tree_cm <- table(tree_tst$earthq_target, tmp_tree_prediction)
      CART_pre_search_result[iter_cnt,4:5] <- t(perf_eval_multi(tmp_tree_cm))
      #parameter 넣어주기 
      CART_pre_search_result[iter_cnt,1] <- min_criterion[i]
      CART_pre_search_result[iter_cnt,2] <- min_split[j]
      CART_pre_search_result[iter_cnt,3] <- max_depth[k]
      CART_pre_search_result[iter_cnt,6] <- length(nodes(tmp_tree,unique(where(tmp_tree))))
      iter_cnt =iter_cnt+1
    }
  }
}
#BCR기준으로 내림차순 정열
best_tree <- CART_pre_search_result[order(CART_pre_search_result[,5],decreasing = TRUE),]
best_tree

#------새로운 트리 구축 최적의 조합들로
best_control <- ctree_control(mincriterion=0.8,minsplit=200,maxdepth=800)
tree_final <- ctree(earthq_target~., data=tree_tv_trn, controls=best_control)
plot(tree_final)
text(tree_final, type="simple")
best_tree_prediction <- predict(tree_final, newdata=tree_tst)
best_tree_cm <- table(tree_tst$earthq_target, best_tree_prediction)
t(perf_eval_multi(best_tree_cm))
perf_table[,3]<-t(perf_eval_multi(best_tree_cm))
perf_table


#-----------9 logistic regression 구축 
earthq_lm <- multinom(earthq_target~., data=tree_tv_trn)
summary(earthq_lm)
t(summary(earthq_lm)$coefficients)
earthq_lm_prediction <- predict(earthq_lm, newdata=tree_tst)
lm_cm <- table(tree_tst$earthq_target, earthq_lm_prediction)
t(perf_eval_multi(lm_cm))
perf_table[,4] <- t(perf_eval_multi(lm_cm))
perf_table












