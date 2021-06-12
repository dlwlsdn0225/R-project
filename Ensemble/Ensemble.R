#Artificial Neural Network
library(nnet)
library(ggplot2)
#performance Evaluation Function
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
#------데이터 전처리 과정
earthq<- read.csv("earthq.csv")
str(earthq)
target_idx <- 40
earthq_target <- as.factor(earthq[,target_idx])
earthq1 <- cbind(earthq[,-c(target_idx)],earthq_target)
char_idx <- c(9:15,27)
char_earthq<- earthq1[,char_idx]
char_earthq

# 1-hot encoding 이진형 명목형 변수들로 바꿔주기
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


#데이터 셋 분할해주기    
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
earthq_input_scaled <- data.frame(numeric_input, binary_input, dummy_input)
earthq_data_scaled <- data.frame(earthq_input_scaled,earthq_data_target)
input_idx <- c(1:68)
target_idx <- 69
#데이터 분할
set.seed(12345)
trn_idx <- sample(1:nrow(earthq_data_scaled),15000)
val_idx <- sample(1:nrow(earthq_data_scaled),5000)
tst_idx <- sample(1:nrow(earthq_data_scaled),6060)
earthq_trn_input <- earthq_input_scaled[trn_idx,]
earthq_trn_target <- class.ind(earthq_target[trn_idx])
earthq_val_input <- earthq_input_scaled[val_idx,]
earthq_val_target <- class.ind(earthq_target[val_idx])
earthq_tst_input <- earthq_input_scaled[tst_idx,]
earthq_tst_target <- class.ind(earthq_target[tst_idx])

#----------Q1. 최적의 단일 모형 구축하기 
#performance matrix 만들기
perf_table <- matrix(0, nrow = 8, ncol = 2)
rownames(perf_table) <- c("MLR","CART","ANN","CART_Bagging","CART_RF","ANN_Bagging","AdaBoost","GBM")
colnames(perf_table) <- c("Accuracy","BCR")
perf_table
#----MLR 구축하기
#MLR data 새로 구축
earthq_lm_data <-data.frame(earthq_input_scaled, earthq_target)
earthq_lm_trn <- earthq_lm_data[trn_idx,]
earthq_lm_val <- earthq_lm_data[val_idx,]
earthq_lm_tst <- earthq_lm_data[tst_idx,]
earthq_lm <- multinom(earthq_target~., data=earthq_lm_trn)
summary(earthq_lm)
t(summary(earthq_lm)$coefficients)
earthq_lm_prediction <- predict(earthq_lm, newdata=earthq_lm_tst)
lm_cm <- table(earthq_lm_tst$earthq_target, earthq_lm_prediction)
lm_cm
t(perf_eval_multi(lm_cm))
perf_table[1,] <- t(perf_eval_multi(lm_cm))
perf_table
#------CART 구축
library(tree)
library(party)
min_criterion <- c(0.6,0.7,0.8)
min_split <- c(300,900,1500,3000)
max_depth <- c(5,15,30)
#데이터 (train data)
earthq_lm_data <-data.frame(earthq_input_scaled, earthq_target)
earthq_CART_trn <- earthq_lm_data[trn_idx,]
earthq_CART_val <- earthq_lm_data[val_idx,]
earthq_CART_tst <- earthq_lm_data[tst_idx,]
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
      tmp_tree <- ctree(earthq_target~., data=earthq_CART_trn, controls=tmp_control)
      tmp_tree_prediction <- predict(tmp_tree, newdata=earthq_CART_val)
      tmp_tree_cm <- table(earthq_CART_val$earthq_target, tmp_tree_prediction)
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
best_control <- ctree_control(mincriterion=0.6,minsplit=900,maxdepth=30)
tree_final <- ctree(earthq_target~., data=earthq_CART_trn, controls=best_control)
plot(tree_final)
text(tree_final, type="simple")
best_tree_prediction <- predict(tree_final, newdata=earthq_CART_tst)
best_tree_cm <- table(earthq_CART_tst$earthq_target, best_tree_prediction)
best_tree_cm
t(perf_eval_multi(best_tree_cm))
perf_table[2,]<-t(perf_eval_multi(best_tree_cm))
perf_table


#--------------ANN
nH=seq(from=5, to=35, by=5)
maxit=seq(from=100, to=160, by=30)
rang <- seq(from=0.3,to=0.7,by=0.2)
val_perf<- matrix(0,length(nH)*length(maxit)*length(rang), 5)
colnames(val_perf) <- c("nH", "maxit","rang","Acc","BCR")
temp <-1
for( i in 1:length(nH)){
  for(j in 1:length(maxit)){
    for(k in 1:length(rang)){
      cat("Training ANN: number of hidden nodes:",nH[i],",maxit:", maxit[j],",rang:",rang[k],"\n")
      eval_fold <- c()
      #Training model
      trn_input <- earthq_trn_input
      trn_target <- earthq_trn_target
      tmp_nnet <- nnet(trn_input, trn_target, size=nH[i], maxit=maxit[j],rang=rang[k],silent = TRUE,MaxNWts = 10000)
      #eval_fold
      val_input <- earthq_val_input
      val_target <- earthq_val_target
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
      val_perf[temp,3] <- rang[k]
      val_perf[temp,4:5] <- t(perf_eval_multi(cfm))
      temp <- temp + 1
    }
  }
}
val_perf
best_val_perf <- val_perf[order(val_perf[,5],decreasing = TRUE),]
colnames(best_val_perf) <- c("nH","maxit","rang","ACC","BCR")
best_val_perf
best_nH <- 30
best_maxit <- 160
rang <- 0.7
#ANN 구축 
eval_fold <- c()
ctgs_nnet <- nnet(earthq_trn_input, earthq_trn_target, size = best_nH, maxit = best_maxit, rang = best_rang, silent = TRUE,MaxNWts = 10000)
eval_fold <- rbind(eval_fold, cbind(max.col(earthq_tst_target),max.col(predict(ctgs_nnet,earthq_tst_input))))
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
cfm
val_cm_perf <- perf_eval_multi(cfm)
val_cm_perf
perf_table[3,]<-val_cm_perf
perf_table
save.image()

#------------Q2 CART bagging
install.packages("ipred")
install.packages("mlbench")
install.packages("caret")
library(ipred)
library(rpart)
library(mlbench)
library(caret)
earthq_CART_trn <- earthq_lm_data[trn_idx,]
earthq_CART_val <- earthq_lm_data[val_idx,]
earthq_CART_tst <- earthq_lm_data[tst_idx,]
CART_bootstrap<- seq(from = 30, to = 300, by = 30)
val_perf_CB <- matrix(0,10,3)
colnames(val_perf_CB) <- c("bootstrap_CART","Acc", "BCR")
for(i in 1:length(CART_bootstrap)){
  cat("CART Bagging Training: the number of bootstrap:",CART_bootstrap[i],"\n")
  best_control <- cforest_control(mincriterion=0.6,minsplit=900,maxdepth=30,mtry=0, ntree=CART_bootstrap[i])
  tmp_bagging <- cforest(earthq_target ~ ., data = earthq_CART_trn, controls=best_control)
  tmp_bagging_prediction <- predict(tmp_bagging, newdata=earthq_CART_val)
  tmp_bagging_cm <- table(earthq_CART_val$earthq_target, tmp_bagging_prediction)
  val_perf_CB[i,2:3] <- t(perf_eval_multi(tmp_bagging_cm))
  val_perf_CB[i,1] <- bootstrap[i]
}
best_val_perf_CB <- val_perf_CB[order(val_perf_CB[,3],decreasing = TRUE),]
colnames(best_val_perf_CB) <- c("bootstrap","ACC","BCR")
best_val_perf_CB
best_bootstrap_CB <- best_val_perf_CB[1,1]
best_bootstrap_CB 
best_control <- cforest_control(mincriterion=0.6,minsplit=900,maxdepth=30,mtry=0, ntree=best_bootstrap_CB)
best_CB <- cforest(earthq_target ~ ., data = earthq_CART_trn, controls=best_control)
best_CB_prediction <- predict(best_CB, newdata=earthq_CART_tst)
best_CB_cm <- table(earthq_CART_tst$earthq_target, best_CB_prediction)
best_CB_cm
t(perf_eval_multi(best_CB_cm))
perf_table[4,]<-t(perf_eval_multi(best_CB_cm))




#-------Q3 CART random forest
library(randomForest)
CART_tree<- seq(from = 30, to = 300, by = 30)
val_perf_CRF <- matrix(0,10,3)
colnames(val_perf_CRF) <- c("tree_RF","Acc", "BCR")

for(i in 1:length(CART_tree)){
  cat("RandomForest Training:",CART_tree[i],"\n")
  tmp_CART_RF <- randomForest(earthq_target ~., data = earthq_CART_trn, ntree = CART_tree[i], mincriterion = best_criterion, min_split = best_split, maxdepth = max_depth, importance = TRUE, do.trace = TRUE)
  CART_RF_pred <- predict(tmp_CART_RF, newdata = earthq_CART_val, type = "class")
  CART_RF_cfm <- table(earthq_CART_val$earthq_target, CART_RF_pred)
  print(tmp_CART_RF)
  val_perf_CRF[i,1] = CART_tree[i]
  val_perf_CRF[i,2:3] = t(perf_eval_multi(CART_RF_cfm))
}
best_val_perf_CRF <- val_perf_CRF[order(val_perf_CRF[,3],decreasing = TRUE),]
colnames(best_val_perf_CRF) <- c("bootstrap","ACC","BCR")
best_val_perf_CRF
best_bootstrap_CRF <- best_val_perf_CRF[1,1]        #120
best_bootstrap_CRF
best_CRF <- randomForest(earthq_target ~., data = earthq_CART_trn, ntree = best_bootstrap_CRF, mincriterion = best_criterion, min_split = best_split, maxdepth = max_depth, importance = TRUE, do.trace = TRUE)
best_CRF_prediction <- predict(best_CRF, newdata=earthq_CART_tst)
best_CRF_cm <- table(earthq_CART_tst$earthq_target, best_CRF_prediction)
best_CRF_cm
t(perf_eval_multi(best_CRF_cm))
perf_table[5,]<-t(perf_eval_multi(best_CRF_cm))
perf_table
print(best_CRF)
plot(best_CRF)

Var_imp <- importance(best_CRF)
Var_imp[order(Var_imp[,4],decreasing = TRUE),]
summary(Var_imp)
barplot(Var_imp[order(Var_imp[,4],decreasing = TRUE),4])
#------Q3-3
library(ggplot2)
CB_BCR <- val_perf_CB[,3]
CRF_BCR <- val_perf_CRF[,3]
CART_CB_CRF <- data.frame(bootstrap,CB_BCR,CRF_BCR)
CART_CB_CRF
plot_Q3 <- ggplot(data=CART_CB_CRF)+geom_line(aes(x=bootstrap,y=CB_BCR, colour="red"))+geom_line(aes(x=bootstrap,y=CRF_BCR,colour="blue"))
Q3_lab <- labs(y="BCR", x = "bootstrap")
Q3_title <- ggtitle("CART_Bagging & CART_Random_Forest")
print(plot_Q3+Q3_lab+Q3_title)



#-------------------Q4 30번 반복
val_perf3 <- matrix(0,30,3)
colnames(val_perf3) <- c("iteration","Acc","BCR")
for(i in 1:30){
  cat("ANNTraining: iteration:",i,"\n")
  eval_fold <- c()
  trn_input <- earthq_trn_input
  trn_target <- earthq_trn_target
  tmp_nnet <- nnet(trn_input, trn_target, size = best_nH, maxit = best_maxit, rang = best_rang, MaxNWts = 10000)
  val_input <- earthq_val_input
  val_target <- earthq_val_target
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
  val_perf3[i,1]<- i
  val_perf3[i,2:3]<-  t(perf_eval_multi(cfm))
}
val_perf3
ANN_ACC_mean <- mean(val_perf3[,2])
ANN_ACC_std <- sd(val_perf3[,2])
ANN_BCR_mean <- mean(val_perf3[,3])
ANN_BCR_std <- sd(val_perf3[,3])
ANN_iter <- cbind(ANN_ACC_mean,ANN_ACC_std,ANN_BCR_mean,ANN_BCR_std)
ANN_iter


#----------Q5 ANN Bagging
library(caret)
library(doParallel)
bootstrap <- seq(from=30, to=300, by=30 )
cl <- makeCluster(4)
registerDoParallel(cl)
val_perf_ANN <- matrix(0, 100, 3)
colnames(val_perf_ANN) <- c("bootstrap", "Acc","BCR")
trn_input <- earthq_trn_input
trn_target <- earthq_trn_target
val_input <- earthq_val_input
val_target <- earthq_val_target
tst_input <- earthq_tst_input
tst_target <- earthq_tst_target
iter <- 1
for( i in 1:length(bootstrap)){
  cat("ANN Bagging: the number of bootstrap:",bootstrap[i],"\n")
  for(j in 1:10){
    cat("Repeat:",j,"\n")
    #Training model
    eval_fold <- c()
    bagging_ann_model <- avNNet(trn_input, trn_target, size=best_nH,maxit=best_maxit,rang=best_rang, decay=5e-4,repeats=bootstrap[i], bag=TRUE, allowParallel = TRUE ,trace=TRUE,MaxNWts = 10000)
    ans <- max.col(val_target)
    prey <- max.col(predict(bagging_ann_model,val_input))
    eval_fold <- rbind(eval_fold,cbind(ans,prey))
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
    val_perf_ANN[iter,1] <- bootstrap[i]
    val_perf_ANN[iter,2:3] <- t(perf_eval_multi(cfm))
    iter <- iter + 1
  }
}
val_perf_ANN

#평균 기록
val_perf_bagging_ANN <- matrix(0,10,5)
colnames(val_perf_bagging_ANN) <- c("bootstrap","mean_ACC","std_ACC","mean_BCR","std_BCR")
val_perf_bagging_ANN
iter_cnt <- 30
avg <- c(1,11,21,31,41,51,61,71,81,91)
for (i in 1:length(avg)){
  iter_cnt <- 30
  mean_ACC <- mean(val_perf_ANN[(avg[i]:avg[i]+9),2])
  std_ACC <- sd(val_perf_ANN[(avg[i]:avg[i]+9),2])
  mean_BCR <- mean(val_perf_ANN[(avg[i]:avg[i]+9),3])
  std_BCR <- sd(val_perf_ANN[(avg[i]:avg[i]+9),3])
  val_perf_bagging_ANN[i,1] <- iter_cnt
  val_perf_bagging_ANN[i,2] <- mean_ACC
  val_perf_bagging_ANN[i,3] <- std_ACC
  val_perf_bagging_ANN[i,4] <- mean_BCR
  val_perf_bagging_ANN[i,5] <- std_BCR
  iter_cnt <- iter_cnt + 30
}

best_val_BA <- val_perf_bagging_ANN[order(val_perf_bagging_ANN[,4],decreasing = TRUE),]
colnames(best_val_BA) <- c("bootstrap","mean_ACC","std_ACC","mean_BCR","std_BCR")
best_val_BA
best_bootstrap1 <- best_val_BA[1,1]
#Bagging 전체 평균 내주기 
BA_average <- matrix(0,1,4)
colnames(BA_average) <-c("mean_ACC","std_ACC","mean_BCR","std_BCR")
BA_average[1,1] <- mean(best_val_BA[,2])
BA_average[1,2] <- mean(best_val_BA[,3])
BA_average[1,3] <- mean(best_val_BA[,4])
BA_average[1,4] <- mean(best_val_BA[,5])
BA_average

#-----최적의 bootstrap 으로 train best_bootstrap=30
eval_fold <- c()
best_bagging_ann_model <- avNNet(trn_input, trn_target, size=best_nH,maxit=best_maxit,rang=best_rang, decay=5e-4,repeats=best_bootstrap1, bag=TRUE, allowParallel = TRUE ,trace=TRUE,MaxNWts = 10000)
ans <- max.col(tst_target)
prey <- max.col(predict(best_bagging_ann_model,tst_input))
eval_fold <- rbind(eval_fold,cbind(ans,prey))
#Confusion matrix
BA_cfm <- matrix(0,nrow = 3, ncol = 3)
BA_cfm[1,1] <- length(which(eval_fold[,1] == 1 & eval_fold[,2] == 1))
BA_cfm[1,2] <- length(which(eval_fold[,1] == 1 & eval_fold[,2] == 2))
BA_cfm[1,3] <- length(which(eval_fold[,1] == 1 & eval_fold[,2] == 3))
BA_cfm[2,1] <- length(which(eval_fold[,1] == 2 & eval_fold[,2] == 1))
BA_cfm[2,2] <- length(which(eval_fold[,1] == 2 & eval_fold[,2] == 2))
BA_cfm[2,3] <- length(which(eval_fold[,1] == 2 & eval_fold[,2] == 3))
BA_cfm[3,1] <- length(which(eval_fold[,1] == 3 & eval_fold[,2] == 1))
BA_cfm[3,2] <- length(which(eval_fold[,1] == 3 & eval_fold[,2] == 2))
BA_cfm[3,3] <- length(which(eval_fold[,1] == 3 & eval_fold[,2] == 3))

BA_cfm

perf_best_bagging_ann <- t(perf_eval_multi(BA_cfm))
perf_best_bagging_ann
perf_table[6,] <- perf_best_bagging_ann
perf_table


#--------Q6 Ada bBoost
install.packages("ada")
install.packages("adabag")
library(adabag)
library(ada)
no_iter <- c(70,100,130)
bag_frac <- c(0.4,0.5,0.6)
val_perf_AB <- matrix(0, 9, 4)
colnames(val_perf_AB) <- c("iter","bag_frac", "Acc","BCR")
val_perf_AB
iter<-1
for( i in 1:length(no_iter)){
  cat("ADA: the number of iter:",no_iter[i],"\n")
  for(j in 1:length(bag_frac)){
    cat("bag_frac:",bag_frac[j],"\n")
    eval_fold <- c()
    #Training model
    AB_model <- boosting(earthq_target~.,data=earthq_CART_trn,boos=TRUE, mfinal=no_iter[i],bag.frac=frac[j],control=rpart.control(min_criterion=0.6,minsplit=900))
    ans <- earthq_CART_val$earthq_target
    prey <- predict(AB_model,val_input)
    eval_fold <- rbind(eval_fold,cbind(ans,prey$class))
    #Confusion matrix
    AB_cfm <- matrix(0,nrow = 3, ncol = 3)
    AB_cfm[1,1] <- length(which(eval_fold[,1] == 1 & eval_fold[,2] == 1))
    AB_cfm[1,2] <- length(which(eval_fold[,1] == 1 & eval_fold[,2] == 2))
    AB_cfm[1,3] <- length(which(eval_fold[,1] == 1 & eval_fold[,2] == 3))
    AB_cfm[2,1] <- length(which(eval_fold[,1] == 2 & eval_fold[,2] == 1))
    AB_cfm[2,2] <- length(which(eval_fold[,1] == 2 & eval_fold[,2] == 2))
    AB_cfm[2,3] <- length(which(eval_fold[,1] == 2 & eval_fold[,2] == 3))
    AB_cfm[3,1] <- length(which(eval_fold[,1] == 3 & eval_fold[,2] == 1))
    AB_cfm[3,2] <- length(which(eval_fold[,1] == 3 & eval_fold[,2] == 2))
    AB_cfm[3,3] <- length(which(eval_fold[,1] == 3 & eval_fold[,2] == 3))
    val_perf_AB[iter,1] <- no_iter[i]
    val_perf_AB[iter,2] <- bag_frac[j]
    val_perf_AB[iter,3:4] <- t(perf_eval_multi((AB_cfm)))
    iter <- iter +1
  }
}

#최적의 boosting 만들기 
best_AB_perf <- val_perf_AB[order(val_perf_AB[,4],decreasing=TRUE),]
best_AB_perf
best_iter <- best_AB_perf[1,1]
best_bag_frac <- best_AB_perf[1,2]
best_AB_model <- boosting(earthq_target~., data=earthq_CART_trn, boos=TRUE, iter=best_iter, bag.frac=best_bag_frac, control=rpart.control(min_criterion=0.6,minsplit=900))
print(best_AB_model)
best_AB_pred <- predict(AB_model,tst_input)
best_AB_cfm <- table(earthq_CART_tst$earthq_target,best_AB_pred$class)
best_AB_eval <- t(perf_eval_multi(best_AB_cfm))
best_AB_eval
best_AB_cfm
perf_table[7,] <- best_AB_eval
perf_table

#---------------Q7 Gradient Boosting Machine
install.packages("gbm")
library(gbm)
trn <- earthq_data_scaled[sample(nrow(earthq_data_scaled),15000),]
val <- earthq_data_scaled[sample(nrow(earthq_data_scaled),5000),]
tst <- earthq_data_scaled[sample(nrow(earthq_data_scaled),6060),]
GBM_trn <- data.frame(trn[,input_idx],DamageYN = trn[,target_idx])
GBM_val <- data.frame(val[,input_idx],DamageYN = val[,target_idx])
GBM_tst <- data.frame(tst[,input_idx],DamageYN = tst[,target_idx])
bag_frac <- c(0.7,0.8,0.9)
ntree <- c(600,800,1000)
shrink <- c(0.02,0.04,0.06)
val_perf_GBM <- matrix(0,27,5)
colnames(val_perf_GBM) <- c("ntree","bag_frac","shrinkage","ACC","BCR")
iter <-1
for( i in 1:length(ntree)){
  cat("GBM: the number of iter:",ntree[i],"\n")
  for(j in 1:length(bag_frac)){
    cat("bag_frac:",bag_frac[j],"\n")
    for(k in 1:length(shrink)){
      cat("shrink:",shrink[k],"\n")
      eval_fold <- c()
      #Training model
      GBM_model <- gbm.fit(GBM_trn[,input_idx], GBM_trn[,target_idx], distribution = "multinomial",verbose = TRUE, n.trees =ntree[i], shrinkage = shrink[k],bag.fraction=bag_frac[j])
      ans <- GBM_val$DamageYN
      prey <- as.data.frame(predict(GBM_model,GBM_val[,input_idx],type="response"))
      prey <- max.col(prey)
      eval_fold <- rbind(eval_fold,cbind(ans,prey))
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
      val_perf_GBM[iter,1] <- ntree[i]
      val_perf_GBM[iter,2] <- bag_frac[j]
      val_perf_GBM[iter,3] <- shrink[k]
      val_perf_GBM[iter,4:5] <- t(perf_eval_multi(cfm))
      iter <- iter +1
    }
  }
}
val_perf_GBM
best_GBM_perf <- val_perf_GBM[order(val_perf_GBM[,5],decreasing=-TRUE),]
best_GBM_perf 
#최적 파라미터 ntree=1000, bag_frac=0.7, shrinkage=0.06
best_bag_frac <- best_GBM_perf[1,2]
best_ntree <- best_GBM_perf[1,1]
best_shrink <- best_GBM_perf[1,3]
#------[Q7-2]최적 파라미터로 GBM 실행 
best_GBM_model <- gbm.fit(GBM_trn[,input_idx],GBM_trn[,target_idx], distribution = "multinomial",verbose = TRUE, n.trees = best_ntree, shrinkage = best_shrink, bag.fraction=best_bag_frac)
summary(best_GBM_model)
best_GBM_prey <- as.data.frame(predict(best_GBM_model,GBM_tst[,input_idx],type="response"))
best_GBM_cfm <- table(max.col(best_GBM_prey), GBM_tst$DamageYN)
best_GBM_cfm
t(perf_eval_multi(best_GBM_cfm))
perf_best_GBM <- t(perf_eval_multi(best_GBM_cfm))
perf_table[8,] <- perf_best_GBM
perf_table



#----------Q9 데이터 불균형 해소 
library(caret)
up_data <- upSample(subset(earthq_data_scaled, select = -earthq_target),earthq_data_scaled$earthq_target)
table(up_data$Class)
table(earthq_data_scaled$earthq_target)
trn <- up_data[sample(nrow(up_data),15000),]
tst <- up_data[sample(nrow(up_data),6060),]
GBM_trn_data <- data.frame(trn[,input_idx],DamageYN = trn[,target_idx])
GBM_tst_data <- data.frame(tst[,input_idx],DamageYN = tst[,target_idx])
up_GBM <- gbm.fit(GBM_trn_data[,input_idx], GBM_trn_data[,target_idx],distribution = "multinomial",verbose = TRUE, n.trees = best_ntree, shrinkage = best_shrink, bag.fraction=best_bag_frac)
up_GBM_prey <-as.data.frame(predict(up_GBM, GBM_tst_data[,input_idx], type="response"))
up_GBM_cfm <- table(max.col(up_GBM_prey),GBM_tst_data$DamageYN)
up_GBM_cfm
t(perf_eval_multi(up_GBM_cfm))












