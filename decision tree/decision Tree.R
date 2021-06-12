#4 Decision Tree
#-------Heart Disease
library(tree)
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


auroc <- function(model_prob,dataset){
  ROC1 <- data.frame(model_prob, dataset$heartYN)
  ROC2 <- arrange(ROC1, desc(model_prob), dataset$heartYN)
  colnames(ROC2) <- c("P(heart)","heart")
  TPR1 <- length(which(ROC2$heart==1))
  FPR1 <- length(which(ROC2$heart==0))
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
  colnames(ROC_data)<-c("P(heart)","heart","TPR(Sensitivity)","FPR(1-Specificity)")
  AUROC <- TPR_FPR %>%
    arrange(FPR) %>%
    mutate(area_rectangle = (lead(FPR)-FPR)*pmin(TPR,lead(TPR)),
           area_triangle = 0.5 * (lead(FPR)-FPR)*abs(TPR-lead(TPR))) %>%
    summarise(area = sum(area_rectangle + area_triangle, na.rm = TRUE))
  return(AUROC[,1])
}


#데이터 불러오기
heart <- read.csv("heart.csv")
#데이터 전처리
input_idx <- c(1:13)
target_idx <- 14
heart_input <- heart[, input_idx]
heart_target <- as.factor(heart[,target_idx])
heart_data <- data.frame(heart_input,heart_target)
#데이터 분할
set.seed(12345)
trn_idx <- sample(1:nrow(heart_data), round(0.6*nrow(heart_data)))
heart_val <- heart_data[-trn_idx,]
val_idx <- sample(1:nrow(heart_val), round(0.5*nrow(heart_val)))
heart_tst <- heart_val[-val_idx,]
tst_idx <- sample(1:nrow(heart_tst))

#Classification and Regression Tree
CART_trn <- data.frame(heart_input[trn_idx,], heartYN=heart_target[trn_idx])
CART_tst <- data.frame(heart_input[tst_idx,], heartYN=heart_target[tst_idx])
CART_val <- data.frame(heart_input[val_idx,], heartYN=heart_target[val_idx])

#-----post prunning 실행전
#Tree 학습
CART_bpost <- tree(heartYN~., CART_trn)
summary(CART_bpost)
#tree 그리기
plot(CART_bpost)
text(CART_bpost, pretty=1)

#post prunning 평가 matrix
perf_table_post <- matrix(0, nrow=3, ncol=7)
rownames(perf_table_post) <- c("Before Post_prunning","After post_prunning","Preprunning")
colnames(perf_table_post) <- c("TPR", "Precision","TNR", "Accuracy", "BCR", "F1-Measure","AUROC")
#prunning 하기 전 예측
CART_bpost_prey <- predict(CART_bpost, CART_tst,type="class")
CART_bpost_cm <- table(CART_tst$heartYN, CART_bpost_prey)
CART_bpost_cm
perf_table_post[1,(1:6)] <- perf_eval(CART_bpost_cm)
auroc_bpost <- auroc(CART_bpost_prey, CART_tst)
perf_table_post[1, 7] <- auroc_bpost
perf_table_post


#--------post-prunning 실행
set.seed(12345)
CART_post_cv <- cv.tree(CART_bpost,FUN=prune.misclass)
#prunning 결과 도출
plot(CART_post_cv$size, CART_post_cv$dev, type="b")
CART_post_cv
#최종 모델 고르기
CART_post_pruned <- prune.misclass(CART_bpost, best=10)
plot(CART_post_pruned)
text(CART_post_pruned, pretty=1)
#성능지표
CART_post_prey <- predict(CART_post_pruned, CART_tst, type="class")
CART_post_cm <- table(CART_tst$heartYN, CART_post_prey)
CART_post_cm
perf_table_post [2,(1:6)] <- perf_eval(CART_post_cm)
auroc_post <- auroc(CART_post_prey, CART_tst)
perf_table_post[2,7] <- auroc_post
perf_table_post

#-------PrePrunning

library(party)
library(ROCR)
min_criterion <- c(0.6,0.7,0.8,0.9,0.95)
min_split <- c(5,20,35,45,60)
max_depth <- c(2,5,8,11,14)

CART_pre_search_result=matrix(0,length(min_criterion)*length(min_split)*length(max_depth),11)
colnames(CART_pre_search_result) <- c("min_criterion","min_split","max_detph","TPR","Precision",
                                      "TNR","ACC","BCR","F1","AUROC","N_leaves")
#optimal parameter 찾기
#125개의 파라미터 조합 만들어주기
iter_cnt=1
for (i in 1:length(min_criterion)){
  for(j in 1:length(min_split)){
    for (k in 1:length(max_depth)){
      cat("CART Min criterion:", min_criterion[i],",Min split:",
          min_split[j], ",Max depth:", max_depth[k], "\n")
      tmp_control=ctree_control(mincriterion = min_criterion[i],minsplit=min_split[j],maxdepth = max_depth[k])
      tmp_tree <- ctree(heartYN ~., data=CART_trn, controls=tmp_control)
      tmp_tree_val_prediction <- predict(tmp_tree, newdata=CART_val)
      tmp_tree_val_response <- treeresponse(tmp_tree)
      tmp_tree_val_prob <- 1-unlist(tmp_tree_val_response, use.names=F)[seq(1,nrow(CART_val)*2,2)]
      tmp_tree_val_rocr <- prediction(tmp_tree_val_prob, CART_val$heartYN)
      tmp_tree_val_cm <- table(CART_val$heartYN, tmp_tree_val_prediction)
      #parameter
      CART_pre_search_result[iter_cnt,1]=min_criterion[i]
      CART_pre_search_result[iter_cnt,2]=min_split[j]
      CART_pre_search_result[iter_cnt,3]=max_depth[k]
      
      #confusion matrix
      CART_pre_search_result[iter_cnt,4:9] <- perf_eval(tmp_tree_val_cm)
      #AUROC
      #CART_pre_search_result[iter_cnt,10] <- unlist(performance(tmp_tree_val_rocr,"auc")@y.values)
      CART_pre_search_result[iter_cnt,10] <- 0
      #No of leaf nodes
      CART_pre_search_result[iter_cnt,11] <- length(nodes(tmp_tree,unique(where(tmp_tree))))
      iter_cnt =iter_cnt+1
    }
  }
}


#정렬시켜주기 
CART_pre_search_result <- CART_pre_search_result[order(CART_pre_search_result[,4], decreasing=T),]
CART_pre_search_result

best_criterion <- CART_pre_search_result[1,1]
best_split <-CART_pre_search_result[1,2]
best_depth <- CART_pre_search_result[1,3]

#-----Q4
#최적의 파라미터로 decision tree 구축
tree_control<- ctree_control(mincriterion=best_criterion,minsplit=best_split, maxdepth=best_depth)
CART_pre <- ctree(heartYN~., data=CART_trn, controls=tree_control)
summary(CART_pre)
#tree 그리기
plot(CART_pre)
text(CART_pre, type="simple")
CART_post_prey <- predict(CART_post_pruned, CART_tst, type="class")

#----Q5
#validation&train data 합쳐서 학습 후 
CART_trn_a <- rbind(CART_trn, CART_val)
CART_pre_a <- ctree(heartYN~., data=CART_trn_a, controls=tree_control)
CART_pre_prediction <-predict(CART_pre_a, new_data=CART_tst)
CART_pre_cm <- table(CART_tst$heartYN, CART_pre_prediction)
perf_table_post[3,(1:6)] <- perf_eval(CART_pre_cm)
CART_pre_prey <- predict(CART_pre_a, newdata=CART_tst)
perf_table_post[3,7] <- auroc(CART_pre_prey, CART_tst)
perf_table_post
summary(CART_pre_a)
#tree 그리기
plot(CART_pre_a)
text(CART_pre_a, type="simple")




#--------------Diabetes 데이터 
library(tree)
library(party)
library(ROCR)
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

#AUROC function 만들어주기 
d_auroc <- function(model_prob,dataset){
  ROC1 <- data.frame(model_prob, dataset$diabetesYN)
  ROC2 <- arrange(ROC1, desc(model_prob), dataset$diabetesYN)
  colnames(ROC2) <- c("P(diabetes)","diabetes")
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


#데이터 셋 불러오기 및 전처리
diabetes <- read.csv("diabetes.csv")
diabetes_input_idx <- c(1:8)
diabetes_target_idx <- 9
#scaling
diabetes_input <- diabetes[, diabetes_input_idx]
diabetes_target <- as.factor(diabetes[,diabetes_target_idx])
diabetes_data <- data.frame(diabetes_target, diabetes_input)
set.seed(12345)
#데이터 분할
d_trn_idx <- sample(1:nrow(diabetes_data), round(0.6*nrow(diabetes_data)))
diabetes_val <- diabetes_data[-dtrn_idx,]
d_val_idx <- sample(1:nrow(diabetes_val), round(0.5*nrow(diabetes_val)))
diabetes_tst <- diabetes_val[-dval_idx,]
d_tst_idx <- sample(1:nrow(diabetes_tst))

#Classification and Regression Tree
d_CART_trn <- data.frame(diabetes_input_scaled[d_trn_idx,], diabetesYN=diabetes_target[d_trn_idx])
d_CART_val <- data.frame(diabetes_input_scaled[d_val_idx,], diabetesYN=diabetes_target[d_val_idx])
d_CART_tst <- data.frame(diabetes_input_scaled[d_tst_idx,], diabetesYN=diabetes_target[d_tst_idx])
#Tree 학습
d_CART_bpost <- tree(diabetesYN~., d_CART_trn)
summary(d_CART_bpost)
#tree 그리기
plot(d_CART_bpost)
text(d_CART_bpost, pretty=1)

#Diabetes Decision Tree의 평가 matrix
d_perf_table <- matrix(0, nrow=3, ncol=7)
rownames(d_perf_table) <- c("Before Prunning","Post_prunning","Pre_prunning")
colnames(d_perf_table) <- c("TPR", "Precision","TNR", "Accuracy", "BCR", "F1-Measure","AUROC")

#Prunning 하기 전 예측
d_CART_bpost_prey <- predict(d_CART_bpost, d_CART_tst,type="class")
d_CART_bpost_cm <- table(d_CART_tst$diabetesYN, d_CART_bpost_prey)
d_CART_bpost_cm
d_perf_table[1,(1:6)] <- perf_eval(d_CART_bpost_cm)

#ROC curve 구축 
d_CART_bpost_response <- predict(d_CART_bpost, newdata=d_CART_tst, type="vector")
d_CART_bpost_response
d_CART_bpost_prob <- 1-unlist(d_CART_bpost_response, use.names=F)[seq(1,nrow(d_CART_tst),1)]
d_CART_bpost_rocr <- prediction(d_CART_bpost_prob, d_CART_tst$diabetesYN)
d_CART_bpost_perf <- performance(d_CART_bpost_rocr, "tpr", "fpr")
plot(d_CART_bpost_perf, col=5, lwd=3)
d_auroc_bpost <- unlist(performance(d_CART_bpost_rocr, "auc")@y.values)
d_perf_table[1, 7] <- d_auroc_bpost
d_perf_table


#--------diabetes data----post-prunning 실행
set.seed(12345)
d_CART_post_cv <- cv.tree(d_CART_bpost,FUN=prune.misclass)
#prunning 결과 도출
plot(d_CART_post_cv$size, d_CART_post_cv$dev, type="b")
d_CART_post_cv
#최종 모델 고르기
d_CART_post_pruned <- prune.misclass(d_CART_bpost, best=4)
plot(d_CART_post_pruned)
text(d_CART_post_pruned, pretty=1)
#성능지표
d_CART_post_prey <- predict(d_CART_post_pruned, d_CART_tst, type="class")
d_CART_post_cm <- table(d_CART_tst$diabetesYN, d_CART_post_prey)
d_CART_post_cm
d_perf_table [2,(1:6)] <- perf_eval(d_CART_post_cm)
#ROC curve 구축 
d_CART_post_response <- predict(d_CART_post_pruned, newdata=d_CART_tst, type="vector")
d_CART_post_response <- d_CART_post_response[,1]
d_CART_post_prob <- 1-unlist(d_CART_post_response, use.names=F)[seq(1,nrow(d_CART_tst),1)]
d_CART_post_rocr <- prediction(d_CART_post_prob, d_CART_tst$diabetesYN)
d_CART_perf <- performance(d_CART_post_rocr, "tpr","fpr")
plot(d_CART_perf, col=5, lwd=3)
d_auroc_post <- unlist(performance(d_CART_post_rocr, "auc")@y.values)
d_perf_table[1, 7] <- d_auroc_post
d_perf_table





#Diabetes data-----pre pruning
d_min_criterion <- c(0.6,0.7,0.8,0.9,0.95)
d_min_split <- c(5,15,35,55,70)
d_max_depth <- c(2,4,6,8,10)
d_CART_pre_search_result=matrix(0,length(d_min_criterion)*length(d_min_split)*length(d_max_depth),11)
colnames(d_CART_pre_search_result) <- c("min_criterion","min_split","max_detph","TPR","Precision",
                                        "TNR","ACC","BCR","F1","AUROC","N_leaves")
#optimal parameter 찾기
#125개의 파라미터 조합 만들어주기
iter_cnt=1
for (i in 1:length(d_min_criterion)){
  for(j in 1:length(d_min_split)){
    for (k in 1:length(d_max_depth)){
      cat("CART Min criterion:", d_min_criterion[i],",Min split:",
          d_min_split[j], ",Max depth:", d_max_depth[k], "\n")
      d_tmp_control=ctree_control(mincriterion = d_min_criterion[i],minsplit=d_min_split[j],maxdepth = d_max_depth[k])
      d_tmp_tree <- ctree(diabetesYN ~., data=d_CART_trn, controls=d_tmp_control)
      d_tmp_tree_val_prediction <- predict(d_tmp_tree, newdata=d_CART_val)
      d_tmp_tree_val_response <- treeresponse(d_tmp_tree)
      d_tmp_tree_val_prob <- 1-unlist(d_tmp_tree_val_response, use.names=F)[seq(1,nrow(d_CART_val)*2,2)]
      d_tmp_tree_val_rocr <- prediction(d_tmp_tree_val_prob, d_CART_val$diabetesYN)
      d_tmp_tree_val_cm <- table(d_CART_val$diabetesYN, d_tmp_tree_val_prediction)
      #parameter
      d_CART_pre_search_result[iter_cnt,1]=d_min_criterion[i]
      d_CART_pre_search_result[iter_cnt,2]=d_min_split[j]
      d_CART_pre_search_result[iter_cnt,3]=d_max_depth[k]
      #confusion matrix
      d_CART_pre_search_result[iter_cnt,4:9] <- perf_eval(d_tmp_tree_val_cm)
      #AUROC
      d_CART_pre_search_result[iter_cnt,10] <- unlist(performance(d_tmp_tree_val_rocr,"auc")@y.values)
      #No of leaf nodes
      d_CART_pre_search_result[iter_cnt,11] <- length(nodes(d_tmp_tree,unique(where(d_tmp_tree))))
      iter_cnt =iter_cnt+1
    }
  }
}

#정렬시켜주기 
d_CART_pre_search_result <- d_CART_pre_search_result[order(d_CART_pre_search_result[,10], decreasing=T),]
d_CART_pre_search_result
d_best_criterion <- d_CART_pre_search_result[1,1]
d_best_split <-d_CART_pre_search_result[1,2]
d_best_depth <- d_CART_pre_search_result[1,3]


#----Q4 pre pruned tree 구축
d_tree_control<- ctree_control(mincriterion=d_best_criterion,minsplit=d_best_split, maxdepth=d_best_depth)
d_CART_pre <- ctree(diabetesYN~., data=d_CART_trn, controls=d_tree_control)
summary(d_CART_pre)
#tree 그리기
plot(d_CART_pre)
text(d_CART_pre, type="simple")

#------Q5 validation과 train data 합쳐서 학습
#----Q5
#validation&train data 합쳐서 학습 후 
d_CART_trn_a <- rbind(d_CART_trn, d_CART_val)
d_CART_pre_a <- ctree(diabetesYN~., data=d_CART_trn_a, controls=d_tree_control)
d_CART_pre_prediction <- predict(d_CART_pre_a, newdata=d_CART_tst)
d_CART_pre_cm <- table(d_CART_tst$diabetesYN, d_CART_pre_prediction)
d_CART_pre_cm
d_perf_table[3,(1:6)] <- perf_eval(d_CART_pre_cm)
d_CART_pre_prey <- predict(d_CART_pre_a, newdata=d_CART_tst)
d_CART_pre_response <- treeresponse(d_CART_pre_a, newdata=d_CART_tst)
#ROC curve 그리기 
d_CART_pre_prob <- 1-unlist(d_CART_pre_response, use.names = F)[seq(1,nrow(d_CART_tst)*2,2)]
d_CART_pre_rocr <- prediction(d_CART_pre_prob, d_CART_tst$diabetesYN)
d_auroc_pre <- unlist(performance(d_CART_post_rocr, "auc")@y.values)
d_perf_table[3, 7] <- d_auroc_pre
d_CART_perf <- performance(d_CART_pre_rocr, "tpr", "fpr")
plot(d_CART_perf, col=5, lwd=3)
d_perf_table
#tree 그리기
plot(d_CART_pre_a)
text(d_CART_pre_a, type="simple")





#-----------Q6
#Data set 1과 2의 통합 성능평가지표
perf_table <- matrix(0, nrow=4, ncol=5)
rownames(perf_table) <- c("Heart-Logistic Regression","Heart-Decision Tree","Diabetes-Logistic Regression", "Diabetes-Decision Tree")
colnames(perf_table) <- c("TPR","TNR", "Accuracy", "BCR", "F1-Measure")
perf_table[2,] <- perf_eval(CART_bpost_cm)[c(1,3,4,5,6)]
perf_table[4,] <- perf_eval(d_CART_bpost_cm)[c(1,3,4,5,6)]
perf_table

#----"HEART" ------logistic Regression 실행
full_lr <- glm (heartYN ~., family=binomial, CART_trn)
#Test the model and evaluation
lr_response <- predict(full_lr, type="response", newdata=CART_tst)
lr_target <- CART_tst$heartYN
lr_predicted <- rep(0, length(lr_target))
lr_predicted[which(lr_response >=0.5)] <- 1
cm_full <- table(lr_target, lr_predicted)
perf_table[1,] <- perf_eval(cm_full)[c(1,3,4,5,6)]
CART_bpost_cm
cm_full

#----"Diabetes" ------logistic Regression 실행
d_full_lr <- glm (diabetesYN ~., family=binomial, d_CART_trn)
#Test the model and evaluation
d_lr_response <- predict(d_full_lr, type="response", newdata=d_CART_tst)
d_lr_target <- d_CART_tst$diabetesYN
d_lr_predicted <- rep(0, length(d_lr_target))
d_lr_predicted[which(d_lr_response >=0.5)] <- 1
d_cm_full <- table(d_lr_target, d_lr_predicted)
perf_table[3,] <- perf_eval(d_cm_full)[c(1,3,4,5,6)]
perf_table
d_CART_bpost_cm
d_cm_full

CART_bpost

#Q7 변수들 비교
summary(full_lr)
summary(d_full_lr)
summary(CART_bpost)
summary(d_CART_bpost)





