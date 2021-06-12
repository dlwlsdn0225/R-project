#Assignment7-------"Association Rule"
install.packages("arules")
install.packages("arulesViz")
install.packages("wordcloud")
library(stringr)
library(arules)
library(arulesViz)
library(wordcloud)
#-----[Q1]-----전처리 과정 
mooc_dataset <- read.csv("bigstudent.csv")
mooc_dataset
#1단계
Transaction_ID <- mooc_dataset$userid_DI
Institute <- mooc_dataset$institute
Course <- mooc_dataset$course_id
Region <- mooc_dataset$final_cc_cname_DI
Degree <- mooc_dataset$LoE_DI
#2단계 
gsub(" ","", Region )
Region

#3단계 
RawTransaction <- paste(Institute,Course,Region,Degree,sep="_")
RawTransaction
#4단계
MOOC_transactions <- paste(Transaction_ID,RawTransaction, sep=" ")
MOOC_transactions
#5단계
write.csv(MOOC_transactions,"MOOC_User_Course.csv",row.names=FALSE,quote=FALSE)
a<-read.csv("MOOC_User_Course.csv")
a

#-----[Q2]-----데이터 불러오기 및 기초 통계랑 확인 
#[Q2-1]
tmp_big<- read.transactions("MOOC_User_Course.csv", format="single", header=TRUE,cols=c(1,2), rm.duplicates=TRUE,skip=1)
summary(tmp_big)
str(tmp_big)
#[Q2-2]
itemname <- itemLabels(tmp_big)
itemcount <- itemFrequency(tmp_big)*nrow(tmp_big)
col <- brewer.pal(11,"Spectral")
wordcloud(words=itemname, freq=itemcount, min.freq = 500, scale=c(2,0.2), col=col, random.order=FALSE)
#[Q2-3]
itemFrequencyPlot(tmp_big, support=0.01, cex.names=0.8)


#-------[Q3]----- 규칙 생성 및 결과 해석
#-----[Q3-1]
support <- c(0.0005,0.001,0.0015)
confidence <- c(0.05,0.01,0.015)     
no_rules <- matrix(0,3,3)
colnames(no_rules) <- paste("Confidence=",confidence)
rownames(no_rules) <- paste("Support=",support)
no_rules

for(i in 1:length(support)){
  for(j in 1:length(confidence)){
    tmp_rules <- apriori(tmp_big, parameter=list(support=support[i],confidence=confidence[j]))
    cnt_rules <- length(tmp_rules)
    no_rules[i,j] <- cnt_rules
  }
}
no_rules


#--------[Q3-2]
rules <- apriori(tmp_big, parameter=list(support=0.001,confidence=0.05))
inspect(sort(rules, by="lift"))
inspect(sort(rules, by="support"))
inspect(sort(rules, by="confidence"))
#효용성 지표 생성
rules1 <- DATAFRAME(rules)
perf <- rules1$support*rules1$confidence*rules1$lift
rules1 <- cbind(rules1,perf)
sorted_rules <- rules1[order(rules1[,8], decreasing=TRUE),]
sorted_rules
#graph method
plot(rules, method="graph")
plot(rules, method="graph",max=15)
#규칙 
rule1 <- subset(rules,lhs%pin%c("MITx_6.002x_United"))
inspect(rule1)
rule2 <- subset(rules,lhs%pin%c("MITx_3.091x_United"))
inspect(rule2)
rule3 <- subset(rules,lhs%pin%c("MITx_6.00x_United"))
inspect(rule3)


#--------Extra Question
plot(rules, method="grouped")
plot(rules, method="matrix3D")




