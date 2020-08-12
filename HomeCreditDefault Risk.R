setwd("~/Downloads")
df<-read.csv("hcdr.csv")

library(e1071)
library(dplyr)
library(randomForest)
library(mlbench)
library(caret)
library(pROC)
library(ROCR)
library(gbm)

df$TARGET<-factor(df$TARGET)

#*******************DATA PREPARATION***********************
df = subset(df, select = -CNT_FAM_MEMBERS )
df = subset(df, select = -OWN_CAR_AGE )
df$AGE<- -round(df$DAYS_BIRTH /365)
df$EMPLOYED<- -round(df$DAYS_EMPLOYED /365)
df$REGISTRATION<- -round(df$DAYS_REGISTRATION /365)
df$ID_PUBLISH<- -round(df$DAYS_ID_PUBLISH /365)
df$NUM_CHANGE<- -round(df$DAYS_LAST_PHONE_CHANGE /365)
df = subset(df, select = -DAYS_EMPLOYED )
df = subset(df, select = -DAYS_BIRTH)
df = subset(df, select = -DAYS_REGISTRATION )
df = subset(df, select = -DAYS_ID_PUBLISH )
df = subset(df, select = -DAYS_LAST_PHONE_CHANGE)

#******************Feature Importance******************
set.seed(7)
model<-glm(TARGET~., data=df, family = binomial )
importance <- varImp(model, scale=FALSE)
print(importance)
plot(importance)

#**********PCA(PRINCIPAL COMPONENT ANALYSIS)*************
v_keep<-c('AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION')
primary_df$TARGET<-as.numeric(primary_df$TARGET)
M<-primary_df[,v_keep]
prcomp_M <- prcomp(M, 2)
summary(prcomp_M)
plot(prcomp_M)
screeplot(prcomp_M)
biplot(prcomp_M)
require("ggfortify")
autoplot(
  object = prcomp_M,
  data = primary_df,
  colour = "DAYS_EMPLOYED")
factoextra::fviz_eig(prcomp_M)
factoextra::fviz_pca_ind(
  prcomp_M,
  col.ind = "cos2",
  gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
  repel = TRUE)
factoextra::fviz_pca_var(
  X = prcomp_M,
  col.var = "contrib",
  gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
  repel = TRUE)
factoextra::fviz_pca_biplot(
  X = prcomp_M,
  repel = TRUE,
  col.var = "#2E9FDF",
  col.ind = "#696969")
factoextra::get_eigenvalue(prcomp_M)
get_pca_var_M <- factoextra::get_pca_var(prcomp_M)
get_pca_ind_M <- factoextra::get_pca_ind(prcomp_M)
prcomp_M <- data.frame(prcomp(  
  x = M,    center = FALSE,    
  scale. = FALSE  )$x[,1:2],  
  Name = rownames(primary_df), 
  Cluster = as.character(prcomp_M$x), 
  stringsAsFactors = FALSE)

#**************LOGISTIC REGRESSION*******************
glm.fit=glm(TARGET~ EXT_SOURCE_3 + CODE_GENDER+ AMT_GOODS_PRICE + AMT_CREDIT+ EXT_SOURCE_1 + DAYS_EMPLOYED + FLAG_OWN_CAR + AMT_ANNUITY+ DAYS_ID_PUBLISH + FLAG_WORK_PHONE+ NAME_FAMILY_STATUS, data=df, family = binomial)
summary(glm.fit)

#Training the data
prediction <- predict(glm.fit, df, type = "response")
summary(prediction)
confusion <- table(df$TARGET, prediction >= 0.0807)
confusion
round(exp(coef(glm.fit)),3)

#Testing the data
application_test = read.csv('seri.csv')
application_test=subset(application_test, select = -Unnamed..0 )
prediction <- predict(glm.fit, application_test, type = "response")
summary(prediction)
submission <- cbind(application_test, prediction)
submission$TARGET <- ifelse(submission$prediction>0.0823,1,0)

#ROC Curve
prob=  predict(glm.fit, df, type = "response")
pred <- prediction(prob, df$TARGET)    
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf, col=rainbow(7), main="Logistics ROC curve", xlab="1-Specificity", ylab="Sensitivity") 
abline(0,1)

#AUC score
g <- roc(TARGET ~ prob, data = df)
plot(g)  
auc(g) 

#Accuracy
submission$model_prob <- predict(glm.fit, submission, type = "response")
Test <- submission  %>% mutate(model_pred = 1*(model_prob > .53) + 0)%>%mutate(ok= 1*(TARGET) + 0)
Test <- application_test  %>% mutate(model_pred = 1*(model_prob > .53) + 0)
Test <- Test %>% mutate(accurate = 1*(model_pred == ok))
cm<-confusionMatrix(submission$TARGET, new$TARGET)



################################SVM#################################
model <- svm(TARGET~ EXT_SOURCE_3 + CODE_GENDER+ AMT_GOODS_PRICE + AMT_CREDIT+ EXT_SOURCE_1 + DAYS_EMPLOYED + FLAG_OWN_CAR + AMT_ANNUITY+ DAYS_ID_PUBLISH + FLAG_WORK_PHONE+ NAME_FAMILY_STATUS, data = primary_df)
#x <- subset(primary_df, select = -TARGET)
#y <- primary_df$TARGET
summary(model)
#pred <- predict(model, x)
predi <- predict(model, primary_df, type = "response")
summary(predi)
confusion <- table(primary_df$TARGET, predi >= 0.0300)

pred <- fitted(model)
pred <- predict(model, x, decision.values = TRUE)
attr(pred, "decision.values")[1:4,]
plot(cmdscale(dist(primary_df[,-5])),
     col = as.integer(primary_df[,5]),
     pch = c("o","+")[1:150 %in% model$index + 1])
prob =  predict(model, primary_df, type = "response")
pred <- prediction(as.numeric(prob), as.numeric(primary_df$TARGET))    
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf, col=rainbow(7), main="SVM ROC curve", xlab="1-Specificity", 
     ylab="Sensitivity") 
abline(0,1)

###################################Random Forest##################################
rf <- randomForest(TARGET ~ ., data=primary_df)
rf
summary(rf)
pred = predict(rf, newdata=test)
pred
length(test)
cm = table(test$TARGET, pred)
cm
library(ROCR)
prob=  predict(rf, primary_df, type = "response")
pred <- prediction(as.numeric(prob), primary_df$TARGET)    
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf, col=rainbow(7), main="Logistics ROC curve", xlab="1-Specificity", 
     ylab="Sensitivity") 
abline(0,1)
pred<-as.numeric(pred)
g <- roc(TARGET ~ prob, data = primary_df)
plot(g)  
auc(g) 




library(dplyr)
select(SK_ID_CURR, TARGET, CNT_CHILDREN, AMT_INCOME_TOTAL, FLAG_OWN_CAR, DAYS_EMPLOYED,
       AMT_GOODS_PRICE, CODE_GENDER, REGION_POPULATION_RELATIVE, AMT_ANNUITY)
install.packages("rpart")
library(rpart.plot)
rpart.plot(arvore)
arvore <- rpart::rpart(TARGET ~ ., data = primary_df)
summary(arvore)
library(caret)

control <- trainControl(method = 'repeatedcv', 
                        number = 10,
                        repeats = 3,
                        verboseIter = F,
                        savePredictions = T,
                        summaryFunction = twoClassSummary, 
                        classProbs = T,
                        sampling = 'rose')
model_balanceado <- caret::train(TARGET ~ .,
                                 data = primary_df,
                                 method = 'rf',
                                 preProcess = c('scale', 'center'),
                                 trControl = control,
                                 importance = TRUE)
ggplot(model_balanceado$pred[indices, ], aes(m = PAGADOR, d = obs)) + 
  plotROC::geom_roc() +
  coord_equal()

########################Gradient Boosting###########3
model_xgb<-gbm(TARGET~., data=primary_df, distribution="gaussian", n.trees =5000 , interaction.depth =4, cv.folds=3)
summary(model_xgb)
gbm.perf(model_xgb, method = "cv")
gbm.perf(model_xgb, method = "OOB")

prediction <- predict(glm.fit, application_test, type = "response")
summary(prediction)
submission <- cbind(application_test, prediction)
submission<- submission[c(1,92)]
submission$TARGET <- ifelse(submission$prediction>0.0823,1,0)
pred<-predict(object=model_xgb, newdata=test, n.trees=CV, type="response")
summary(pred)
print(pred)
test$TARGET<-cbind(test, pred)
pred<-as.factor(ifelse(pred>0.7,1,0))
test$TARGET<- as.factor(test$TARGET)
confusionMatrix(pred, test$TARGET)

#ROC CURVE
GBM_pred_tes<-prediction(pred, test$TARGET)
GBM_ROC_tes<-prediction(GBM_pred_tes,"tpr","fpr")
plot(GBM_pred_tes)
plot(GBM_pred_tes, add=TRUE, col="green")
legend("right", legend=c("GBM"), col=c("green"), lty=1:2, cex=0.6)
auc.tmp<-performance(GBM_pred_tes, "auc")
GBM_auc_tes<-as.numeric(auc.tmp@y.values)
GBM_auc_tes