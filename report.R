library(neuralnet)#神经网络
library(ipred) #Bagging
library(tidyverse) #数据预处理
library(factoextra)# 可视化PCA结果
library(tidytext) #分组排序
library(rpart)#决策树
library(randomForest)#随机森林
# library(fastAdaboost)#boosting
library(e1071)#SVM
library(kknn)#kknn
library("skimr")
library("ggplot2")
library(dplyr)
library(broom)
library("data.table")
library("mlr3verse")

set.seed(100) # set seed for reproducibility

# Load data
ploan <- read.csv("bank_personal_loan.csv")
attach(ploan)
head(ploan)
ploan[,9]<-factor(ploan[,9])   

##源数据画图
library("GGally")

ggpairs(ploan |> select(Age,Experience,Income,ZIP.Code,Family,Education,Mortgage,CCAvg),
        aes(color = 'Age'))
##源数据画图 
# Personal.Loan Securities.Account CD.Account Online CreditCard
#数据预处理
# 
ploan_pca = ploan%>%
  mutate(Personal.Loan = ifelse(Personal.Loan==0,
                                     'unloan',
                                     'loan')
  )%>%
  select_if(is.numeric)
PCA = prcomp(ploan_pca, center = T, scale. = T)
PCA
summary(PCA)
screeplot(PCA,type = 'line')
fviz_pca_ind(PCA)
fviz_pca_var(PCA)
fviz_pca_biplot(PCA)

DataExplorer::plot_histogram(ploan, ncol = 3)

###
min_max_scale = function(x){
  (x-min(x))/(max(x)-min(x))
}

ploan = ploan%>%
  mutate(Personal.Loan = as.factor(Personal.Loan))%>%
  mutate_if(.predicate = is.numeric,
            .funs = min_max_scale)%>%
  as.data.frame()

#将分类变量转换成哑元变量
dummy_data = model.matrix(Personal.Loan~.,data = ploan)[,-1]
ploan = data.frame(Personal.Loan = ploan$Personal.Loan,dummy_data)
###
#################################
#simple model fit
#数据集划分
train_id = sample(1:nrow(ploan),0.7*nrow(ploan))
train = ploan[train_id,]
test = ploan[-train_id,]


#模型构建

# logistic regression
logit = glm(Personal.Loan~., family = binomial(link = 'logit'), data = train)
#决策树
tree =rpart(Personal.Loan~.,data = train)
#Bagging
bag_tree = bagging(Personal.Loan~.,data = train)
#random forest
rf = randomForest(Personal.Loan~.,data = train)
#svm
svm_linear = svm(Personal.Loan~.,data = train,kernel = "linear")
svm_polynomial = svm(Personal.Loan~.,data = train,kernel = "polynomial")
svm_radial = svm(Personal.Loan~.,data = train,kernel = "radial")
svm_sigmoid = svm(Personal.Loan~.,data = train,kernel = "sigmoid")
# 
# #knn
knn = kknn(Personal.Loan~.,train = train, test = test)
#朴素贝叶斯
naive_bayes = naiveBayes(Personal.Loan~.,data = train)


#predict
prob_logit = predict(logit, newdata = test,type = 'response')
pre_logit = ifelse(prob_logit>0.5,1,0)
summary(pre_logit)

pre_tree = predict(tree, newdata = test,type = 'class')

pre_bagtree = predict(bag_tree, newdata = test,type = 'class')

pre_rf = predict(rf, newdata = test,type = 'class')
summary(pre_rf)

pre_svm_lin = predict(svm_linear, newdata = test,type = 'class')
pre_svm_poly = predict(svm_polynomial, newdata = test,type = 'class')
pre_svm_rad = predict(svm_radial, newdata = test,type = 'class')
pre_svm_sigm = predict(svm_sigmoid, newdata = test,type = 'class')

pre_knn = knn$fitted.values

pre_naive_bayes = predict(naive_bayes, newdata = test,type = 'class')

#自定义函数 评估标准

binary_class_metric = function(true,predict,positive_level){
  
  accuracy = mean(true==predict)
  precision = sum(true==positive_level & predict==positive_level)/sum(predict==positive_level)
  recall  = sum(true==positive_level&predict==positive_level)/sum(true==positive_level)
  f1_score = 2*precision*recall/(precision+recall)
  
  return(list(accuracy = accuracy,
              precision = precision,
              recall = recall,
              f1_score = f1_score))
  
}

#模型评估

logit_metric = binary_class_metric(true = test$Personal.Loan,
                                   predict = pre_logit,
                                   positive_level = 1)
tree_metric = binary_class_metric(true = test$Personal.Loan,
                                  predict = pre_tree,
                                  positive_level = 1)
bag_tree_metric = binary_class_metric(true = test$Personal.Loan,
                                      predict = pre_bagtree,
                                      positive_level = 1)
rf_metric = binary_class_metric(true = test$Personal.Loan,
                                predict = pre_rf,
                                positive_level = 1)
svm_linear_metric = binary_class_metric(true = test$Personal.Loan,
                                        predict = pre_svm_lin,
                                        positive_level = 1)
svm_polynomial_metric = binary_class_metric(true = test$Personal.Loan,
                                            predict = pre_svm_poly,
                                            positive_level = 1)
svm_radial_metric = binary_class_metric(true = test$Personal.Loan,
                                        predict = pre_svm_rad,
                                        positive_level = 1)
svm_sigmoid_metric = binary_class_metric(true = test$Personal.Loan,
                                         predict = pre_svm_sigm,
                                         positive_level = 1)
knn_metric = binary_class_metric(true = test$Personal.Loan,
                                 predict = pre_knn,
                                 positive_level = 1)
naive_bayes_metric = binary_class_metric(true = test$Personal.Loan,
                                         predict = pre_naive_bayes,
                                         positive_level = 1)

#模型评估

bind_rows(unlist(logit_metric),
          unlist(tree_metric),
          unlist(bag_tree_metric),
          unlist(rf_metric),
          unlist(svm_linear_metric),
          unlist(svm_polynomial_metric),
          unlist(svm_radial_metric),
          unlist(svm_sigmoid_metric),
          unlist(knn_metric),
          unlist(naive_bayes_metric)
          )%>%
  mutate(model = c('logistic regression',
                   'decision tree',
                   'bagging',
                   'random forest',
                   'SVM linear',
                   'SVM Polynomial',
                   'SVM radial',
                   'SVM sigmoid',
                   'knn',
                   'naive bayes'
                   ))%>%
  pivot_longer(cols = -model,
               names_to = 'metric',
               values_to = 'value')%>%
  mutate(model = reorder_within(x = model,by = value,within = metric))%>%
  ggplot(aes(x=model, y = value, fill = metric))+
  geom_col()+
  scale_x_reordered()+
  facet_wrap(~metric,scales = 'free')+
  labs(x = 'Model',
       y = 'Value',
       fill = 'Model')+
  coord_flip()+
  theme_test()+
  theme(legend.position = 'none')

#################################

##################################
#super model
# Define task
loan_task <- TaskClassif$new(id = "Bankloan",
                               backend = ploan,
                               target = "Personal.Loan",
                               positive = '1')

# Cross validation resampling strategy
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(loan_task)

# Define a collection of base learners
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart     <- lrn("classif.rpart", predict_type = "prob")
lrn_cart_cp  <- lrn("classif.rpart", predict_type = "prob", cp = 0.016, id = "cartcp")
lrn_ranger   <- lrn("classif.ranger", predict_type = "prob")
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")
lrn_log_reg  <- lrn("classif.log_reg", predict_type = "prob")

lrn_glmnet <- lrn("classif.glmnet", predict_type = "prob")
lrn_knn <- lrn("classif.kknn", predict_type = "prob")
lrn_naive_bayes <- lrn("classif.naive_bayes", predict_type = "prob")
# Define a super learner
lrnsp_log_reg <- lrn("classif.log_reg", predict_type = "prob", id = "super")

# Missingness imputation pipeline
pl_missing <- po("fixfactors") %>>%
  po("removeconstants") %>>%
  po("imputesample", affect_columns = selector_type(c("ordered", "factor"))) %>>%
  po("imputemean")

# Factors coding pipeline
pl_factor <- po("encode")

# Now define the full pipeline
spr_lrn <- gunion(list(
  # First group of learners requiring no modification to input
  gunion(list(
    po("learner_cv", lrn_baseline),
    po("learner_cv", lrn_cart),
    po("learner_cv", lrn_cart_cp)

  )),
  # Next group of learners requiring special treatment of missingness
  pl_missing %>>%
    gunion(list(
      po("learner_cv", lrn_ranger),
      po("learner_cv", lrn_log_reg),
      po("nop") # This passes through the original features adjusted for
      # missingness to the super learner
    )),
  # Last group needing factor encoding
  pl_factor %>>%
    po("learner_cv", lrn_xgboost),
    po("learner_cv", lrn_glmnet),
    po("learner_cv", lrn_knn),
    po("learner_cv", lrn_naive_bayes)
)) %>>%
  po("featureunion") %>>%
  po(lrnsp_log_reg)

# This plot shows a graph of the learning pipeline
spr_lrn$plot()

# Finally fit the base learners and super learner and evaluate
res_spr <- resample(loan_task, spr_lrn, cv5, store_models = TRUE)

res_spr$aggregate(list(msr("classif.ce"),
                       msr("classif.acc"),
                       msr("classif.fpr"),
                       msr("classif.fnr"),
                       msr("classif.logloss")))
