# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 18:25:43 2018

@author: senthil kumar
"""

import pandas as pd  #data analyze package
import seaborn as sb #data visualization package
import matplotlib.pyplot as plt #data visualization package

data=pd.read_csv("Dataset.csv") # to read the csv file in to variable in python

data.info() # this will give the overall information about the dataset features 
            #(attribute name, size, data type and memory size )
            
# by inspecting the data information, we can understand that the "Species" is the 
# Target of this dataset i.e flower type. There are four features(Septal Length, Septal Width, 
#Petal Length, Petal Width) are given for each flower in the database. So the prime task of
#this work is to develop a Machine Learning system to predict the Species for the given
#four features. 

# ANALYZE, VIZUALIZE and UNDERSTAND DATA before PROCESSING
            
# first ultimate question arise about this dataset is howmany different species(flowers)
#are exist and in each species catagory how many entities are exist. For analysing the
#target(species) column, first we have to know how to access the particular column of data alone.
    
targ=data["Species"] # access the data frame column through its name
targ1=data.iloc[:,4] # accessing the data frame column through its position

#need to check whether any data in target is missing
print(targ.isnull().sum()) # printing total number of null values in the target column
print(data.isnull().sum()) # printing total number of null values in each column of dataset.

# in this dataset provided there is no null data. If the null data is exist , we can drop it
#using data.dropna() or replace the null values using mean or median values [data.fillna(data.mean())]

print(targ.value_counts()) # to count the target catagories. 

sb.countplot(targ) # to visualize the target counts as a plot. In the database given all three catagories
                   # evenly distributed as 50

# DATA NORMALIZATION
feat=data.drop("Species",axis=1) # We need to normalize the input features, so the target variable
                                 # is removed/dropped from the dataset.
# subtracting the mean of each feature column from each of its , will make the feature more discriminative
# and divide it by standanrd deviation makes the feature value be in the uniform range.                                 
norm_feat=(feat-feat.mean())/feat.std()   
                                           

#DATA VISUALIZATION 
#seaborn visualization package is used in this assignment for visualization. Seaborn have more advantages
#than matplotlib

#Violin plot, jointplot and swarmplot are some special features of seaborn package. In order to use 
#of this plot effectively, dataset need to melted (bring all the features in to single column)
# and concatenated with target as below. 
plt_data1 = pd.concat([targ,norm_feat.iloc[:,0:4]],axis=1)
plt_data1 = pd.melt(plt_data1,id_vars="Species",var_name="features",value_name='value')

# violin plot with hue option will give the clear idea about input feature distribution over target class.
#violin plot is plotted with median and percentlie lines which will give the idea about the distribution of feature values.
#This plot reveals that how particular feature values are overlapped for different target class.
# if this overlap is less, that particular feature is more discriminative.
# from seeing the graph we can understand that sepal width feature have more overlap among the thre classes.
#setosa class is clearly discriminative with respect to petal width and length.

plt.figure()
sb.violinplot(x="features", y="value", hue="Species", data=plt_data1,inner="quart")

#box plot is also powerful as violin plot and its give more information about outliers
#setosa petal width and length have more outliers
plt.figure()
sb.boxplot(x='features', y='value',hue = 'Species',data=plt_data1)

# joint plot is an another cool feature of seaborn, which will give the probablilty distribution graph 
#and pearson correlation coefficient vale between features.
sb.jointplot(norm_feat.loc[:,'Sepal.Length'], 
             norm_feat.loc[:,'Sepal.Width'], kind="regg", color="#ce1414")
 # p value in the above graph is -0.12 (- sign indicates negative correlation). The Low p value
#indicates that correlation between Sepal Length and Sepal width is less, which means both are differnt 
#kind of features            

sb.jointplot(norm_feat.loc[:,'Sepal.Length'], 
             norm_feat.loc[:,'Petal.Length'], kind="regg", color="#ce1414")

 # p value in the above graph is 0.9. The Low p value indicates that correlation between Sepal Length 
 #and Sepal width is more, which means both are similar kind of features 
             
plt.figure()
sb.swarmplot(x="features", y="value", hue="Species", data=plt_data1)
# by seeing petal length and Width swarm plot we can understand significance of it in classifying three
#classes, but as both are same kind of feature (can check with joint plot p value), any one can be taken
# for regression out of these two.


# CLASSIFICATION USING LOGISTIC REGRESSION

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score,confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

 #target attribute of this data set (flower type) is catagorical data. This need to be converted into 
 #non-catagorical(numeric) data. LabelEncoder can do this. OneHot encoder also one another popular way of doing it.
LE=LabelEncoder()
targ_value=LE.fit_transform(targ)

# For the development of any ML system, Data need to be split in to 3 gatogories: Training data, Validating
#data and Testing Data. As the dataset size for this given database is less, I splitted the data for 
#training and testing alone( 80 percent for training, 20 % for testing)
Xtrain, Xtest , Ytrain, Ytest = train_test_split(norm_feat,targ_value,test_size=0.2)
                                                
lreg=LogisticRegression()  #syntax for logistic regression object definition
lreg.fit(Xtrain,Ytrain)  # fit the train data in to logistic regression model
res=lreg.predict(Xtest)   # predict the fitted model with the test data
acc=accuracy_score(Ytest,res) # compute the Accuracy
print(acc)

#k fold is another statergy to divide the data for training and testing. In this data is divided in to k sections
#In this assignment k is taken as 5, which means 150 samples are divided in to 5 groups (30 samples each)
# out of five groups one group used for testing while other 4 used for training. This process iteratively happen for 
# five times such that all the group used for testing at once. The relaiability of the result is more in folding method.
model=LogisticRegression()
kfold=KFold(n_splits=5,shuffle=True)
result=cross_val_score(model,Xtrain,Ytrain,cv=kfold,scoring='accuracy')
print(result.mean())

#FEATURE SELECTION

#Pearson Coefficient correlation based feature selection method is used for this assignemnt. 
#The following heatmap gives the p value of each feature Vs other feature.
#The p value between petal length and petal width is 1, which means both these features are simillar,
#so we can drop anyone. Also sepal length and pedal length correlation is high(0.9). 
#By comparing both these scenarios, petal length and septal length  features are dropped as these features are redundant.

plt.subplots()
sb.heatmap(norm_feat.corr(),annot=True,linewidths=.5,fmt= '.1f')
plt.yticks(rotation=0)

sel_feat=norm_feat.drop(["Sepal.Length","Petal.Length"],axis=1)

#train and test data split for reduced feature dataset
XtrainR, XtestR , YtrainR, YtestR = train_test_split(sel_feat,targ_value,test_size=0.2)

#logical regression model for reduced feature dataset
lregR=LogisticRegression()
lregR.fit(XtrainR,YtrainR)
resR=lregR.predict(XtestR)
accR=accuracy_score(YtestR,resR)
print(accR)

# The accuracy is improved from 83% to 90% after removing the redundant feature

#Parameter Tuning

#performance of logical regression will vary based on C value. using Gridsearch method differnt C values
# are tried on logistic regression model and the best parameter was tuned. 
#the best estimator function prints the C value as 100.

# with parameter tuning, the Accuracy is improved to 93.3%

from sklearn.model_selection import GridSearchCV
param_grid1 = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

clf = GridSearchCV(LogisticRegression(), param_grid1)
clf.fit(XtrainR, YtrainR)
clf_pred=clf.predict(XtestR)

clf_acc=accuracy_score(YtestR,clf_pred)
print(clf_acc)
print (clf.best_estimator_)


# kfold statergy is applied with logistic regression of tuned value C=100. 
#This produced a accuracy of 94.2%, which is 4% more than the kfold statergy used intially
#(before feature reduction and tuning) in the coding line 129

modelT=LogisticRegression(C=100)
kfoldT=KFold(n_splits=5,shuffle=True)
resultT=cross_val_score(modelT,XtrainR,YtrainR,cv=kfoldT,scoring='accuracy')
print(resultT.mean())

# appart from accuracy, someother logistic regression output measures are computed
#as follows:

cm= confusion_matrix(YtestR,clf_pred) # which will give TP,TN, FP and FN
print(cm)
plt.figure()
sb.heatmap(cm,annot=True,fmt="d")

R2=r2_score(YtestR,clf_pred) # Generally if this value is on higher side, then it 
                             #means the model fitted the data well.
print(R2)

from sklearn.metrics import classification_report
print(classification_report(YtestR, clf_pred)) # which will give Precision, Recall and F1 score


# Summary of Work done

#1. Data normalized and checked for any null values
#2. Different visualizations are performed to understand the data
#3. Catagorized target data converted in to non catagorical
#4. Data Splitted for Training and Testing (Validation is not considered)
#5. Logistic Regression model applied and accuracy computed
#6. Feature reduction carried out through pearson correlation coefficient
#7. Logistic Regression applied on reduced feature set and the accuracy has been improved
#8. Hyper Parameter tuning performed for logistic regression
#9. with the tuned parameter,  logistic regression model accuracy was further incresed
#10. k-fold statergy is applied  with tuned C value
#11. Other regression measures such as F1, R2, Precision, Recall, Confusion Matrix are computed.













            

             

            

            
            

