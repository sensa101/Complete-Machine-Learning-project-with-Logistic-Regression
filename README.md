# Completed-Machine-Learning-project-with-Logistic-Regression
Complete approach to go with for a ML model

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
#12. Save the final trianed model for future prediction process using pickle package

#The initial accuracy with 80-20 percentage spilt    		          = 0.966666666667
#Initial accuracy with 5 fold method                           		= 0.891666666667
#Accuracy after feature reduction                              		= 0.966666666667
#Accuracy after parameter tuning (for the same 80-20 split)       = 1.0
#Accuracy with 5 fold (with tuned parameter)			                =0.94
#These values will change if the train_test_split is carried out without random_state argument. That is #why in this work, I also computed #the accuracy through Kfold method to justify the reliability of this #work.

