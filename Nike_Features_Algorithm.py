# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:31:15 2019

@author: evanm

This code is used to train and test an SVM algorithm to predict injury based on
demographics data and features extracted from the forces in specific regions of 
the foot while walking and running.

To run you will need to ensure that the file paths are pointing to the directory 
where this python script is located and that the 'demographics_data_norm.csv' file
and 'features.xlsx' file are in the same directory. If using a PC and the data 
stored on SFU Vault, you should simply need to change the 'comp' variable to 
match your computer username. (look in file explorer for this)

Contact Evan if there are any questions or concerns.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import decomposition

comp = 'xxx' # insert computer name here

#%% Import demographic data and features
# Import demographic data.
file = open('C:/Users/'+comp+'/sfuvault/Protex II/python_code/demographic_data_norm.csv') #update file path accordingly
Demographics = pd.read_csv(file)
Demographics = Demographics.rename(columns={Demographics.columns[0]: 'PID' })
# Pull out a legend of PID where pressure data and injury data exists
Demographics = Demographics[(Demographics[['press_data','pain_data']] != 0).all(axis=1)].reset_index(drop=True)

#selected demographics
Demographics_data = Demographics[['gender','age','run_xp','half_marathon_completion',
                             'half_marathon_training','prior_weekly_volume',
                             'R_length','L_length','current_size','weight','height','FPI']]

# Import features
file = 'C:/Users/'+comp+'/sfuvault/Protex II/python_code/features.xlsx'
Features = pd.read_excel(file, sheet_name='Features')
Features = Features[Features['PID'].isin(Demographics['PID'])].reset_index(drop=True)

# Create outcome measure (injury vs. no injury)
Injury = np.asarray(Demographics['Injury'])

#combine all possible data
Data = pd.concat([Features.iloc[:,1:],Demographics_data],axis=1)

#%% Manually select data to begin feature reduction
# Here I have removed all the right foot data since the correlation with left 
# foot data was extremely high.

X = Data[['mvwl_1','mvwl_2','mvwl_3','mvwl_4','mvwl_5','mvwl_6','mvwl_7','mvwl_8','mvwl_9','mvwl_10','mvwl_11',
          'mvrl_1','mvrl_2','mvrl_3','mvrl_4','mvrl_5','mvrl_6','mvrl_7','mvrl_8','mvrl_9','mvrl_10','mvrl_11',
          'swl_1','swl_2','swl_3','swl_4','swl_5','swl_6','swl_7','swl_8','swl_9','swl_10','swl_11',
          'srl_1','srl_2','srl_3','srl_4','srl_5','srl_6','srl_7','srl_8','srl_9','srl_10','srl_11',
          'mlwl_1','mlwl_2','mlwl_3','mlwl_4','mlwl_5','mlwl_6','mlwl_7','mlwl_8','mlwl_9','mlwl_10','mlwl_11',
          'mlrl_1','mlrl_2','mlrl_3','mlrl_4','mlrl_5','mlrl_6','mlrl_7','mlrl_8','mlrl_9','mlrl_10','mlrl_11',
          'gender','age',
          'run_xp','half_marathon_completion',
          'half_marathon_training','prior_weekly_volume',
          'R_length','L_length',
          'current_size','weight',
          'height','FPI']]

#%% Identify best features using recursive feature elimination 
# and automatically select best n_feats for training and testing
# Note: n_feats was optimized to 63 for best results

n_feats = 63 #best: 63: F1 = 0.55
estimator = svm.SVC(kernel="linear")
selector = RFE(estimator, n_feats, step=1) 
selector = selector.fit(X, Injury)
Feats = selector.support_ 
Ranks = selector.ranking_

X_reduced = X.iloc[:,np.asarray(np.where(Feats==True))[0,:]]

#%% try Principle Component Analysis
# Note: automatic feature selection was better, so left this out.

#pca = decomposition.PCA(n_components=10)
#pca.fit(X)
#X_reduced = pca.transform(X)
#
##Find correlations
#X_corr = pd.concat([pd.DataFrame(X_reduced),pd.DataFrame(Injury)],axis=1)
#corr = X_corr.corr()

#%% [Method 1] Train / test split using fixed 75% / 25%
# Note: Can only run with either this section or the section below commented out.
# Accuracy is not a good measure for this since ~%80 accuracy can be achieved
# by predicting all participants as 'no injury'. For this reason, F1-score was
# calcualted and used as the primary measure of algorithm performance

# Shuffles data and then splits it into 75% training and 25% testing data
# Keep random_state = # so that shuffling is consistent
X_train, X_test, y_train, y_test = train_test_split(X_reduced, Injury, random_state=0)

# fit and test model
model = svm.SVC(kernel="linear",C=1000)
#model = LogisticRegression(multi_class='multinomial',solver='newton-cg', C=100, tol=0.01)
model.fit(X_train,y_train)

#test and get stats
y_ = model.predict(X_test)
cm = confusion_matrix(y_test, y_)
accuracy = (np.trace(cm))/sum(sum(cm))
f1 = f1_score(y_test,y_)

#print stats
print('Accuracy: ', '%.2f' %(accuracy*100),'%')
print('F1-score: ', '%.2f' %(f1), '/ 1')
print(cm)

#%% [Method 2] Train / test split using Leave One Out Cross Validation
# Note: Can only run with either this section or the section above commented out
# This method will either predict injury or no injury for a particular participant.
# Accuracy is determined by sum of total correct predictions / total samples. 
# In this case, accuracy actually does make sense since we are only looking at one
# participant at a time. With 63 features automatically selected Acc = 66%

#acc_list = []
#f1_list = []
#loo = LeaveOneOut()
#num_correct = 0
#for train_index, test_index in loo.split(X):
#    X_train, X_test = X_reduced.iloc[train_index,:], X_reduced.iloc[test_index,:]
#    y_train, y_test = Injury[train_index], Injury[test_index]
#    # fit and test model
#    model = svm.SVC(kernel='linear', C=1000)
#    model.fit(X_train,y_train)
#    #Test and get stats
#    y_ = model.predict(X_test)
#    if y_ == y_test:
#        num_correct = num_correct +1
#    print(test_index)
#
#accuracy = num_correct / len(X_reduced)
#print('Accuracy: ', '%.2f' %(accuracy*100),'%')
    
#%% Plot feature importance

imp = (model.coef_).T
names = list(X_reduced.columns)
df = pd.concat([pd.DataFrame(names),pd.DataFrame(imp)],axis=1)
df.columns = ['feature','importance']
df = df.sort_values(by='importance').reset_index(drop=True)
colors = ['red' if c < 0 else 'blue' for c in df['importance']]
plt.bar(np.arange(0,len(df)),df['importance'],color=colors)
plt.xticks(np.arange(0,len(df)),df['feature'],rotation=60, ha='right',fontsize=17)
plt.ylabel('Relative Importance',fontsize=20)
plt.title('Relative importance of top 63 features',fontsize=25,pad=30)
plt.grid()
