# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:22:03 2019

@author: evanm
"""

import numpy as np
import pandas as pd
import nike_functions as nf
import os
import matplotlib.pyplot as plt

comp = 'Evan Macdonald'

#%% Open legend file used to catalog all available data (update file as needed)
file = open('C:/Users/'+comp+'/sfuvault/Protex II/python_code/legend_csv.csv')
legend = pd.read_csv(file)

#%% 

#values used for normalization
max_value_walk = 4585 #maximum force value for L and R walking over all participants
max_value_run = 7940 #maximum force value for L and R running over all participants

#declaration of empty arrays to fill with features
#max values
max_value_walk_L = []
max_value_walk_R = []
max_value_run_L = []
max_value_run_R = []
#sum
sum_walk_L = []
sum_walk_R = []
sum_run_L = []
sum_run_R = []
#max location
max_loc_walk_L = []
max_loc_walk_R = []
max_loc_run_L = []
max_loc_run_R = []

for PID in legend['ID']:
    legend_index = legend.index[legend['ID']==PID][0]
    ID = PID[1:]
    date = legend['date'][legend_index]
    trial_walk = str(legend['trial_walk'][legend_index]).zfill(3)
    trial_run = str(legend['trial_run'][legend_index]).zfill(3)
    
    #check walk data first
    exists = os.path.isfile('C:/Users/'+comp+'/sfuvault/Protex II/'+PID+'/'+PID+'_Protex_-_'+trial_walk+'_W_P-'+ID+'_-_'+date+'_-_Pressures_and_Forces.xls')
    if exists:
        rect_prop_PF_walk, PF_data_walk,_,_,_ = nf.import_PandF(ID, comp, date, 'W', trial_walk)
        if (rect_prop_PF_walk['foot'].isin(['Left']).any() and rect_prop_PF_walk['foot'].isin(['Right']).any()):
            #next check running data
            exists = os.path.isfile('C:/Users/'+comp+'/sfuvault/Protex II/'+PID+'/'+PID+'_Protex_-_'+trial_run+'_R_P-'+ID+'_-_'+date+'_-_Pressures_and_Forces.xls')
            if exists:
                rect_prop_PF_run, PF_data_run,_,_,_ = nf.import_PandF(ID, comp, date, 'R', trial_run)
                if (rect_prop_PF_run['foot'].isin(['Left']).any() and rect_prop_PF_run['foot'].isin(['Right']).any()):
                    print(PID)
                    
                    #open up data from each foot and each activity
                    left_index_walk = rect_prop_PF_walk[rect_prop_PF_walk['foot']=='Left'].index[0]
                    PF_L_walk = PF_data_walk[left_index_walk]
                    
                    right_index_walk = rect_prop_PF_walk[rect_prop_PF_walk['foot']=='Right'].index[0]
                    PF_R_walk = PF_data_walk[right_index_walk]
                    
                    left_index_run = rect_prop_PF_run[rect_prop_PF_run['foot']=='Left'].index[0]
                    PF_L_run = PF_data_run[left_index_run]
                    
                    right_index_run = rect_prop_PF_run[rect_prop_PF_run['foot']=='Right'].index[0]
                    PF_R_run = PF_data_run[right_index_run]
                    
                    #Normalize
                    PF_L_walk = PF_L_walk/max_value_walk
                    PF_R_walk = PF_R_walk/max_value_walk
                    PF_L_run = PF_L_run/max_value_run
                    PF_R_run = PF_R_run/max_value_run
                    
                    #Extract features
                    #max values
                    max_value_walk_L.append(np.asarray(np.max(PF_L_walk)))
                    max_value_walk_R.append(np.asarray(np.max(PF_R_walk)))
                    max_value_run_L.append(np.asarray(np.max(PF_L_run)))
                    max_value_run_R.append(np.asarray(np.max(PF_R_run)))
                    #sum
                    sum_walk_L.append(np.asarray(PF_L_walk.sum()))
                    sum_walk_R.append(np.asarray(PF_R_walk.sum()))
                    sum_run_L.append(np.asarray(PF_L_run.sum()))
                    sum_run_R.append(np.asarray(PF_R_run.sum()))
                    #location of max value
                    max_loc_walk_L_temp =[]
                    max_loc_walk_R_temp =[]
                    max_loc_run_L_temp =[]
                    max_loc_run_R_temp =[]
                    for i in range(0,11):
                        max_loc_walk_L_temp.append((PF_L_walk.index[PF_L_walk.iloc[:,i]==np.max(PF_L_walk)[i]][0])/len(PF_L_walk))
                        max_loc_walk_R_temp.append((PF_R_walk.index[PF_R_walk.iloc[:,i]==np.max(PF_R_walk)[i]][0])/len(PF_R_walk))
                        max_loc_run_L_temp.append((PF_L_run.index[PF_L_run.iloc[:,i]==np.max(PF_L_run)[i]][0])/len(PF_L_run))
                        max_loc_run_R_temp.append((PF_R_run.index[PF_R_run.iloc[:,i]==np.max(PF_R_run)[i]][0])/len(PF_R_run))
                    max_loc_walk_L.append(max_loc_walk_L_temp)
                    max_loc_walk_R.append(max_loc_walk_L_temp)
                    max_loc_run_L.append(max_loc_run_L_temp)
                    max_loc_run_R.append(max_loc_run_R_temp)

#%% Resutls dataframe used to copy data out to excel
# select the data you want to xopy out and then copy it from the dataframe into 'features.csv'
results = pd.DataFrame(max_loc_run_R)
