# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:41:06 2019

@author: evanm

Functions used to import the raw data for the Protex II study
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, freqz

# function used to import data from the Center of Force Line data format .xls files. 
# file_path variable can be modified if file structure isn't the exact same on your particular computer.
# comp is the name of your computer
# best way to check is to copy the file you are trying to open and then paste it as a new line
# then change the structure of file_path below. 
# If different, this will need ot be modified for each import function below.
def import_COFL(ID, comp, date, activity, trial):
    skip_rows = [0,1,2]
    titles = ['Frame', 'ms', 'X', 'Y', 'Force']
    #Input file - modify structure as needed, but keep comp, ID, trial and activity in it.
    file_path = open('C:/Users/'+comp+'/sfuvault/Protex II/P'+ID+'/P'+ID+'_Protex_-_'+trial+'_'+activity+'_P-'+ID+'_-_'+date+'_-_Center_of_Force_line.xls')
    data = pd.read_csv(file_path, header=None, skiprows=skip_rows, skipinitialspace=True, sep='\s+')
    data = data.drop(data.columns[[5,6,7,8,9,10,11,12,13]], axis=1) 
    data.columns = titles
    
    #find number of trials
    if len(data[data['Frame']=='Left'].index) == 0:
        num_left = 0
    else:
        num_left = int(data['Frame'].value_counts()['Left'] / 2)
    if len(data[data['Frame']=='Right'].index) == 0:
        num_right = 0
    else:
        num_right = int(data['Frame'].value_counts()['Right'] / 2)
    
    #extract rectangle properties
    num_index = data.index[data['Frame']=='Rectangle']
    num_index = num_index + 1 #index of where the rectangle dimensions are (come in multiples of 4)
    
    rect_prop_lst = []
    i=0
    for a in range(0,(num_left+num_right)):
        temp_lst = []
        temp_lst.append(data['Frame'].iloc[num_index[i]-2])
        temp_lst.append(data['Frame'].iloc[num_index[i]])
        temp_lst.append(data['Frame'].iloc[num_index[i+1]])
        temp_lst.append(data['Frame'].iloc[num_index[i+2]])
        temp_lst.append(data['Frame'].iloc[num_index[i+3]])
        i = i+4
        rect_prop_lst.append(temp_lst)
        
    rect_prop = pd.DataFrame(rect_prop_lst, columns=['foot','bottom','left','width','height'])
    
    #pull out values and put them into a list of dataframes
    values = []
    i=0
    for a in range(0,(num_left+num_right-1)):
        vals_tmp = data.iloc[(num_index[i]+8):(num_index[i+4]-3)].astype(float)
        values.append(vals_tmp)
        i = i+4
    
    vals_tmp = data.iloc[(num_index[i]+8):].astype(float)
    values.append(vals_tmp)
    
    return rect_prop, values

# function used to import data from the Dynamic Maximum Image data format .xls files.
def import_DMI(ID, comp, date, activity, trial):
    #Input file - modify structure as needed, but keep comp, ID, trial and activity in it.
    file_path = 'C:/Users/'+comp+'/sfuvault/Protex II/P'+ID+'/P'+ID+'_Protex_-_'+trial+'_'+activity+'_P-'+ID+'_-_'+date+'_-_Dynamic_Maximum_Image.xls'
    
    # Find width of largest row in data
    largest_column_count = 0
    
    with open(file_path, 'r') as temp_f:
        lines = temp_f.readlines()
        for l in lines:
            column_count = len(l.split('\t')) + 1
            largest_column_count = column_count if largest_column_count < column_count else largest_column_count
    temp_f.close()
    
    # Generate column names (will be 0, 1, 2, ..., largest_column_count - 1)
    column_names = [i for i in range(0, largest_column_count)]
    
    # Read file using width of widest row
    data = pd.read_csv(file_path, header=None, sep='\s+', names=column_names, skipinitialspace=True)
    
    #det'n number of right and left samples
#    num_left = int(data[0].value_counts()['Left'])
#    num_right = int(data[0].value_counts()['Right'])
    
    #find number of trials
    if len(data[data[0]=='Left'].index) == 0:
        num_left = 0
    else:
        num_left = int(data[0].value_counts()['Left'])
    if len(data[data[0]=='Right'].index) == 0:
        num_right = 0
    else:
        num_right = int(data[0].value_counts()['Right'])
    
    #extract rectangle properties
    num_index = data.index[data[0]=='Rectangle']
    num_index = num_index + 1 #index of where the rectangle dimensions are (come in multiples of 4)
    
    rect_prop_lst = []
    i=0
    for a in range(0,(num_left+num_right)):
        temp_lst = []
        temp_lst.append(data[0].iloc[num_index[i]-2])
        temp_lst.append(data[0].iloc[num_index[i]])
        temp_lst.append(data[0].iloc[num_index[i+1]])
        temp_lst.append(data[0].iloc[num_index[i+2]])
        temp_lst.append(data[0].iloc[num_index[i+3]])
        i = i+4
        rect_prop_lst.append(temp_lst)
        
    rect_prop = pd.DataFrame(rect_prop_lst, columns=['foot','bottom','left','width','height'])
    
    #pull out values for each sample and put them into a list of dataframes
    values = []
    i=0
    for a in range(0,(num_left+num_right-1)):
        vals_tmp = data.iloc[(num_index[i]+7):(num_index[i+4]-3)].dropna(axis='columns').astype(float)
        values.append(vals_tmp)
        i = i+4
    
    vals_tmp = data.iloc[(num_index[i]+7):].dropna(axis='columns').astype(float)
    values.append(vals_tmp)
    
    return rect_prop, values

# function used to import data from the Dynamic Roll Off data format .xls files.
def import_DRO(ID, comp, date, activity, trial):
    #Input file - modify structure as needed, but keep comp, ID, trial and activity in it.
    file_path = 'C:/Users/'+comp+'/sfuvault/Protex II/P'+ID+'/P'+ID+'_Protex_-_'+trial+'_'+activity+'_P-'+ID+'_-_'+date+'_-_Dynamic_Roll_Off.xls'
    
    # Find width of largest row in data
    largest_column_count = 0
    
    with open(file_path, 'r') as temp_f:
        lines = temp_f.readlines()
        for l in lines:
            column_count = len(l.split('\t')) + 1
            largest_column_count = column_count if largest_column_count < column_count else largest_column_count
    temp_f.close()
    
    # Generate column names (will be 0, 1, 2, ..., largest_column_count - 1)
    column_names = [i for i in range(0, largest_column_count)]
    
    # Read file using width of widest row
    data = pd.read_csv(file_path, header=None, sep='\s+', names=column_names, skipinitialspace=True)
    
    #det'n number of right and left samples
    num_left = int(data[0].value_counts()['Left']/2)
    num_right = int(data[0].value_counts()['Right']/2)
    
    #extract rectangle properties
    num_index = data.index[data[0]=='Rectangle']
    num_index = num_index + 1 #index of where the rectangle dimensions are (come in multiples of 4)
    
    rect_prop_lst = []
    i=0
    for a in range(0,(num_left+num_right)):
        temp_lst = []
        temp_lst.append(data[0].iloc[num_index[i]-2])
        temp_lst.append(data[0].iloc[num_index[i]])
        temp_lst.append(data[0].iloc[num_index[i+1]])
        temp_lst.append(data[0].iloc[num_index[i+2]])
        temp_lst.append(data[0].iloc[num_index[i+3]])
        i = i+4
        rect_prop_lst.append(temp_lst)
        
    rect_prop = pd.DataFrame(rect_prop_lst, columns=['foot','bottom','left','width','height'])
    
    #pull out values for each sample and put them into a list of dataframes
    values = []
    i=0
    for a in range(0,(num_left+num_right-1)):
        vals_tmp = data.iloc[(num_index[i]+7):(num_index[i+4]-3)].dropna(axis='columns', how='all')
        values.append(vals_tmp)
        i = i+4
    
    vals_tmp = data.iloc[(num_index[i]+7):].dropna(axis='columns', how='all')
    values.append(vals_tmp)
    
    return rect_prop, values

# function used to import data from the Pressure and Forces data format .xls files.
def import_PandF(ID, comp, date, activity, trial):
    #Input file - modify structure as needed, but keep comp, ID, trial and activity in it.
    file_path = 'C:/Users/'+comp+'/sfuvault/Protex II/P'+ID+'/P'+ID+'_Protex_-_'+trial+'_'+activity+'_P-'+ID+'_-_'+date+'_-_Pressures_and_Forces.xls'
    
    # Find width of largest row in data
    largest_column_count = 0
    
    with open(file_path, 'r') as temp_f:
        lines = temp_f.readlines()
        for l in lines:
            column_count = len(l.split('\t')) + 1
            largest_column_count = column_count if largest_column_count < column_count else largest_column_count
    temp_f.close()
    
    # Generate column names (will be 0, 1, 2, ..., largest_column_count - 1)
    column_names = [i for i in range(0, largest_column_count)]
    
    # Read file using width of widest row
    data = pd.read_csv(file_path, header=None, sep='\s+', names=column_names, skipinitialspace=True)
    
    #find number of trials
    if len(data[data[0]=='Left'].index) == 0:
        num_left = 0
    else:
        num_left = int(data[0].value_counts()['Left'] / 2)
    if len(data[data[0]=='Right'].index) == 0:
        num_right = 0
    else:
        num_right = int(data[0].value_counts()['Right'] / 2)
    
#    #det'n number of right and left samples
#    num_left = int(data[0].value_counts()['Left']/2)
#    num_right = int(data[0].value_counts()['Right']/2)
    
    #extract rectangle properties
    num_index = data.index[data[0]=='Rectangle']
    num_index = num_index + 1 #index of where the rectangle dimensions are (come in multiples of 4)
    
    rect_prop_lst = []
    i=0
    for a in range(0,(num_left+num_right)):
        temp_lst = []
        temp_lst.append(data[0].iloc[num_index[i]-2])
        temp_lst.append(data[0].iloc[num_index[i]])
        temp_lst.append(data[0].iloc[num_index[i+1]])
        temp_lst.append(data[0].iloc[num_index[i+2]])
        temp_lst.append(data[0].iloc[num_index[i+3]])
        i = i+4
        rect_prop_lst.append(temp_lst)
        
    rect_prop = pd.DataFrame(rect_prop_lst, columns=['foot','bottom','left','width','height'])
    
    #extract force zones
    force_labels = ['toe_1','toe_2-5','meta_1','meta_2','meta_3','meta_4','meta_5','midfoot','heel_medial','heel_lateral','sum','calibration_factor']
    force_zones = []
    i=0
    for a in range(0,(num_left+num_right)-1):
        vals_tmp = data.iloc[(num_index[i]+9):(num_index[i+4]-3)].reset_index(drop=True)
        vals_tmp = vals_tmp[:vals_tmp[vals_tmp[0] == 'Pressure'].index[0]].dropna(axis='columns', how='all').astype(float)
        vals_tmp.columns = force_labels
        force_zones.append(vals_tmp)
        i = i+4
    
    vals_tmp = data.iloc[(num_index[i]+9):].reset_index(drop=True)
    vals_tmp = vals_tmp[:vals_tmp[vals_tmp[0] == 'Pressure'].index[0]].dropna(axis='columns', how='all').astype(float)
    vals_tmp.columns = force_labels
    force_zones.append(vals_tmp)
    
    #extract pressure zones
    pressure_labels = ['toe_1','toe_2-5','meta_1','meta_2','meta_3','meta_4','meta_5','midfoot','heel_medial','heel_lateral','calibration_factor']
    pressure_zones = []
    i=0
    for a in range(0,(num_left+num_right)-1):
        vals_tmp = data.iloc[(num_index[i]+9):(num_index[i+4]-3)].reset_index(drop=True)
        start = vals_tmp.index[vals_tmp[0]=='Pressure'][0]+2
        end = vals_tmp.index[vals_tmp[0]=='Force'][0]
        vals_tmp = vals_tmp[start:end].dropna(axis='columns', how='all').astype(float)
        vals_tmp.columns = pressure_labels
        pressure_zones.append(vals_tmp)
        i = i+4
    
    vals_tmp = data.iloc[(num_index[i]+9):].reset_index(drop=True)
    start = vals_tmp.index[vals_tmp[0]=='Pressure'][0]+2
    end = vals_tmp.index[vals_tmp[0]=='Force'][0]
    vals_tmp = vals_tmp[start:end].dropna(axis='columns', how='all').astype(float)
    vals_tmp.columns = pressure_labels
    pressure_zones.append(vals_tmp)
    
    #extract force cursors
    force_cursors = []
    i=0
    for a in range(0,(num_left+num_right)-1):
        vals_tmp = data.iloc[(num_index[i]+9):(num_index[i+4]-3)].reset_index(drop=True)
        start = vals_tmp.index[vals_tmp[0]=='Force'][0]+2
        end = vals_tmp.index[vals_tmp[0]=='Pressure'][1]
        vals_tmp = vals_tmp[start:end].dropna(axis='columns', how='all').astype(float)
        vals_tmp.columns = force_labels
        force_cursors.append(vals_tmp)
        i = i+4
    
    vals_tmp = data.iloc[(num_index[i]+9):].reset_index(drop=True)
    start = vals_tmp.index[vals_tmp[0]=='Force'][0]+2
    end = vals_tmp.index[vals_tmp[0]=='Pressure'][1]
    vals_tmp = vals_tmp[start:end].dropna(axis='columns', how='all').astype(float)
    vals_tmp.columns = force_labels
    force_cursors.append(vals_tmp)
    
    #extract pressure cursors
    pressure_cursors = []
    i=0
    for a in range(0,(num_left+num_right)-1):
        vals_tmp = data.iloc[(num_index[i]+9):(num_index[i+4]-3)].reset_index(drop=True)
        start = vals_tmp.index[vals_tmp[0]=='Pressure'][1]+2
        end = len(vals_tmp)
        vals_tmp = vals_tmp[start:end].dropna(axis='columns', how='all').astype(float)
        vals_tmp.columns = pressure_labels
        pressure_cursors.append(vals_tmp)
        i = i+4
    
    vals_tmp = data.iloc[(num_index[i]+9):].reset_index(drop=True)
    start = vals_tmp.index[vals_tmp[0]=='Pressure'][1]+2
    end = len(vals_tmp)
    vals_tmp = vals_tmp[start:end].dropna(axis='columns', how='all').astype(float)
    vals_tmp.columns = pressure_labels
    pressure_cursors.append(vals_tmp)
    
    return rect_prop, force_zones, pressure_zones, force_cursors, pressure_cursors

''' 
Function used to import data from the IMU data format .txt files.
Files must be created by exporting from the Xsens MT Manager
To do so, follow these steps:
1. Open file you want to export in MT Manager
2. Under Tools > Preferences select Exporters > ASCII Exporter
3. Select Tab as the delimiter option
4. In the Exported Data section check the following boxes:
    - Position
    - Velocity
    - Orientation Increment
    - Velocity Increment
    - Rate of Turn
    - Acceleration
    - Free Acceleration
    - Magnetic Field
    - Packet Counter
    - Sample Time Fine
    Make sure all other options are unselected
    You should only have to do this once, however it is worth checking periodically
5. Select File > Export and choose the appropriate participant folder
6. Go into the folder and re-name the file to match the original file (ex. 001_R_I-191)
''' 
def import_IMU(ID, comp, activity, trial):
    #Input file - modify structure as needed, but keep comp, ID, trial and activity in it.
    file_path = 'C:/Users/'+comp+'/sfuvault/Protex II/P'+ID+'/'+trial+'_'+activity+'_I-'+ID+'.txt'
    data = pd.read_csv(file_path, header=0, skiprows=4, skipinitialspace=True, sep='\t')
    
    return data

# low pass filter for accelerometer data
def low_pass_filter(X,order,cutoff):
    # order = order of filter
    fs = 100  # sample rate, Hz
    #cutoff = desired cutoff frequency of the filter, Hz
    
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    y = lfilter(b, a, X)
    return y
    
#input must be dataframe with three columns labeled 'Acc_X', 'Acc_Y' and 'Acc_Z'
# Returns dataframe with one column titled 'acc' which is the vectorized acceleration
def process_acc(X,order,cutoff):
    header = np.asarray(X[:][0:cutoff])
    X['Acc_X'] = low_pass_filter(X['Acc_X'],order,cutoff)
    X['Acc_Y'] = low_pass_filter(X['Acc_Y'],order,cutoff)
    X['Acc_Z'] = low_pass_filter(X['Acc_Z'],order,cutoff)
    X[:][0:cutoff] = header
    acc = np.sqrt((X['Acc_X']**2)+(X['Acc_Y']**2)+(X['Acc_Z']**2))
    X['acc'] = acc
    X = X.drop(['Acc_X', 'Acc_Y', 'Acc_Z'],axis=1)
    return X