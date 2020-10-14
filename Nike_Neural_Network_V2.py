# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 08:05:49 2019

@author: Evan Macdonald

A Convolutional Nueral Netowrk designed to predict injury events using Dynamic
Maximum Image (DMI), Center of Force Line (COFL), and basic demographic data

To run this code: 
1: Ensure that all file paths are pointing to the correct locations
    - demographics data
    - DMI run/walk combined .txt files
    - COFL run/walk combined .txt files
2: Restart the python kernel you are using so this runs in a fresh instance
3: Run the code to initialize the neural network
4: Enter >>> optimize(n_iterations) where n_instances is any number of iterations.
    - Training accuracy and cost will print every 100 iterations
    - continue to enter >>> optimize(n_iterations) until accuracy and cost are 
    appropriately high / low (usually somewhere around 2000 iterations). 
    - Final training accuracy should be 100% with cost below 0.01
5: Enter >>> check_accuracy(cm=True) to check the accuracy of the algorithm 
    on the test data and get a confusion matrix (leave the brackets empy if you
    do not want to show the confusion matrix)
    
Contact Evan if there are any questions or concerns.
    
"""

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

comp = 'evanm'

#%% Import Data

#Import legend including injury results
#file = open('C:/Users/'+comp+'/sfuvault/Protex II/python_code/Press_Data_Used.csv')
#legend = pd.read_csv(file)
#legend.columns = ['PID','Injury']

# Import demographic data.
file = open('C:/Users/'+comp+'/sfuvault/Protex II/python_code/demographic_data_norm.csv')
Demographics = pd.read_csv(file)
Demographics = Demographics.rename(columns={Demographics.columns[0]: 'PID' })
# Pull out a legend of PID where pressure data and injury data exists
legend = Demographics[(Demographics[['press_data','pain_data']] != 0).all(axis=1)]

#Import all dynamic maximum image data
filename = 'C:/Users/'+comp+'/sfuvault/Protex II/python_code/DMI_images/'+legend['PID'][0]+'.txt'
P1 = np.loadtxt(filename, delimiter=" ")
filename = 'C:/Users/'+comp+'/sfuvault/Protex II/python_code/DMI_images/'+legend['PID'][1]+'.txt'
P2 = np.loadtxt(filename, delimiter=" ")
DMI_data = np.dstack((P1,P2))
for PID in legend['PID'][2:]:
    filename = 'C:/Users/'+comp+'/sfuvault/Protex II/python_code/DMI_images/'+PID+'.txt'
    data = np.loadtxt(filename, delimiter=" ")
    DMI_data = np.dstack((DMI_data,data))
# reshape DMI data
DMI_data = np.transpose(DMI_data,(2,0,1))
DMI_data = np.reshape(DMI_data,(DMI_data.shape[0],DMI_data.shape[1],DMI_data.shape[2],1))

#Import all center of force line image data
filename = 'C:/Users/'+comp+'/sfuvault/Protex II/python_code/COFL_images/'+legend['PID'][0]+'.txt'
P1 = np.loadtxt(filename, delimiter=" ")
filename = 'C:/Users/'+comp+'/sfuvault/Protex II/python_code/COFL_images/'+legend['PID'][1]+'.txt'
P2 = np.loadtxt(filename, delimiter=" ")
COFL_data = np.dstack((P1,P2))
for PID in legend['PID'][2:]:
    filename = 'C:/Users/'+comp+'/sfuvault/Protex II/python_code/COFL_images/'+PID+'.txt'
    data = np.loadtxt(filename, delimiter=" ")
    COFL_data = np.dstack((COFL_data,data))
# reshape COFL data
COFL_data = np.transpose(COFL_data,(2,0,1))
COFL_data = np.reshape(COFL_data,(COFL_data.shape[0],COFL_data.shape[1],COFL_data.shape[2],1))

#create demographic data array
demo_data = legend[['gender','age','run_xp','half_marathon_completion','half_marathon_training','prior_weekly_volume',
                    'R_length','L_length','current_size','weight','height','FPI']]

# Create logit outcome measure (injury vs. no injury)
injury = np.asarray(pd.get_dummies(legend['Injury']), dtype = np.int8)

##Use to check specific image to ensure it imported correctly
#PID = 'P003'
#legend_index = legend.index[legend['PID']==PID][0]
#plt.imshow(DMI_data[legend_index,:,:,0],cmap='gray')

#%% Break into training and testing data
# Shuffles data and then splits it into 75% training and 25% testing data
# Keep random_state = # so that shuffling is the same for both data types
# (i.e. each time will be the same split)
X_train_DMI, X_test_DMI, y_train, y_test = train_test_split(DMI_data, injury, random_state=0)
X_train_COFL, X_test_COFL, _, _ = train_test_split(COFL_data, injury, random_state=0)
X_train_DEMO, X_test_DEMO, _, _ = train_test_split(demo_data, injury, random_state=0)

#%% CNN setup parameters

'''Configuration of parameters (these can be modified without chaning anything else)'''
#DMI Kernel Properties.
#layer 1
c1_filter_size_DMI = 3
c1_num_filters_DMI = 4
c1_pool_size_DMI = 5
#layer 2
c2_filter_size_DMI = 3
c2_num_filters_DMI = 4
c2_pool_size_DMI = 5

#COFL Kernel Properties.
#layer 1
c1_filter_size_COFL = 7
c1_num_filters_COFL = 4
c1_pool_size_COFL = 7
#layer 2
c2_filter_size_COFL = 3
c2_num_filters_COFL = 4
c2_pool_size_COFL = 5

#Fully-connected layer
num_nodes = 80

''' NOTE: modifying anything below here will require some re-work below '''


#DMI Image properties:
MAX_WIDTH_DMI = 64
MAX_HEIGHT_DMI = 100

#COFL Image properties:
MAX_WIDTH_COFL = 196
MAX_HEIGHT_COFL = 630
img_size_flat_COFL = MAX_WIDTH_COFL * MAX_HEIGHT_COFL
img_shape_COFL = (MAX_HEIGHT_COFL, MAX_WIDTH_COFL)


#Number of ouput classes
num_classes = 2
num_channels = 1

#Other parameters
learning_rate = 0.0001

#Helper functions
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

#Layer constructors
def new_conv_layer(x,
                   num_input_channels,
                   filter_size,
                   num_filters,
                   pool_size):
    #Shape of the convolution kernel
    # [buffLen, 20, 1, num_filters]
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    #Create weights
    weights = new_weights(shape=shape)
    #Create biases
    biases = new_biases(length=num_filters)
    #Create convolution layer activations
    activations = tf.nn.conv2d(input=x, filter=weights, strides=[1,1,1,1], padding='SAME')
    #Add in biases
    activations = tf.add(activations, biases)
    #Complete max pooiling
    activations = tf.nn.max_pool(value=activations, 
                                 ksize=[1,pool_size,pool_size,1],
                                 strides=[1,pool_size,pool_size,1],
                                 padding='SAME')
    #Rectified Linear Unit (ReLU)
    # calcs max(x,0) for each input pixel
    # note this is okay to do after max pooling since relu(max_pool(x)) == max_pool(relu(x)) 
    activations = tf.nn.relu(activations)
    #weights are being returned since there may be a use to  look at them later
    return activations, weights

def flatten_layer(layer):
    #get shape of input layer
    # [#, img_height, img_width, 1]
    layer_shape = layer.get_shape()
    #Calculate number of features 
    # =img_height*img_width*1
    num_features = layer_shape[1:4].num_elements()
    #Reshape the layer to flat
    # [#, img_height*img_width*1]
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def new_fc_layer(x,
                 num_inputs,
                 num_outputs):
    #Create new weights and biases
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    #create activations with biases added in and ReLU
    activations = tf.nn.relu(tf.add(tf.matmul(x, weights), biases))
    return activations

def new_logits(x,
               num_inputs,
               num_outputs):
    #Create new weights and biases
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    #create activations with biases added in and ReLU
    activations = tf.matmul(x, weights) + biases
    return activations

def concat_2flat_layers(x1,x2,num_feat1,num_feat2):
    layer_concat = tf.concat([x1,x2],axis=1)
    num_features = num_feat1+num_feat2
    return layer_concat, num_features


'''CNN STRUCTURE'''

'''Input variables'''
#Input 'images'
X_DMI = tf.placeholder(tf.float32, shape=[None, MAX_HEIGHT_DMI, MAX_WIDTH_DMI, 1], name='X_DMI')
#output[#,100,64,1][numsamples, height, width, 1]
X_COFL = tf.placeholder(tf.float32, shape=[None, MAX_HEIGHT_COFL, MAX_WIDTH_COFL, 1], name='X_COFL')
#output[#,630,196,1]
X_DEMO = tf.placeholder(tf.float32, shape=[None, np.shape(demo_data)[1]], name='X_DEMO')
#output[#,n_demographics]

# solutions
Y = tf.placeholder(tf.float32, shape=[None,num_classes], name='Y')
#output[#,2][numsamples,num_classes]
Y_cls = tf.argmax(Y, axis=1)

#DMI CNN layer
'''Create convolution layer #1 - DMI'''
c1_DMI, w1_DMI = new_conv_layer(X_DMI, num_channels, c1_filter_size_DMI, c1_num_filters_DMI, c1_pool_size_DMI)
'''Create convolution layer #2 - DMI'''
c2_DMI, w2_DMI = new_conv_layer(c1_DMI, c1_num_filters_DMI, c2_filter_size_DMI, c2_num_filters_DMI, c2_pool_size_DMI)
'''Create flattened layer - DMI'''
flat_DMI, num_features_DMI = flatten_layer(c2_DMI)

#COFL CNN layer
'''Create convolution layer #1 - COFL'''
c1_COFL, w1_COFL = new_conv_layer(X_COFL, num_channels, c1_filter_size_COFL, c1_num_filters_COFL, c1_pool_size_COFL)
'''Create convolution layer #2 - COFL'''
c2_COFL, w2_COFL = new_conv_layer(c1_COFL, c1_num_filters_COFL, c2_filter_size_COFL, c2_num_filters_COFL, c2_pool_size_COFL)
'''Create flattened layer - COFL'''
flat_COFL, num_features_COFL = flatten_layer(c2_COFL)

#Combined layers
'''Concatendat DMI and COFL layers'''
flat_comb, num_features_comb = concat_2flat_layers(flat_DMI, flat_COFL, num_features_DMI, num_features_COFL)

'''Concatenate flat_comb and X_DEMO layers'''
flat_comb_demo, num_features_comb_demo = concat_2flat_layers(flat_comb, X_DEMO, num_features_comb, np.shape(demo_data)[1])

#Fully connected and output layers
'''Create fully-connected layer'''
f1 = new_fc_layer(flat_comb_demo, num_features_comb_demo, num_nodes)

'''Create fully-connected output layer'''
f_out = new_logits(f1, num_nodes, num_classes)

'''Create softmax output'''
y_ = tf.nn.softmax(f_out)
#Produces index of largest element
y_pred_cls = tf.argmax(y_, axis=1)

'''Cost function and optimizer'''
# This is where the most improvement could come from
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=f_out, labels=Y)
cost = tf.reduce_mean(cross_entropy) #this is the mean of the cross entropy of all the image classifications

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

'''Performance Measures'''
correct_prediction = tf.equal(y_pred_cls, Y_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#%%

'''TensorFlow Session'''

session = tf.Session()
session.run(tf.global_variables_initializer())


# Counter for total number of iterations performed so far.
total_iterations = 0

# Function to optimize algorithm based on the training data
def optimize(num_iterations):
    global total_iterations
    for i in range(total_iterations,
                   total_iterations + num_iterations):
        # Setup the feed_dict with training data
        feed_dict_train = {X_DMI: X_train_DMI, X_COFL: X_train_COFL, X_DEMO: X_train_DEMO, Y: y_train}
        # Run the optimizer using this batch of training data.
        session.run([optimizer, cost], feed_dict=feed_dict_train)
        # Print status every 100 iterations.
        if i % 100 == 0:
            acc,cst = session.run([accuracy,cost], feed_dict=feed_dict_train)
            msg = "Iteration: {0:>6}, Tr. Accuracy: {1:>6.1%}, Cost: {2:>6.2}"
            print(msg.format(i + 1, acc, cst))
    # Update the total number of iterations performed.
    total_iterations += num_iterations
    # Get final stats
    acc,cst = session.run([accuracy,cost], feed_dict=feed_dict_train)

    return acc, cst

def plot_confusion_matrix(cls_pred):
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=np.argmax(y_test,axis=1), y_pred=cls_pred)
    print(cm)
    sn.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
# Function to check the accuracy of the trained algorithm on the test data
def check_accuracy(cm=False):
    num_test = len(X_test_DMI)
    # setup feed_dict to pull data from test data
    feed_dict = {X_DMI: X_test_DMI, X_COFL: X_test_COFL, X_DEMO: X_test_DEMO, Y: y_test}
    # get predictions for test set
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict)
    # Create a boolean array of whether each image is correctly classified.
    correct = (np.argmax(y_test,axis=1) == cls_pred)
    # Calculate the number of correctly classified images.
    correct_sum = correct.sum()
    # Classification accuracy
    acc = float(correct_sum) / num_test
    #F1-score
    f1 = f1_score(y_test,y_)
    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))
    print('F1-score: ',f1)
    
    if cm==True:
        plot_confusion_matrix(cls_pred)
    
    return acc, correct_sum, num_test
