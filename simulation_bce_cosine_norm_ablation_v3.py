
# logic changed - first 3 epochs - use all indexes - V is updated slowly, also instead of using previous V in the next cosines, we use
# V calculated in that epoch itself to aggregate weights -> 
# ABLATION WITH NORM AND COSINE
# use norn - pick 10 indexes - aggregate them for V, Quick V
'''
major bug fixes from v6:
1. in calculate cosine we had not divided by 10000000 and by number of samples
2. After epoch 3 we were aggregating weights of first 2 users ony because of bug on line 784-786 (len(picked_indexes was being used))
'''
from cProfile import label
from email import header
import json
from marshal import load
from operator import mod
import pickle
from pyexpat import model
from pyexpat.errors import XML_ERROR_DUPLICATE_ATTRIBUTE
from re import A
from cv2 import mean, norm
#from matplotlib.font_manager import _Weight
from matplotlib.lines import Line2D
import numpy as np
#import user1 as am1 # file for calling functions of first node
from collections import Counter
from collections import defaultdict
from random import random
import random
from collections import defaultdict
#from IPython import display
from PIL import Image
#from scipy import rand
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as F 
from torchvision import models
import json
import itertools
import torch
import torch.nn as nn
import torch.optim as optim 
import matplotlib.ticker as ticker 
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pt
import matplotlib as plt
from glob import glob
import cv2                
import matplotlib.pyplot as plt  
import json
import math
import pandas as pd
from sklearn.metrics import accuracy_score
from web3 import Web3 
import json
#from web3 import web3practice
from web3 import Web3
import datetime
from sklearn.metrics import confusion_matrix
import random as rand
import copy
'''
CREATED CLASSES FOR THE USER
CONTAINS SIMULATION
NO ROLL OVER
'''

##### GLOBAL VARIABLES ##########
random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
num1 = 1024
num2 = 1
num3 = num1*num2
nums = []
f = open("accounts.txt", "a")

weightsBlock = [0]*num2
for k in range(num2):
    weightsBlock[k] = [0]*num1
print ("----- weights block size ------------")
print (len(weightsBlock[0]))
biasBlock = [0] * num2
addWeightsBlock = [0]*num2
for k in range(num2):
    addWeightsBlock[k] = [0]*num1

addBiasBlock = [0] * num2
totalSamplesBlock = 0

num_users = 16
cosines = [0] * num_users # 16 is the number of users in the system
norms = [0] * num_users
##### Variables for Cosine Similarity
for m in range (num_users):
    cosines[m]= [-2]*2 
for m in range (num_users):
    norms[m]= [0]*2 
Voi = copy.deepcopy(weightsBlock)
Woi = copy.deepcopy(weightsBlock)#
print ("---- Initial value of weights block-------", weightsBlock[0][:5])
# DEFINE THE MODEL
class cancerClassification(nn.Module):
    def __init__(self, input_size, num_classes):
        super(cancerClassification, self).__init__()
        self.l1 = nn.Sequential(
            #nn.LayerNorm(input_size),
            nn.Linear(input_size,num_classes)
        )
    def forward(self, x):
        out = self.l1(x)
        #out = torch.sigmoid(out) # We require this if we use BCE LOSS # remove this in case of cross entropy loss
        out = torch.sigmoid(out)
        return out

model1 = cancerClassification(1024, 1)
criterion = nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model1.parameters(), weight_decay = 0.05, lr = 0.005)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
#n_epochs = 2 # need to be changed for each rolling case

print ("--------------- Hospital model -----------------")
features = np.load("histoptest_features.npy")
labels = np.load("histoptest_labels.npy")
#features = np.load("COVIDX_test_features.npy")
#labels = np.load("COVIDX_test_labels.npy")
print (features.shape)
X_test = features
Y_test = labels
print (Y_test[:10])
print (X_test.shape)
sh = Y_test.shape[0]
print (Y_test.shape)
Y_test = Y_test.reshape(Y_test.shape[0],1)

# DATASET 

class cancerDataset(Dataset):
    def __init__(self,X,Y):
        self.n_samples = X.shape[0]
        self.X = (torch.from_numpy(X))
        self.Y = (torch.from_numpy(Y))
    def __getitem__(self,index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.n_samples

dataset_test = cancerDataset(X_test, Y_test)

# Recheck this function
def testing():
    #CODE FOR LAST TESTING
    weights = []  
    for i in range (num2):
        wts = copy.deepcopy(weightsBlock[i])
        wts = [float(item/100000000) for item in wts]
        weights.append(wts)
    biasa = copy.deepcopy(biasBlock)
    bias = [float(item/100000000) for item in biasa]
    weights = torch.tensor(weights)
    bias = torch.tensor(bias)
    with torch.no_grad():
        model1.l1[0].weight.copy_(weights.float())
        model1.l1[0].bias.copy_(bias.float())
    test_performance()

def pred(x, y):  
        comp = torch.zeros((1,x.shape[0]))
        predict = model1(x.float())  
        pred = (predict.data>0.5).float() 
        loss = criterion(predict,y.float())
        return (pred)

def test_performance():
    predicted = pred(dataset_test[:][0], dataset_test[:][1])
    pred1 = torch.reshape(predicted,(-1,))
    cm = confusion_matrix(Y_test,pred1)
    count = 0
    for i in range (sh):
        if Y_test[i] == int(pred1[i]):
            count +=1 
    total_images = Y_test.shape[0]
    accuracy = (count/total_images)*100
    f = open("result_ablation_norm_cosine.csv", "a")
    f.write(str(accuracy)+"\n")
    f.close()
    print ("Testing accuracy", count/total_images)

test_performance()
for param in model1.parameters():
    param.data = (param.data)* 100000000

wt2 = model1.l1[0].weight
bias = model1.l1[0].bias
print (type(list(wt2)))

print ("Start Time ",datetime.datetime.now())
for i in range (num2):
    for j in tqdm(range (num1)):
        val = int(wt2[i][j])
        weightsBlock[i][j] = val

for i in tqdm(range(num2)):
    val = int(bias[i])
    biasBlock[i] = val
features_all = np.load('histoptrain_features.npy')
labels_all = np.load('histoptrain_labels.npy')
num = 100
##### CLASS VARIABLES ##########
class Users:
    def __init__(self, accoun_index) -> None:
        self.ac = accoun_index
        #self.filenameFeatures = 'COVIDX_train_features_100_' + str(self.ac) + '.npy'
        #self.filenameLabels = 'COVIDX_train_labels_100_' + str(self.ac) + '.npy'
        self.index = (self.ac)*num
        self.features = features_all[(self.ac)*num:(num)*(self.ac+1)]
        self.labels = labels_all[(self.ac)*num:(num)*(self.ac+1)]
        self.X_df = self.features[:95]
        self.Y_df = self.labels[:95]
        self.X_df_test = self.features[95:]
        self.Y_df_test = self.labels[95:]
        self.dataset_test_user = cancerDataset(self.X_df_test, self.Y_df_test)
        self.device =''
        if torch.cuda.is_available() :
            self.device = torch.device("cuda")
            print("cuda") 
        else:
            self.device = torch.device("cpu")
        self.n_inputs = self.X_df.shape[1]
        self.n_outputs = 1
        self.n_samples = self.X_df.shape[0]
        self.model = cancerClassification(self.n_inputs, self.n_outputs) # --------  Apply Sanity check here -- see play function
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), weight_decay = 0.05, lr = 0.005) #0.0005
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.9)#
        self.dataset = cancerDataset(self.X_df, self.Y_df)
        self.dataloader_train = DataLoader(dataset = self.dataset, batch_size = 16, shuffle = True)
        self.dataiter = iter(self.dataloader_train)
        self.total_loss = []
        self.average_loss = 0
        self.training_samples = len(self.X_df)
        self.num1 = 1024
        self.num2 = 1
        self.num3 = self.num1*self.num2
        
    
    def train(self,epochs):
        self.average_loss = 0
        for i in range(0,epochs):
            self.model.train()
            for b_index, (feat, lab) in enumerate(self.dataloader_train):
                outputs = self.model(feat)
                loss = self.criterion(outputs.view(-1),lab.float())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.total_loss.append(loss)
                #if(b_index)%2 == 0:
                print (f'epoch {i+1}/{epochs} Batch {b_index+1} loss {loss}')
            
            self.average_loss = sum(self.total_loss)/len(self.total_loss)               
            self.scheduler.step()
        print ("Training done")

    def sanity_check(self):
        print ("---- Model Weights -----")
        print (self.model.l1[0].weight)
    
    def play(self,epochs):
        
        weights = []
        for i in range (self.num2):
            wts = copy.deepcopy(weightsBlock[i])
            wts = [float(item/100000000) for item in wts]
            weights.append(wts)
        bias = copy.deepcopy(biasBlock)
        bias = [float(item/100000000) for item in bias]
        # Weights before training #
        weights = torch.tensor(weights)
        bias = torch.tensor(bias)
        with torch.no_grad():
            self.model.l1[0].weight.copy_(weights.float())
            self.model.l1[0].bias.copy_(bias.float())
        # ADD TIME HERE
        print ("---------- Training Start Time ----------------")
        print (datetime.datetime.now())
        self.train(epochs)
        print ("---------- Training End Time ----------------")
        print (datetime.datetime.now())
        print ("------- Sanity check ------ Weights after training---")
        self.sanity_check()
        # FOR WEIGHT POISONING
        '''
         if self.ac == 10:
            print ("------ Before poisoning ----------")
            print (self.model.parameters)
            for param in self.model.parameters():
                print (param.data)
                param.data = param.data * (-10)
                print ("------ After poisoning ----------")
                print (param.data)
            print ("poisoned weights")
        '''
    
    def prepare_for_blockchain(self):
        # Prepare weights and bias for blockchain. 
        # Segregating this function from add to blockchain function because, later is called twice for same user.
        #------------------------ Prepare the weights to store in blockchain ------------------#
        for param in self.model.parameters():
            param.data = (param.data)* 100000000 * self.n_samples
        # weighst after training #
        wts2 = self.model.l1[0].weight
        wbias = self.model.l1[0].bias

    def add_to_blockchain(self):

        wts2 = self.model.l1[0].weight
        wbias = self.model.l1[0].bias
        for i in range (self.num2):
            for j in tqdm(range (self.num1)):
                val = int(wts2[i][j])
                addWeightsBlock[i][j] = addWeightsBlock[i][j] + val
                
        for i in tqdm( range (self.num2)):
            val = int(wbias[i])
            addBiasBlock[i] = addBiasBlock[i] + val
        global totalSamplesBlock
        #print ("---total samples before adding user---",totalSamplesBlock)
        totalSamplesBlock = totalSamplesBlock + self.n_samples
        #print ("---total samples after adding user---",totalSamplesBlock)
    
    def calculate_cosine(self,epoch_num,Voi):
        weights = []
        for i in range (self.num2):
            wt2s = copy.deepcopy(weightsBlock[i])
            wt2s = [float(item/100000000) for item in wt2s]
            weights.append(wt2s)
        bias = copy.deepcopy(biasBlock)
        bias = [float(item/100000000) for item in bias]
        # Weights before training #
        weights = torch.tensor(weights)
        bias = torch.tensor(bias)
        wt2train = self.model.l1[0].weight
        wbias = self.model.l1[0].bias
        wt2train = torch.tensor(wt2train)
        wbias = torch.tensor(wbias)
        wt2train = wt2train/ self.n_samples
        wt2train = wt2train/100000000
        wbias = wbias/ (self.n_samples*100000000)
        print ("----------------------Downloaded trained user weight in cosine funtion--------------------")
        print (wt2train)
        if epoch_num>0: # k>0
            W =  wt2train - weights
            cos = nn.CosineSimilarity()
            Voi = torch.tensor(Voi)
            Voi = Voi/100000000
            output = cos(Voi, W)
            cosines[self.ac][0] = output
            cosines[self.ac][1] =  self.ac

    def calculate_norm(self):
       
        weights = []
        for i in range (self.num2):
            wts = copy.deepcopy(weightsBlock[i])
            wts = [float(item/100000000) for item in wts]
            weights.append(wts)
        #bias = bias 
        bias = copy.deepcopy(biasBlock)
        bias = [float(item/100000000) for item in bias]
        # Weights before training #
        weights = torch.tensor(weights)
        bias = torch.tensor(bias)
        print ("----------------------Downloaded weights user (original unchanged weights, in norm function) --------------------")
        print (weights)
        ## Trained weights
        wt2 = self.model.l1[0].weight
        wbias = self.model.l1[0].bias
        wt2 = torch.tensor(wt2)
        wbias = torch.tensor(wbias)
        W =  weights - wt2
        norma = ((W)**2)**0.5
        norm = torch.sum(norma)
        norms[self.ac][0] = norm
        norms[self.ac][1] = self.ac

    #---------- user Performance Functions -------------- #
    def pred_user(self,x, y):  
        comp = torch.zeros((1,x.shape[0]))
        predict = self.model1(x.float())   
        pred = (predict.data>0.5).float()  
        return (pred)
    def test_performance_user(self,dataset_test, Y_test):
        predicted = self.pred_user(dataset_test[:][0], dataset_test[:][1])
        pred1 = torch.reshape(predicted,(-1,))
        count = 0
        sh = Y_test.shape[0]
        for i in range (sh):
            if Y_test[i] == int(pred1[i]):
                count +=1 
        total_images = Y_test.shape[0]
        print ("------------- User Performance----------------------")
        print (total_images)
        print (count)
        print (count/total_images)

n_epochs = 24
n_training_epochs = 3# individual training epochs for users
user_list = [0] * num_users
for k in range(1,n_epochs+1):
    print ("#################################################### Epoch ", k, " ##########################################################")
    # Reset the matrix
    print ("----------------Reset Weights-----")
    for i in range (num2):
        for j in tqdm(range (num1)):
            addWeightsBlock[i][j] = 0
    print ("----------------Reset Bias-----")  
    for i in tqdm(range(num2)):
        addBiasBlock[i] = 0
    for user_id in range(0,num_users):
        #m = (k*batch_size)-i
        m = user_id
        user_list[user_id] = Users(user_id) 
        user_list[user_id].play(n_training_epochs)
        user_list[user_id].calculate_norm()
    norms = torch.tensor(norms)
    sum_of_diff = [0]*num_users
    for g in range(num_users):
        sum_of_diff[g] = [0]*2 # store the sum of difference and index
    for e in range(0,num_users):
        sum1 = 0
        for s in range(0,num_users):
            sum1 = sum1 + (((norms[e][0] - norms[s][0])**2)**0.5)
        sum_of_diff[e][0] = sum1
        sum_of_diff[e][1] = e
    # sort the sum of differences
    sorted_norm_differences = sorted(sum_of_diff, key=lambda x: x[0])
    print ("-------Sorted sum of differences -------", sorted_norm_differences)
    # Pick first 10 indexes from norm
    # calculate cosine only for these
    picked_norm_indexes = []
    for s in range(0,10):
        picked_norm_indexes.append(sorted_norm_differences[s][1])
    print ("Picked norm indexes  ", picked_norm_indexes)
    print ("----------- Picked norm indexes and their norms ------")
    for d in range(0, len(picked_norm_indexes)):
        ind = picked_norm_indexes[d]
        print (" Index ", ind)
        print (" norm ", norms[ind])
    print ("-------------------------------------------------")
    picked_indexes = [c for c in range (num_users)]
    print (" Add weights block array before aggregation ", addWeightsBlock[0][:5])
    for h in range(0,len(picked_norm_indexes)):
        ac = picked_norm_indexes[h]
        user_list[ac].prepare_for_blockchain()
        user_list[ac].add_to_blockchain()
    # Aggregate the weights
    addWeightsBlockCopy = copy.deepcopy(addWeightsBlock) # This copy variable should exist in the blockchain as well
    for i in range (num2):
        for j in tqdm(range (num1)):
            """ ----- Changing the original code here - now the addweight array itself is being divided by total number of samples to calculate cosine similarity ----"""
            #weightsBlock[i][j] = int(addWeightsBlock[i][j]/totalSamplesBlock)
            addWeightsBlockCopy[i][j] = int(addWeightsBlock[i][j]/totalSamplesBlock)
    for i in range (num2):
        for j in tqdm(range (num1)):
                Voi[i][j] = addWeightsBlockCopy[i][j] - weightsBlock[i][j]
    print ("---------- Voi ----------------  ", Voi[0][:5])
    for user_id in picked_norm_indexes:
        print (user_id)
        user_list[user_id].calculate_cosine(k,(Voi))

    if k > 0:  
        
        # sort the cosines
        picked_indexes = []
        print ("----- Cosines before sorting----- ", cosines)
        sorted_cosines = sorted(cosines, key=lambda x: x[0], reverse=True)
        print ("-------Sorted Cosines -------", sorted_cosines)
        
        # pick the first n elements - n can be 2, 4 or 8
        # change this according to requirement
        for x in range(4):
            picked_indexes.append(sorted_cosines[x][1])
        f1 = open("result_picked_indexes.csv", "a")
        f1.write(str(picked_indexes)+"\n")
        for i in range (num2):
            for j in tqdm(range (num1)):
                addWeightsBlock[i][j] = 0
        print ("----------------Reset Bias-----")  
        for i in tqdm(range(num2)):
            addBiasBlock[i] = 0          
        # ------- Restart number of samples ----------------#
        totalSamplesBlock = 0

        print ("-- After reset for k > 0 ---- ")
        print (addWeightsBlock[0][:5])

        print ("--- Now in adding to blockchain after thrid epoch ---- check selected user indexes----")
        #for b in range (0, len(picked_indexes)):
        for b in (picked_indexes):
            user_list[b].add_to_blockchain()
        for i in range (num2):
            for j in tqdm(range (num1)):
                """ ----- Changing the original code here - now the addweight array itself is being divided by total number of samples to calculate cosine similarity ----"""
                weightsBlock[i][j] = int(addWeightsBlock[i][j]/totalSamplesBlock)
                
    else:
        # update weights in blockchain
        print ('picked indexes ',picked_norm_indexes)
        f1 = open("result_picked_indexes.csv", "a")
        f1.write(str(picked_norm_indexes)+"\n")
        weightsBlock = copy.deepcopy(addWeightsBlockCopy)

    print ("-------- Final copied weights updated in Blockchain ---------", weightsBlock[0][:5]) 

    '''
    ADD SANITY CHECK HERE FOR BIAS
    ''' 
    for i in tqdm( range (num2)):
        biasBlock[i] = int(addBiasBlock[i]/totalSamplesBlock)
    print ("----- reset cosines -----------")
    cosines = [0] * num_users
    for p in range (num_users):
        cosines[p]= [-2]*2
    
    print ("----------- Cosines after epoch - should be -2 -----------")
    print (cosines)
    # ------------ Test performance after every epoch ------------#
    print ("-----------------Testing performance------------------")
    testing()
    # ------- Restart number of samples ----------------#
    totalSamplesBlock = 0
    