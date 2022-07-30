from cProfile import label
from email import header
import json
from marshal import load
from operator import mod
import pickle
from pyexpat import model
from pyexpat.errors import XML_ERROR_DUPLICATE_ATTRIBUTE
from re import A
from cv2 import mean
#from matplotlib.font_manager import _Weight
from matplotlib.lines import Line2D
import numpy as np
#import user1 as am1 # file for calling functions of first node
from collections import Counter
from collections import defaultdict
from random import random
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
ORIGINAL CODE SIMULATION
ROLL OVER
SIMULATION

variables to change: n_epochs, batch_size_users depending on the roll over case

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

cosines = [0] * 5 # 5 is the number of users in the system
##### Variables for Cosine Similarity
#Voi = weightsBlock[:]
for m in range (5):
    cosines[m]= [0]*2 # we will store the cosine and the account index as well

Voi = copy.deepcopy(weightsBlock)
print ("------- Voi ------- ", Voi[0][:5])
#Woi = weightsBlock
Woi = copy.deepcopy(weightsBlock)

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
#optimizer = torch.optim.SGD(model1.parameters(), lr = 0.01)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
#n_epochs = 4 # need to be changed for each rolling case

print ("--------------- Hospital model -----------------")
#features = np.load("histoptest_features.npy")
#labels = np.load("histoptest_labels.npy")
features = np.load("COVIDX_test_features.npy")
labels = np.load("COVIDX_test_labels.npy")
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
    #web3.eth.defaultAccount = web3.eth.accounts[0] 
    weights = []  
    for i in range (num2):

        wts = copy.deepcopy(weightsBlock[i])
        wts = [float(item/100000000) for item in wts]
        weights.append(wts)
    biasa = copy.deepcopy(biasBlock)
    bias = [float(item/100000000) for item in biasa]
    weights = torch.tensor(weights)
    bias = torch.tensor(bias)
    print ("----------------------Downloaded weights for testing-------------------")
    with torch.no_grad():
        model1.l1[0].weight.copy_(weights.float())
        model1.l1[0].bias.copy_(bias.float())
    print ("Weights for testing")
    print ("----------------- Parameters for testing--------------")
    for param in model1.parameters():
        print(param.data)
    test_performance()

def pred(x, y):  
       
        predict = model1(x.float())  
        pred = (predict.data>0.5).float() 
        print ("Criterion ", criterion)
        loss = criterion(predict,y.float())
        
        print ("----------------------- Testing Loss ---------------------- ", loss.item())
        print ("---------------- Prediction Shape -----------")
        print (predict.shape)
        return (pred)
def test_performance():
    predicted = pred(dataset_test[:][0], dataset_test[:][1])
    pred1 = torch.reshape(predicted,(-1,))
    cm = confusion_matrix(Y_test,pred1)
    print ("**************** Confusion Matrix *************************")
    print (cm)
    count = 0
    for i in range (sh):
        if Y_test[i] == int(pred1[i]):
            count +=1 
    total_images = Y_test.shape[0]
    print ("predicted shape", pred1.shape)
    print ("total images ", total_images)
    print ("Count", count)
    accuracy = (count/total_images)*100
    f = open("result_roll_class.csv", "a")
    f.write(str(accuracy)+"\n")
    f.close()
    print ("Testing accuracy", count/total_images)

test_performance()

#print ("initial random weights")
#print (model1.l1[0].weight)
for param in model1.parameters():
    param.data = (param.data)* 100000000

wt2 = model1.l1[0].weight
bias = model1.l1[0].bias

print (type(list(wt2)))


# Storing Initial random Weights in the Global arrays 

print ("Start Time ",datetime.datetime.now())
for i in range (num2):
    for j in tqdm(range (num1)):
        val = int(wt2[i][j])
        weightsBlock[i][j] = val

for i in tqdm(range(num2)):
    val = int(bias[i])
    biasBlock[i] = val

print ("---- Initial value of weights block-------", weightsBlock[0][:5])
# Ideally hospital does not have access to these files
features_all = np.load('histoptrain_features.npy')
labels_all = np.load('histoptrain_labels.npy')
num = 100
##### CLASS VARIABLES ##########
class Users:
    def __init__(self, accoun_index) -> None:
        self.ac = accoun_index 
        # everything defined in user_play function
        # Defining the data
        #print ("--------- Data points ---- ", (self.ac)*num, (num)*(self.ac+1))
        #self.features = features_all[(self.ac)*num:(num)*(self.ac+1)]
        #self.labels = labels_all[(self.ac)*num:(num)*(self.ac+1)]
        self.filenameFeatures = 'COVIDX_train_features_100_' + str(self.ac) + '.npy'
        self.filenameLabels = 'COVIDX_train_labels_100_' + str(self.ac) + '.npy'
        self.features = np.load(self.filenameFeatures)
        self.labels= np.load(self.filenameLabels)
        print ("--- Feature data sanity check-----", self.features[:5])
        print ("--- Label data sanity check-----", self.labels[:5])
        
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
        #criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), weight_decay = 0.05, lr = 0.005) #0.0005
        #self.optimizer =  torch.optim.SGD(self.model.parameters(), lr = 0.01)
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
        """
        Training function which takes as input a model, a learning rate and a batch size.
        After completing a full pass over the data, the function exists, and the input model will be trained.
        """
        # -- Your code goes here --
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
        #return (self.model,average_loss)

    def sanity_check(self):
        print ("---- Model Weights -----")
        print (self.model.l1[0].weight)
    
    def play(self,epochs):
        print ("------------------Downloaded Weights User from Blockchain  ----------------------------")
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
        print (weights)
        print (bias)
        #print ("------------Updated weights----------")
        with torch.no_grad():
            self.model.l1[0].weight.copy_(weights.float())
            self.model.l1[0].bias.copy_(bias.float())
        print ("Weights before training")
        print ("----------------- Parameters before trainin--------------")
        for param in self.model.parameters():
            print(param.data)
        print ("---------- Training Start Time ----------------")
        print (datetime.datetime.now())
        self.train(epochs)
        print ("---------- Training End Time ----------------")
        print (datetime.datetime.now())
        print ("------- Sanity check ------ Weights after training---")
        self.sanity_check()

        """
        #CHANGE THE CODE FROM HERE FOR USER SELECTION - PUT THE BELOW SECTOIN IN DIFFERENT FUNCTIONS
        """
        #------------------------ Prepare the weights to store in blockchain ------------------#

        for param in self.model.parameters():
            param.data = (param.data)* 100000000 * self.n_samples
        
        '''
        Code for weight poisoning
        '''
        
        if (self.ac == 10):
            print ("------------- Poisoning the weights -----")
            for param in self.model.parameters():
                param.data = param.data * (-10)

        # weighst after training #
        wt2 = self.model.l1[0].weight
        wbias = self.model.l1[0].bias

        print ("--------- Sanity check -------- Weights prepared for blockchain after training ---------")
        print (wt2)
        for i in range (self.num2):
            for j in tqdm(range (self.num1)):
                val = int(wt2[i][j])
                addWeightsBlock[i][j] = addWeightsBlock[i][j] + val
                
        for i in tqdm( range (self.num2)):
            val = int(wbias[i])
            addBiasBlock[i] = addBiasBlock[i] + val
        
        print ("$$$$$$$$$$$$$ After Training $$$$$$$$$$$$$")
        print ("------- Weights Block (should remain same )--------- ",weightsBlock[0][:5] )
        print ("------- Added weights Block (Should have been updates) ----", addWeightsBlock[0][:5])

        global totalSamplesBlock
        print ("---total samples---",totalSamplesBlock)
        totalSamplesBlock = totalSamplesBlock + self.n_samples
        print ("---total samples---",totalSamplesBlock)
        
    
    

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
        #loss = criterion(pred1,Y_test)
        #print ("Initial Loss", loss)
        print ("------------- User Performance----------------------")
        print (total_images)
        print (count)
        print (count/total_images)

#######-------------- USER CLASS ENDS -------------------------------------------- ####################

### ----------------- FUNCTIONS CALLED BY HOSPITAL MODEL ------------------####################

#randomlist = rand.sample(range(0, 80), 16)
num_users = 16
#print ("$$$$$$$$$$$$$$$$$$$$$$$",randomlist)
for all_epochs in range(3):
    n_epochs = 8 # represents batch roll overs, in actual we are dealing with 1 epoch overall
    n_training_epochs = 8
    user_list = [0]*num_users
    for k in range(1,n_epochs+1): # here n_epochs is actually batch
        print ("#################################################### BATCH ", k, " ##########################################################")
        # Reset the matrix
        print ("----------------Reset Weights-----")
        for i in range (num2):
            for j in tqdm(range (num1)):
                addWeightsBlock[i][j] = 0
        print ("----------------Reset Bias-----")  
        for i in tqdm(range(num2)):
            addBiasBlock[i] = 0
        #------------------ For each user ---------------------------------#
        #for i in range(1,9):
        batch_size_users = 2
        # 5 users

        for user_id in range(0,batch_size_users):
            m = (k*batch_size_users)-user_id
            print ("Index value ", m-1)
            #m = user_id
            # random variables - delete later - adding as per original functions
            web3 = 0
            contract = ''
            add = m 
            #ind = randomlist[i]
            # Define a new user
            user_list[m-1] = Users(m-1) # ideally we should not need to define users again and again
            print ("---------USER ----",add,"-----------------")
            user_list[m-1].play(n_training_epochs)
        #-------- Average out the weights ----------------#
        print ("---------------------Averaging out the weights and bias ---------------")
        print ("--------------------Original Weighst -----------------------")
        print ("---------- Block Weights (should not be changed) ------ ", weightsBlock[0][:5])
        print(" --------------------- Added weights (should be changed) ------------------------ ")
        print (addWeightsBlock[0][:5])
        for i in range (num2):
            for j in tqdm(range (num1)):
                """ ----- Changing the original code here - now the addweight array itself is being divided by total number of samples to calculate cosine similarity ----"""
                weightsBlock[i][j] = int(addWeightsBlock[i][j]/totalSamplesBlock)
                #addWeightsBlockCopy[i][j] = int(addWeightsBlock[i][j]/totalSamplesBlock)
        #print ("---------- Add weights -------- ", addWeightsBlock)
        print ("---------- Block Weights (should be changed)------ ", weightsBlock[0][:5])
        #print ("-------------Samples ------------", totalSamplesBlock[0][:5])
        print ("--------- Aggregated Weights ------ ", addWeightsBlock[0][:5])
        for i in tqdm( range (num2)):
            biasBlock[i] = int(addBiasBlock[i]/totalSamplesBlock)
        #print ("----------- Cosines after epoch -----------")
        #print (cosines)
        # ------------ Test performance after every epoch ------------#
        print ("-----------------Testing performance------------------")
        testing()
        # ------- Restart number of samples ----------------#
        totalSamplesBlock = 0

