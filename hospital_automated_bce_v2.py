# code without normalization and automated users
# This time we roll over all the users in the system 
# The only difference in this file -> We are running for more than 1 overall epoch
# Use output size = 1
# DEPENDING ON THE NUMBER OF TOTAL USERS - BATCH SIZE AND NUM_EPOCHS WILL CHANGE
# TAKING TOTAL AGGREGATIONS = 24
from email import header
import json
from marshal import load
from operator import mod
from pyexpat import model
from pyexpat.errors import XML_ERROR_DUPLICATE_ATTRIBUTE
from xml.etree.ElementTree import C14NWriterTarget
from cv2 import mean
from matplotlib.lines import Line2D
import numpy as np
#import user1 as am1 # file for calling functions of first node
from collections import Counter
from collections import defaultdict
#from random import Random
import random as rand
from collections import defaultdict
#from IPython import display
from PIL import Image
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
from sklearn.metrics import confusion_matrix
from web3 import Web3 
import json
#from web3 import web3practice
from web3 import Web3
import automated_user_bce as u1
import datetime

#from hospital_blockchain_withoutnorm import X_test, Y_test

#play_functions = [u1.play]
#num_users = len(play_functions)
#model_file = 'histop_linear_model.pt'
#model_file = 'covidlinear.pt'
#model_file = 'bilinear.pt'

random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

class cancerClassification(nn.Module):
    def __init__(self, input_size, num_classes):
        super(cancerClassification, self).__init__()
        self.l1 = nn.Sequential(
            #nn.LayerNorm(input_size),
            nn.Linear(input_size,num_classes)
        )
    def forward(self, x):
        out = self.l1(x)
        out = torch.sigmoid(out) # We require this if we use BCE LOSS # remove this in case of cross entropy loss
        return out
"""
Change this after defining test set
n_inputs = X_df.shape[1]
n_outputs = 7
n_samples = X_df.shape[0]
"""        

#model1 = cancerClassification(n_inputs, n_outputs)
#model1 = cancerClassification(1024, 2)
model1 = cancerClassification(1024, 1)
# for 1 epoch
#criterion = nn.CrossEntropyLoss(reduction='mean')
criterion = nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model1.parameters(), weight_decay = 0.05, lr = 0.005)
#optimizer = torch.optim.SGD(model1.parameters(), lr = 0.005)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
n_epochs = 8 # need to be changed for each rolling case
#model1.l1.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
print ("--------------- Hospital model -----------------")

#print (wt2[0][1])

# ------------------- Code for testing --------------------------- #
#features = np.load("bitest_features.npy")
#labels = np.load("bitest_labels.npy")
######features = np.load("histoptest_features.npy")
######labels = np.load("histoptest_labels.npy")
features = np.load("COVIDX_test_features.npy")
labels = np.load("COVIDX_test_labels.npy")
print (features.shape)
X_test = features
Y_test = labels
print (Y_test[:10])
print (X_test.shape)
#X_test = features[1200:1300]
#Y_test = labels[1200:1300]
sh = Y_test.shape[0]
print (Y_test.shape)
Y_test = Y_test.reshape(Y_test.shape[0],1)
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
def testing():
    #CODE FOR LAST TESTING
    web3.eth.defaultAccount = web3.eth.accounts[0] 
    #normalized_weights = contract.functions.returnNormalizedWeights().call()
    #normalized_weights = [float(item/100000000) for item in normalized_weights]

    #beta = contract.functions.returnBeta().call()
    #beta = [float(item/100000000) for item in beta]
    weights = []  
    for i in range (num2):
            
        wts = contract.functions.returnInitialWeights(i).call()
        wts = [float(item/100000000) for item in wts]
        weights.append(wts)
        #print ("---------",len(wts),"-----------")
        #print (len(weights))
    bias = contract.functions.returnBias().call()
    bias = [float(item/100000000) for item in bias]

    #normalized_weights = torch.tensor(normalized_weights)
    weights = torch.tensor(weights)
    bias = torch.tensor(bias)
    #beta = torch.tensor(beta)
    print ("----------------------Downloaded weights for testing-------------------")
    #print (weights)
    #print (bias)
    #print ("------------Updated weights----------")
    with torch.no_grad():
        #model1.l1[0].weight = torch.nn.parameter.Parameter(normalized_weights)
        #model1.l1[1].weight = torch.nn.parameter.Parameter(weights)
        #model1.l1[0].weight.copy_(normalized_weights.float())
        #model1.l1[0].bias.copy_(beta.float())
        model1.l1[0].weight.copy_(weights.float())
        model1.l1[0].bias.copy_(bias.float())
    print ("Weights for testing")
    #print (model1.l1[0].weight)
    #print (model1.l1[0].bias)

    print ("----------------- Parameters for testing--------------")
    for param in model1.parameters():
        print(param.data)
    #predict = model1(x.float())  
    #loss = criterion(predict,y)
    #print ("--------------------- Loss for the epoch ---------------- ", loss.item())

    test_performance()



def pred(x, y):  
        """
        print (x.shape)
        print ("X")
        print (x[:10])
        """
        comp = torch.zeros((1,x.shape[0]))
        print ("Initial comp----", comp)

        #print ("y")
        #print (y[:10])
        
        #for param in model1.parameters():
        #    print (param.data)

        predict = model1(x.float())  
        pred = (predict.data>0.5).float()  
        #print ("-----------New Pred -----------", pred)
        """
        print ("Predict")
        print (predict[:10]) 
        print (predict.shape)
        print (Y_test.shape)
        """
        #Yt = torch.from_numpy(Y_test, dtype=torch.long)
        #Yt = torch.tensor(y, dtype=torch.long)
        #xx = torch.tensor([[0.1, 0.9], [0.4, 0.5]])
        #Yy = torch.tensor([1, 1])

        print ("Criterion ", criterion)
        #print ("Y",y)
        #print (type(predict), type(y))
        loss = criterion(predict,y.float())
        print ("----------------------- Testing Loss ---------------------- ", loss.item())
        print ("--------------------prdict shape----------------------------",predict.shape)
        '''
        for i in range(0, predict.shape[0]):
            #comp[0,i] = max(predict[i]) 
            comp[0,i] = torch.argmax(predict[i]) 
        print ("New comp--------------",comp)
        comp = comp.T
        print (comp.shape)
        '''
        """
        Confusion Matrix
        """

        #return (comp)
        return pred

def test_performance():
    predicted = pred(dataset_test[:][0], dataset_test[:][1])
    print ("pred1 shape", predicted.shape)
    pred1 = torch.reshape(predicted,(-1,))
    print ("pred1 shape", pred1.shape)
    #print ("New pred in testing", pred1)
    cm = confusion_matrix(Y_test,pred1)
    print ("**************** Confusion Matrix *************************")
    print (cm)
    count = 0
    print ("Shape of Y_test", Y_test.shape)
    #print (Y_test)
    print ("Sh",sh)
    #print (pred1)
    for i in range (sh):
        #print (Y_test[i])
        if Y_test[i][0] == int(pred1[i]):
            #print (Y_test[i])
            count +=1 
    total_images = Y_test.shape[0]
    print ("predicted shape", pred1.shape)
    print ("total images ", total_images)
    print ("Count", count)
    accuracy = (count/total_images)*100
    f = open("result_rollover.csv", "a")
    f.write(str(accuracy)+"\n")
    f.close()
    print ("Testing accuracy", count/total_images)
#################################### Blockchain Code#######################

test_performance()

for param in model1.parameters():
    param.data = (param.data)* 100000000
    #print (param.data)

#print (model1.l1[0].weight)
#print (model1.l1[1].weight)
#wt1 = model1.l1[0].weight
wt2 = model1.l1[0].weight
bias = model1.l1[0].bias
#beta = model1.l1[0].bias
print (type(list(wt2)))

ganache_url = "HTTP://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url, request_kwargs={'timeout': 60}))
print (web3.isConnected())
#truffleFile = json.load(open('src/abis/decentraldl.json'))
truffleFile = json.load(open('/Users/garimaaggarwal/Documents/Blockchainprojects/decentralizeddl/build/contracts/decentraldl.json'))
abi = truffleFile['abi']
#print (abi)
contract_address_string = truffleFile['networks']['5777']['address'] # contract address
print (contract_address_string)
contract_address = web3.toChecksumAddress(contract_address_string)
accounts = web3.eth.accounts
contract = web3.eth.contract(address = contract_address, abi = abi)
first = 1 # set this to 1 only if this is the first time this program is being run
#for i in range (len(accounts)):
num1 = 1024
num2 = 1
num3 = num1*num2
print (accounts[0])
web3.eth.defaultAccount = web3.eth.accounts[0]  
nums = []
# Saving initial weights to Blockchain
"""
for i in tqdm( range (1024)):
    val = int(wt1[i])
    val2 = int(beta[i])
    #print (type(val))
    tx_hash1 = contract.functions.storeNormalizedWeights(i,val).transact()
    tx_receipt1 = web3.eth.waitForTransactionReceipt(tx_hash1)
    tx_hash = contract.functions.storeBeta(i,val2).transact()
    tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)
"""
print ("Start Time ",datetime.datetime.now())
for i in range (num2):
    for j in tqdm(range (num1)):
        val = int(wt2[i][j])
        tx_hash = contract.functions.storeInitialWeights(i,j,val).transact()
        tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)

for i in tqdm(range(num2)):
    val = int(bias[i])
    tx_hash = contract.functions.storeBias(i,val).transact()
    tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)
"""
"""
"""
print ("Uploaded Weights")
for i in range (num2):
    wts = contract.functions.returnInitialWeights(i).call()
    print (wts)
"""
########################## CODE FOR  USERS ##############################
# original code - delete this function


######## ------------ Code for users -------------------------- #####################
#randomlist = rand.sample(range(0, 80), 16)
randomlist = [x for x in range(0,16)]

for epoch_id in range(3):

    print ("@@@@@@@@@@@@@@@@@@@@@ EPOCH ",epoch_id," @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    
    for k in range(1,n_epochs+1):
        print ("#################################################### Batch ", k, " ##########################################################")
        # Reset the matrix
        #tx_hash = contract.functions.resetMatrices().transact()
        #tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)
        print ("----------------Reset Weights-----")
        for i in range (num2):
            for j in tqdm(range (num1)):
                tx_hash = contract.functions.resetWeights(i,j).transact()
                tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)
        print ("----------------Reset Normalized Weights and Beta-----")
        """
        for i in tqdm( range (1024)):
            tx_hash1 = contract.functions.resetNormalizedweights(i).transact()
            tx_receipt1 = web3.eth.waitForTransactionReceipt(tx_hash1)
            tx_hash = contract.functions.resetBeta(i).transact()
            tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)
        """
        print ("----------------Reset Bias-----")
        for i in tqdm(range(num2)):
            tx_hash = contract.functions.resetBias(i).transact()
            tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)
        #------------------ For each user ---------------------------------#
        
        #for i in range(1,9):
        batch_size = 2
        for i in range(0,batch_size):
            m = (k*batch_size)-i
            #add = accounts[i]
            add = accounts[m]
            print (add)
            ind = randomlist[i]
            print ("-------------",add,"-----------------")
            #u1.play(web3,add,i,contract)
            u1.user_play(web3,add, m, contract, k, ind)
            #play_functions[i-1](web3,add,i,contract)
        #-------- Average out the weights ----------------#
        """
        for i in tqdm( range (1024)):
            tx_hash = contract.functions.averageNormalizedWeights(i).transact()
            tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)
            tx_hash2 = contract.functions.averageBeta(i).transact()
            tx_receipt2 = web3.eth.waitForTransactionReceipt(tx_hash2)
        """
        for i in range (num2):
            for j in tqdm(range (num1)):
                tx_hash = contract.functions.averageWeights(i,j).transact()
                tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)
            
        for i in tqdm( range (num2)):
            tx_hash = contract.functions.averageBias(i).transact()
            tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)

        # ------------ Test performance after every epoch ------------#
        print ("-----------------Testing performance------------------")
        testing()
        # ------- Restart number of samples ----------------#
        tx_hash = contract.functions.restartNsamples().transact()
        tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)
    print ("@@@@@@@@@@@@@@@@@@@@@@ End of EPOCH ", epoch_id," @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print ("----------------------Accuracy------------")
    testing()
    print ("End Time ",datetime.datetime.now())
    #print ("End Time ",datetime.datetime.now())
#print ("-----------------Testing performance------------------")
#test_performance()
