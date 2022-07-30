# Automated User, but we roll over all the users 
# output size = 1
# Changed the optimizer to SGD, removed the schgeduler
"""
original plan
# DEFINE DATA SET
# DEFINE LOCAL MODEL
# DOWNLOAD WEIGHTS FROM BLOCK CHAIN AND UPDATE LOCAL MODEL
# TRAIN THE LOCAL MODEL
# SEND WEIGHTS AND NUMBER OF IMAGES TO BLOCKCHAIN
"""
"""
New plan
1. Call in the local data
2. Call the weights file
3. Update the weights
4. Train for 1 epochs
5. Send the weights to hospital
6. Calculate Performance
"""

# BATCH SIZE 16
# LR 0.005

# TIME FOR TRAINING
# PUT THE DIAGRM ON GOOGLE SLIDES


from email import header
from pyexpat import features
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as F 
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim 
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pt
import matplotlib as plt
from glob import glob        
import matplotlib.pyplot as plt  
from sklearn.metrics import accuracy_score
from web3 import Web3 
import pandas as pd
import datetime
import random


random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

class cancerDataset(Dataset):
    def __init__(self,X,Y):
        self.n_samples = X.shape[0]
        self.X = (torch.from_numpy(X))
        self.Y = (torch.from_numpy(Y))
    def __getitem__(self,index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.n_samples

# Custom Neural network
class cancerClassification(nn.Module):
    def __init__(self, input_size, num_classes):
        super(cancerClassification, self).__init__()
        self.l1 = nn.Sequential(
            #nn.LayerNorm(input_size),
            nn.Linear(input_size,num_classes)
        )
    def forward(self, x):
        out = self.l1(x)
        out = torch.sigmoid(out) # We require this if we use BCE LOSS, REMOVE for Cross ENtropy Loss
        return out



def pred(x, y, model1):  
        comp = torch.zeros((1,x.shape[0]))
        predict = model1(x.float()) 
        pred = (predict.data>0.5).float()  
        #test_correct = (pred.view(-1) == labels).float()
        #loss = criterion(predict, )
        #print ("Shape of predict", predict.shape)
        '''
        for i in range(0, predict.shape[0]):
            #comp[0,i] = max(predict[i]) 
            comp[0,i] = torch.argmax(predict[i]) 
        #print (comp)
        comp = comp.T
        #print (comp.shape)
        return (comp)
        '''
        return pred

def test_performance(dataset_test, model, Y_test):
    predicted = pred(dataset_test[:][0], dataset_test[:][1], model)
    pred1 = torch.reshape(predicted,(-1,))
    count = 0
    sh = Y_test.shape[0]
    for i in range (sh):
        if Y_test[i][0] == int(pred1[i]):
            count +=1 
    total_images = Y_test.shape[0]
    #loss = criterion(pred1,Y_test)
    #print ("Initial Loss", loss)
    print ("------------- User Performance----------------------")
    print (total_images)
    print (count)
    print (count/total_images)

def user_play(web3,add, account_index, contract, epoch_num,ind):
    # Define training set

    """
    filenameFeatures = 'user' + str(account_index) + 'biFeatures.npy'
    filenameLabels = 'user' + str(account_index) + 'biLabels.npy'
    """
    #features = np.load('user1biFeatures.npy')
    #labels = np.load('user1biLabels.npy')
    #features = np.load(filenameFeatures)
    #labels = np.load(filenameLabels)
    
    
    #features_all = np.load('bitrain_features.npy')
    #labels_all = np.load('bitrain_labels.npy')
    ####features_all = np.load('histoptrain_features.npy')
    ####labels_all = np.load('histoptrain_labels.npy')
    #features_all = np.load('simclrfeatures.npy')
    #labels_all = np.load('simclrlabels.npy')
  
    #Generate 5 random numbers between 10 and 30
    #COVIDX_train_features_100_0
    #filenameFeatures = 'COVIDX_train_features_100_' + str(ind) + '.npy'
    #filenameLabels = 'COVIDX_train_labels_100_' + str(ind) + '.npy'
    filenameFeatures = 'COVIDX_train_features_100_' + str(account_index-1) + '.npy'
    filenameLabels = 'COVIDX_train_labels_100_' + str(account_index-1) + '.npy'
    
    print ("----------------------------------------",filenameFeatures,"---------------------------------")
    
    features = np.load(filenameFeatures)
    labels = np.load(filenameLabels)
    
    #num = int((features_all.shape[0])/100)
    num = 100
    """
    features = features_all[(account_index-1)*num:(num)*account_index]
    labels = labels_all[(account_index-1)*num:(num)*account_index]
    """
    
    print ("------------ Account index ", account_index, str((account_index)*num),str((num)*(account_index+1))," ------------------")
    #features = features_all[(account_index)*num:(num)*(account_index+1)]
    #labels = labels_all[(account_index)*num:(num)*(account_index+1)]
    labels = labels.reshape(labels.shape[0],1)
    X_df = features[:95]
    Y_df = labels[:95]
    X_df_test = features[95:]
    Y_df_test = labels[95:]

    #print ("training shape ", X_df.shape)
    #print ("testing shape ", X_df_test.shape)
    dataset_test_user = cancerDataset(X_df_test, Y_df_test)

    device =''
    if torch.cuda.is_available() :
        device = torch.device("cuda")
        print("cuda") 
    else:
        device = torch.device("cpu")
# defining the parameters

    n_inputs = X_df.shape[1]
    #n_outputs = 2
    n_outputs = 1
    n_samples = X_df.shape[0]
    model1 = cancerClassification(n_inputs, n_outputs)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model1.parameters(), weight_decay = 0.05, lr = 0.005) #0.0005
    #optimizer = torch.optim.SGD(model1.parameters(), lr = 0.005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)#
    dataset = cancerDataset(X_df, Y_df)
    dataloader_train = DataLoader(dataset = dataset, batch_size = 16, shuffle = True)
    dataiter = iter(dataloader_train)
    total_loss = []
    average_loss = 0
    training_samples = len(X_df)
    num1 = 1024
    #num2 =2 
    num2 = 1
    num3 = num1*num2
    #address = '0xAB0313359bAc4278bA5873c355EB5D20356D7238'

    average_loss = play(web3,add, account_index, contract, num2, model1, n_samples, num1, dataloader_train,criterion, optimizer, scheduler,total_loss, dataset_test_user,Y_df_test,epoch_num,learning_rate=0.005, batch_size= 16, epochs=3)
    #data = [epoch_num, account_index, average_loss]
    #df = pd.DataFrame(data, columns = ['epoch','user','average_loss'])
    #pd.to_csv(df, mode='a', header= False)
    print ("--------------------- Average Loss for the User ---------------------", average_loss)
    # TRAIN FUNCTION - IT SHOULD ALSO RETURN THE MODEL WEIGHTS AND NUMBER OF SAMPLES
def train(model, dataloader_train,criterion, optimizer, scheduler,total_loss, learning_rate=0.005, batch_size=16, epochs=3  ):
    """
    Training function which takes as input a model, a learning rate and a batch size.
    After completing a full pass over the data, the function exists, and the input model will be trained.
    """
    # -- Your code goes here --
    average_loss = 0
    for i in range(0,epochs):
        model.train()
        for b_index, (features, labels) in enumerate(dataloader_train):
            #print (featueres.shape)
            #print (b_index)
            #outputs = model(features.float())
            outputs = model(features)
            loss = criterion(outputs,labels.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss.append(loss)
        
            if(b_index)%2 == 0:
                print (f'epoch {i+1}/{epochs} Batch {b_index+1} loss {loss}')
                

        average_loss = sum(total_loss)/len(total_loss)
        

               
        scheduler.step()
    print ("Training done")
    return (model,average_loss)

    
def play(web3,add, account_index, contract, num2, model1, n_samples, num1, dataloader_train,criterion, optimizer, scheduler,total_loss, dataset_test_user, Y_df_test,epoch_num,learning_rate=0.005, batch_size= 16, epochs=3):
    print ("------------------User Address -------------------------------------------- ", add)
    web3.eth.defaultAccount = web3.eth.accounts[account_index]  
    

    print ("------------------Downloaded Normalized Weights User  ----------------------------")
    weights = []
    for i in range (num2):
        #wts = contract.functions.returnInitialWeights(i).call()
        wts = contract.functions.returnInitialWeights(i).call()
        wts = [float(item/100000000) for item in wts]
        weights.append(wts)
        #print ("---------",len(wts),"-----------")
    #print (len(weights))
    #for Weight Poisoning:
    """
    if epoch_num == 0:
        for p in range(num1):
            weights[0][p] = 0
    """


     
    bias = contract.functions.returnBias().call()
    bias = [float(item/100000000) for item in bias]

    #normalized_weights = torch.tensor(normalized_weights)
    weights = torch.tensor(weights)
    bias = torch.tensor(bias)
    #beta = torch.tensor(beta)
    print ("----------------------Downloaded weights user --------------------")
    #print (weights)
    #print (bias)
    #print ("------------Updated weights----------")
    with torch.no_grad():
        model1.l1[0].weight.copy_(weights.float())
        model1.l1[0].bias.copy_(bias.float())
    """
    print ("Weights before training")
    print ("----------------- Parameters before trainin--------------")
    for param in model1.parameters():
        print(param.data)
    """
    # ---------------------------- Checking the accuracy of previous full model for every user ---------------------------- #
    print ("---Accuracy of Previous Fully Trained Model for this user ----")
    test_performance(dataset_test_user, model1, Y_df_test)
    print ("----------------------------------------------------")
    #----------------------------- train -------------------------#
    # ADD TIME HERE
    print ("---------- Training Start Time ----------------")
    print (datetime.datetime.now())
    model1, average_loss = train(model1, dataloader_train,criterion, optimizer, scheduler,total_loss, learning_rate=0.005, batch_size=16, epochs=3)
    # ADD TIME HERE
    print ("---------- Training End Time ----------------")
    print (datetime.datetime.now())
    """
    print ("updated weights after training")
    for param in model1.parameters():
        print(param.data) 
    """
    #------------------------ Checking the accuracy of the model after training for this user only ---------------------------- #
    print ("--------- Accuracy of Trained Model for this user --------- ")
    test_performance(dataset_test_user, model1, Y_df_test)
    print ("----------------------------------------------------")

    #------------------------ Prepare the weights to store in blockchain ------------------#
    for param in model1.parameters():
        param.data = (param.data)* 100000000 * n_samples
    #wt1 = model1.l1[0].weight
    wt2 = model1.l1[0].weight
    #wbeta = model1.l1[0].bias
    wbias = model1.l1[0].bias
    for i in range (num2):
        for j in tqdm(range (num1)):
            val = int(wt2[i][j])
            tx_hash = contract.functions.updateAddWeights(i,j,val).transact()
            tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)

    for i in tqdm( range (num2)):
        val = int(wbias[i])
        #print (type(val))
        tx_hash = contract.functions.updateAddBias(i,val).transact()
        tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)
    # Save the number of samples
    tx_hash = contract.functions.addSamples(n_samples).transact()
    tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)
    print ("------------------ Uploaded Weights ---------------------")
    print ("Number of samples")
    ns = contract.functions.returnSamples().call()
    print (ns)
    return average_loss
    