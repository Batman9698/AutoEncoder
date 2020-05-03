import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from MNIST import MNIST
from AutoEncoder import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'training hardware:{device}')

path = os.getcwd()+'/data'
trainset = MNIST(path+'/mnist_train.csv', transform = transforms.ToTensor())
testset = MNIST(path+'/mnist_test.csv', transform = transforms.ToTensor())
trainloader = DataLoader(trainset, batch_size = 8, shuffle = True)
testloader = DataLoader(testset, batch_size = 8, shuffle = True)

model = AutoEncoder_v2(encoding_dim = 32).to(device)#change to required version
print(model)

MSE = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

epochs = 20
print(f'training for {epochs} epochs')
for epoch in range(epochs):
    training_loss = 0.0
    validation_loss = 0.0
    
    model.train()
    for data in trainloader:
        images, labels = data
        images = images.view(images.size(0), -1).to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = MSE(outputs, images)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()*images.size(0)
        
    model.eval()
    for data in testloader:
        images, labels = data
        images = images.view(images.size(0), -1).to(device)
        
        outputs = model(images)
        
        loss = MSE(outputs, images)
        validation_loss += loss.item()*images.size(0)
    
    training_loss = training_loss/len(trainloader)
    validation_loss = validation_loss/len(testloader)
    
    print(f'epoch: {epoch+1} \t Training loss: {round(training_loss, 4)} \t Validation loss: {round(validation_loss, 4)}')


torch.save(model.state_dict(), os.getcwd()+'/models/AE_v5')
print('model saved')