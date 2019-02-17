# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 21:45:43 2019

Structure Adapted from:
https://github.com/pytorch/examples/blob/master/mnist/main.py#L39
"""


import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import  numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.datasets as datasets

#Ignore warnings
import warnings 
warnings.filterwarnings("ignore")

#Define the structure of Neural Network
class MNIST_CNN(nn.Module):
    
    #define each layer of neural network
    def __init__(self):    
        super(MNIST_CNN, self). __init__()
        #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(1,16,5,1,2)
        self.conv2 = nn.Conv2d(16,32,3,1,1)
        self.conv3 = nn.Conv2d(32,64,3,1,1)
        self.conv4 = nn.Conv2d(64,128,3,1,1)
        self.conv5 = nn.Conv2d(128,256,3,1,1)
        #Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(256*3*3,256)
        self.fc2 = nn.Linear(256,10)
    
    #define how input will be processed through those layers
    def forward(self, x):
        x = F.relu(self.conv1(x)) # 28
        x = F.max_pool2d(x,2,2)   # 14
        x = F.relu(self.conv2(x)) # 14
        x = F.max_pool2d(x,2,1)   # 13
        x = F.relu(self.conv3(x)) # 13
        x = F.max_pool2d(x,2,1)   # 12
        x = F.relu(self.conv4(x)) # 12
        x = F.max_pool2d(x,2,2)   # 6
        x = F.relu(self.conv5(x)) # 6
        x = F.max_pool2d(x,2,2)   # 3
        
        #flatten the feature map
        x = x.view(-1, 256*3*3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        
        return x
    
#Define Training Method
def train(args, model, device, train_loader, optimizer, epoch):
        model.train()
        
        #define the operation batch-wise
        for batch_idx, (data, target) in enumerate(train_loader):
            #send the data into GPU or CPU
            data, target = data.to(device), target.to(device)
            #clear the gradient in the optimizer in the begining of each backpropagation
            optimizer.zero_grad()
            #get out
            output = model(data)
            #define loss
            loss = F.cross_entropy(output, target)
            #do backprobagation to get gradient
            loss.backward()
            #update the parameters
            optimizer.step()
            #show the training progress
            if batch_idx % args.log_interval == 0:
                #loss.item() gets the a scalar value held in the loss
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
#Define Test Method
def test(args, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        #What is this ?
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                #sum up batch loss
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                #get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                #calculate the right classification
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        #Average the loss (batch_wise)
        test_loss /= len(test_loader.dataset)
        
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def count_parameters(model):
    n_params = 0
    for param in model.parameters():
        if param.requires_grad:
            n_params += param.numel()
    return n_params

def main():    
    print("Pytorch Version:", torch.__version__)
    parser = argparse.ArgumentParser(description='TP1 MNIST CNN')
    #Training args
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    #set seed for randome number generation
    torch.manual_seed(args.seed)
    
    #Use GPU if it is available
    device = torch.device("cuda" if use_cuda else "cpu")

    #parameters left for multiple GPU for parallel work
    #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    #MNIST dataset
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
    mnist_test = datasets.MNIST(root='./data', train=False, download=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
    
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=args.test_batch_size, shuffle=True)
    
    #create Neural Network Object
    model = MNIST_CNN().to(device)
    #print the model summery
    print(model)
    #print parameter count
    print('\nParameter count:', count_parameters(model), '\n')
    
    #Visualize the output of each layer via torchSummary
    summary(model, (1, 28, 28))
    
    #choose the optimizer 
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    #Start Training
    for epoch in range(1, args.epochs+1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
    
    #Save the trained model(which means parameters)
    if(args.save_model):
        torch.save(model.state_dict(), "tp1_cnn_mnist")
        
if __name__ == '__main__':
    main()
    







