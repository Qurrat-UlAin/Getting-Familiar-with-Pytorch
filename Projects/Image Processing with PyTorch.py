from torchvision import datasets
import torch 
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam


data='~/data/MNIST'
mnist=datasets.FashionMNIST(data, download=True, \
                            train=True)

tr_images=mnist.data 
tr_targets=mnist.targets

#now we might as well define classes like we did earlier ie inherit from nn.Module and then define our layers
#but first lets define the mnist class dataset
class MNISTDataset(Dataset):
    def __init__(self, x,y):
        x=x.float()/255
        x=x.view(-1,1,28,28)
        self.x,self.y=x,y
    
    def __getitem__(self, ix):
        x,y = self.x[ix], self.y[ix]
        return x,y
    def __len__(self):
        return len(self.x)
    

def get_model():
    model=nn.Sequential(
        nn.Conv2d(1,64,kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(64,128, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3200,256),
        nn.ReLU(),
        nn.Linear(256,10)

    )
    loss_fn=nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer

#summarize the mode: 
from torchsummary import summary
model, loss_fn, optimizer=get_model()
summary(model, input_size=(1, 28, 28))