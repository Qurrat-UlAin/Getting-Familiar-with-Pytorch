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
#output to expect: Comment it out 
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 26, 26]             640   #-1 means uncertain batch size, 64 output channels and 26x26 image size. 640 learnable parameters: (number of input channels * kernel height * kernel width + 1 (bias)) * number of output channels: (1 * 3 * 3 + 1) * 64 = 64
         MaxPool2d-2           [-1, 64, 13, 13]               0   # max pooling layer has no learnable parameters. 13x13 is height after pooling.
              ReLU-3           [-1, 64, 13, 13]               0   #
            Conv2d-4          [-1, 128, 11, 11]          73,856   # 128 is the output cannel. (64×3×3+1)×128=(576+1)×128=577×128=73,856 
         MaxPool2d-5            [-1, 128, 5, 5]               0   #
              ReLU-6            [-1, 128, 5, 5]               0   #
           Flatten-7                 [-1, 3200]               0   #
            Linear-8                  [-1, 256]         819,456   #
              ReLU-9                  [-1, 256]               0   #
           Linear-10                   [-1, 10]           2,570   #
================================================================
Total params: 896,522
Trainable params: 896,522
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.69
Params size (MB): 3.42
Estimated Total Size (MB): 4.11
----------------------------------------------------------------
#640: Parameters in the first Conv2d layer.
#73,856: Parameters in the second Conv2d layer.
#819,456: Parameters in the first Linear (fully connected) layer.
#2,570: Parameters in the second Linear (fully connected) layer.
#3200: Number of features after flattening, determined by the dimensions of the output from the last convolutional layer (after pooling and flattening).













