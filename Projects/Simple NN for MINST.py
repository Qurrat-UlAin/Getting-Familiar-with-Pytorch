import pandas as pd 
import numpy as np 
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data 
import torch
from torch.autograd import Variable


df=pd.read_csv("/home/annie/PytorchPractice/exercise_files/pytorch_dataset/train.csv")
df.head(5)
df_labels=df['label'].values

df=df.drop('label',axis=1).values.reshape(len(df),1,28,28) #output is: len(df) number of images, each having 1 channel, 
#with height and width of each image being 28x28
# 
#convert the df and the labels into tensors
#X=torch.tensor(df.astype(float)) #x = torch.tensor(df.values).float()
#y=torch.tensor(df_labels).long()

X = torch.tensor(df, dtype=torch.float32) # Convert to float32 directly
y = torch.tensor(df_labels, dtype=torch.long) # Convert to long (int64)


class Classifier(nn.Module): 
    #nn.Module = all models in pytorch inherit from this class, it helps build nn
    def __init__(self): #constructor//here in init we define the layers of the nn 
        super().__init__() #initialozes the parent class//necessary
        #fc1 = full connected layer#1 
        self.fc1=nn.Linear(784,392) #input: 784=28*28, output=392, chosen arbitarily but kept smaller than input
        self.fc2=nn.Linear(392,196)
        self.fc3=nn.Linear(196,98)
        self.fc4=nn.Linear(98,10)  #final output is 10 because we have 10 classes to choose from
        self.dropout=nn.Dropout(p=0.2) #In forward pass each step, 20% of neurons will be set to zero randomly//regularization technique to reduce overfitting 
    #input tensor=x
    def forward(self,x):
        x=x.view(x.shape[0],-1) #reshape the input tensor, first paraemeter represent total batch size andd the second one represent the desired shape. 
        #here -1 means that pytorch will itself infer the best dimesnion
        x=self.dropout(F.relu(self.fc1(x)))  #apply relu activation on the first layer and so on and then apply dropout
        x=self.dropout(F.relu(self.fc2(x)))
        x=self.dropout(F.relu(self.fc3(x)))
        x=F.log_softmax(self.fc4(x),dim=1) #input size goes from 98 to 10 here.apply log softmax function on 98 and dim=1 means
        #the log must applied accross the secind dimension ie 10 
        return x

model=Classifier()
loss_func=nn.NLLLoss()
opt=optim.Adam(model.parameters(),lr=0.001)

for epoch in range(50):
    images=Variable(X)
    labels=Variable(y)
    opt.zero_grad()
    outputs=model(images)
    loss=loss_func(outputs,labels)
    loss.backward()
    opt.step()
    print('Epoch[%d/%d]LOss= %4f' %(epoch+1, 50, loss.data.item()))


#Now lwts test it out 
test=pd.read_csv("/home/annie/PytorchPractice/exercise_files/pytorch_dataset/test.csv")
test_lables=test['label'].values
test=test.drop("label",axis=1).values.reshape(len(test),1,28,28)


X_test = torch.tensor(test, dtype=torch.float32) # Convert to float32 directly
y_test = torch.tensor(test_lables, dtype=torch.long) # Convert to long (int64)


preds=model(X_test)
print(preds[0])

#now lets construct a datattframe
_, predictionlabel=torch.max(preds.data,1)
predictionlabel=predictionlabel.tolist()

predictionlabel=pd.Series(predictionlabel)
test_lables=pd.Series(test_lables)

pred_table=pd.concat([predictionlabel,test_lables],axis=1)
pred_table.columns=['Predicted value', 'TRue Value']
print(pred_table.head(10))

#lets evaluate our model
preds=len(predictionlabel)
correct=len([1 for x,y in zip(predictionlabel, test_lables) if x==y])
print((correct/preds)*100)