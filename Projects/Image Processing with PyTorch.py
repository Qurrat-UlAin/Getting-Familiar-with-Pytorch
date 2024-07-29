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

#lets have a look at an image
preds=[]
ix=24301
for px in range(-5,6):
    img=tr_images[ix]/225
    img=img.view(28,28)
    img2=np.roll(img, px, axis=1)
    plt.imshow(img2)
    plt.show()
    img3=torch.Tensor(img2).view(-1,1,28,28)
    np_output=model(img3).cpu().detach().numpy()
    preds.append(np.exp(np_output)/np.sum(np.exp(np_output)))

#displaying porobabilty using a heatmap
import seaborn as sns 
fig,ax=plt.subplots(1,1,figsize=(12,10))
plt.title("Probabaility of each class for various translations")
sns.heatmap(np.array(preds).reshape(11,10), annot=True, 
            ax=ax, fmt='2f', xticklabels=mnist.classes, 
            yticklabels=[str(i)+str('pixels') for i in range(-5,6)],
            cmap='gray'
            )


#Now we should augment the image for a) learning pytorch b)we have some predictions from the heatmap
#and although they are correct, this does not solve the problem completley 

#That is why image augmentation comes in place.
#  We can create more images from a given image. 
# Each of the created images can vary in terms of rotation, translation, scale , 
# noise and brightness. Furthermore, the extent of the variation in each of these 
# parameters can also vary (for example, the translation of a certain image in a given iteration can be +10 pixels while in different iteration, it can be -5 pixels)

plt.imshow(tr_images[2])
from imgagug import augmenters as iaa
aug=iaa.Affine(scale=2)

plt.imshow(aug.augment_image(tr_images[2]))
plt.title("scaled image")

aug = iaa.Affine(translate_px=10)
plt.imshow(aug.augment_image(tr_images[0]))
plt.title('Translated image by 10 pixels')

aug = iaa.Affine(translate_px={'x':10, 'y':2})
plt.imshow(aug.augment_image(tr_images[0]))
plt.title('Translation of 10 pixels \naccross columns \nand 2 pixels over rows')

# We will see te output when we scale, translate, rotate and shear the image
plt.figure(figsize=(20,20))
plt.subplot(161)
plt.imshow(tr_images[0])
plt.title('Original image')
plt.subplot(162)
aug = iaa.Affine(scale=2, fit_output=True)
plt.imshow(aug.augment_image(tr_images[0]))
plt.title('Scaled image')
plt.subplot(163)
aug = iaa.Affine(translate_px={'x':10, 'y':2}, fit_output=True)
plt.imshow(aug.augment_image(tr_images[0]))
plt.title('Translation of 10 pixels accross \ncolumns and \n2 pixels over rows')
plt.subplot(164)
aug = iaa.Affine(rotate=60, fit_output=True)
plt.imshow(aug.augment_image(tr_images[0]))
plt.title('Rotation of image \nby 60 degrees')
plt.subplot(165)
aug = iaa.Affine(shear=25, fit_output=True)
plt.imshow(aug.augment_image(tr_images[0]))
plt.title('Shear the image \nby 25 degrees')

aug = iaa.Affine(rotate=60, fit_output=True,cval=255)
plt.imshow(aug.augment_image(tr_images[0]))
plt.title('Rotation of image \nby 60 degrees')

plt.figure(figsize=(20,20))
plt.subplot(141)
aug = iaa.Affine(rotate=(-45,45), fit_output=True, cval=0, \
                 mode='constant')
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.subplot(142)
aug = iaa.Affine(rotate=(-45,45), fit_output=True, cval=0, \
                 mode='constant')
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.subplot(143)
aug = iaa.Affine(rotate=(-45,45), fit_output=True, cval=0, \
                 mode='constant')
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.subplot(144)
aug = iaa.Affine(rotate=(-45,45), fit_output=True, cval=0, \
                 mode='constant')
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')

aug = iaa.Multiply(0.5)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray', \
           vmin= 0, vmax=255)
plt.title('Pixels multiplied by 0.5')

aug = iaa.LinearContrast(0.5)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray', \
           vmin= 0, vmax=255)
plt.title('Pixels multiplied by 0.5')

# Adding noise
# There 2 main methods that help us to simulate the grainy image. They are Dropout and SaltAndPepper
# In real world we will may face the bad image due to bad photography
plt.figure(figsize=(10,10))
plt.subplot(121)
aug = iaa.Dropout(p=0.2)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray', \
           vmin= 0, vmax=255)
plt.title('Random 30% of dropout')
plt.subplot(122)
aug = iaa.SaltAndPepper(0.2)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray', \
           vmin= 0, vmax=255)
plt.title('Random 30% salt and peper noise')

sequential = iaa.Sequential(
    [
     iaa.Dropout(p=0.2), iaa.Affine(rotate=(-30,30))
    ], random_order=True
)
plt.imshow(sequential.augment_image(tr_images[0]), cmap='gray', \
           vmin=0, vmax=255)
plt.title('Image augmented using a \nrandom order of the 2 augmentations')











