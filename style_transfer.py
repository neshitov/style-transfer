import torch
from torch import nn, optim
import numpy as np
from torchvision import datasets, transforms, models
import cv2 as cv
import torch.nn.functional as F

IMG_SIZE=224
###outline
'''
1)G=noise image
initialise G

2) compute content Loss
3) compute style Loss
4) optimize loss.
'''

def img_show(array):
    '''
    Input:
        array: np.array of shape (3,IMG_SIZE,IMG_SIZE) or (1,3,IMG_SIZE,IMG_SIZE) in the range (0,1)
    Otuput:
        shows the image
    '''
    assert (array.shape==(3,IMG_SIZE,IMG_SIZE) or array.shape==(1,3,IMG_SIZE,IMG_SIZE)), 'got wrong input shape'
    array=np.squeeze(array)
    array=np.moveaxis(array,0,-1)
    cv.imshow(' ',array)
    cv.waitKey(0)
    cv.destroyAllWindows()
def img_save(array,name):
    '''
    Input:
        array: np.array of shape (3,IMG_SIZE,IMG_SIZE) or (1,3,IMG_SIZE,IMG_SIZE) in the range (0,1)
        name: name of the file to save image
    Otuput:
        saves image to file
    '''
    assert (array.shape==(3,IMG_SIZE,IMG_SIZE) or array.shape==(1,3,IMG_SIZE,IMG_SIZE)), 'got wrong input shape'
    array=np.squeeze(array)
    array=np.moveaxis(array,0,-1)*255
    cv.imwrite(name+'.png',array)


vgg=models.vgg19(pretrained=True)
for param in vgg.parameters():
    param.requires_grad=False

feature_layers=[11,20,29]
content_layer=29
style_weights={11:0.33,20:0.33,29:0.33}

#print(vgg.features)

def intermediate_features(x):
    n_C,n_H,n_W=x.shape
    x=x.view(1,n_C,n_H,n_W)
    ans={}
    for i,layer in enumerate(vgg.features):
        x=layer(x)
        if i in feature_layers:
            ans[i]=x
    return ans



def content_loss(content_image,image):
    '''
    Inputs:
        n: number of layer to compute content loss
    '''
    content_image_features=torch.squeeze(intermediate_features(content_image)[content_layer])
    image_features=torch.squeeze(intermediate_features(image)[content_layer])
    n_C, n_H,n_W=image_features.shape
    return (torch.nn.MSELoss(size_average=False)(content_image_features,image_features)*1/(4*n_C*n_H*n_W))

def gram_matrix(x):
    '''
    Input:
        x: tensor x of shape (n_C,n_H,n_W)
    Output: tensor of shape (n_C,n_C)
    '''
    n_C,n_H,n_W=x.shape
    x=x.view(n_C,n_H*n_W)
    xT=torch.transpose(x,0,1)
    return torch.mm(x,xT)

def style_cost(style_image,image,n):
    '''
    Inputs:
    '''
    style_image_features=torch.squeeze(intermediate_features(style_image)[n])
    image_features=torch.squeeze(intermediate_features(image)[n])
    n_C,n_H,n_W=image_features.shape
    style_image_gram=gram_matrix(style_image_features)
    image_gram=gram_matrix(image_features)
    return (torch.nn.MSELoss(size_average=False)(style_image_gram,image_gram)*1/(4*(n_C*n_H*n_W)**2))

def total_style_cost(style_image,image):
    ans=0
    for i in feature_layers:
        ans+=style_cost(style_image,image,i)*style_weights[i]
    return ans

def total_loss(image, content_image,style_image,alpha=1,beta=1):
    return alpha*content_loss(content_image,image)+beta*total_style_cost(style_image,image)

content_image=cv.imread('photo-cropped.jpg')
content_image=cv.resize(content_image,(IMG_SIZE,IMG_SIZE))
style_image=cv.imread('filonov.jpg')
style_image=cv.resize(style_image,(IMG_SIZE,IMG_SIZE))
image=np.random.rand(3,IMG_SIZE,IMG_SIZE)
image=torch.from_numpy(image).float()

content_image=torch.from_numpy(content_image).float()/255
style_image=torch.from_numpy(style_image).float()/255
content_image=content_image.permute(2,0,1)
style_image=style_image.permute(2,0,1)

#normalize images for pretrained network
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
content_image=normalize(content_image)
style_image=normalize(style_image)
print(image.shape)
image=normalize(image)
print(content_image)
image.requires_grad=True

epochs=3
for epoch in range(epochs):
    loss=total_loss(image,content_image,style_image,alpha=1,beta=4)
    optimizer=optim.Adam([image])
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    print('loss',loss)


#def content_loss
#vgg=models.vgg19(pretrained=True)
# load pretrianed vgg19 model
# normalization for input of pretrainedd models
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
