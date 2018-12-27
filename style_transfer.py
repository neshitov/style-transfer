import torch
from torch import nn, optim
import numpy as np
from torchvision import datasets, transforms, models
import cv2 as cv
import torch.nn.functional as F
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau

IMG_SIZE=224
###outline

if torch.cuda.is_available():
    device='cuda'
    print('GPU availabale, device=cuda')
else:
    device='cpu'
    print('no GPU, device=cpu')

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

vgg.eval()

feature_layers=[1,6,11,20,22,29]
content_layer=22
style_weights={1:0.5,6:1.0,11:1.5, 20:3.,22:0, 29:4.}

'''
print(vgg)
for i in feature_layers:
    print(vgg.features[i])
sys.exit()

their_model={
'input': <tf.Variable 'Variable:0' shape=(1, 300, 400, 3) dtype=float32_ref>,
'conv1_1': <tf.Tensor 'Relu:0' shape=(1, 300, 400, 64) dtype=float32>,
'conv1_2': <tf.Tensor 'Relu_1:0' shape=(1, 300, 400, 64) dtype=float32>,
'avgpool1': <tf.Tensor 'AvgPool:0' shape=(1, 150, 200, 64) dtype=float32>,
'conv2_1': <tf.Tensor 'Relu_2:0' shape=(1, 150, 200, 128) dtype=float32>,
'conv2_2': <tf.Tensor 'Relu_3:0' shape=(1, 150, 200, 128) dtype=float32>,
'avgpool2': <tf.Tensor 'AvgPool_1:0' shape=(1, 75, 100, 128) dtype=float32>,
'conv3_1': <tf.Tensor 'Relu_4:0' shape=(1, 75, 100, 256) dtype=float32>,
'conv3_2': <tf.Tensor 'Relu_5:0' shape=(1, 75, 100, 256) dtype=float32>,
'conv3_3': <tf.Tensor 'Relu_6:0' shape=(1, 75, 100, 256) dtype=float32>,
'conv3_4': <tf.Tensor 'Relu_7:0' shape=(1, 75, 100, 256) dtype=float32>,
'avgpool3': <tf.Tensor 'AvgPool_2:0' shape=(1, 38, 50, 256) dtype=float32>,
'conv4_1': <tf.Tensor 'Relu_8:0' shape=(1, 38, 50, 512) dtype=float32>,
'conv4_2': <tf.Tensor 'Relu_9:0' shape=(1, 38, 50, 512) dtype=float32>,
'conv4_3': <tf.Tensor 'Relu_10:0' shape=(1, 38, 50, 512) dtype=float32>,
'conv4_4': <tf.Tensor 'Relu_11:0' shape=(1, 38, 50, 512) dtype=float32>,
'avgpool4': <tf.Tensor 'AvgPool_3:0' shape=(1, 19, 25, 512) dtype=float32>,
'conv5_1': <tf.Tensor 'Relu_12:0' shape=(1, 19, 25, 512) dtype=float32>,
'conv5_2': <tf.Tensor 'Relu_13:0' shape=(1, 19, 25, 512) dtype=float32>,
'conv5_3': <tf.Tensor 'Relu_14:0' shape=(1, 19, 25, 512) dtype=float32>,
'conv5_4': <tf.Tensor 'Relu_15:0' shape=(1, 19, 25, 512) dtype=float32>,
'avgpool5': <tf.Tensor 'AvgPool_4:0' shape=(1, 10, 13, 512) dtype=float32>}

style layers= ('conv1_1', 0.2), 1 for us
    ('conv2_1', 0.2), 6 for us
    ('conv3_1', 0.2), 11 for us
    ('conv4_1', 0.2), 20 for us
    ('conv5_1'        29 for us
content=('conv 4_2') 22 for us
sys.exit()
'''


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
    return (1/(4*(n_C*n_H*n_W)**2)*torch.sum(torch.pow(style_image_gram-image_gram,2)))
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

content_image=torch.from_numpy(content_image).float()/255
style_image=torch.from_numpy(style_image).float()/255
content_image=content_image.permute(2,0,1)
style_image=style_image.permute(2,0,1)

#normalize images for pretrained network
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
denormalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[1/0.229, 1/0.224, 1/0.225])
content_image=normalize(content_image)
style_image=normalize(style_image)
#print(image.shape)
#image=normalize(image)
#image=torch.rand((3,IMG_SIZE,IMG_SIZE), requires_grad=True, device=device)
#image=normalize(image)
#image=image.detach()
#image.requires_grad=True

content_image=content_image.to(device)
style_image=style_image.to(device)
image=content_image.clone().detach()
image.requires_grad=True
vgg=vgg.to(device)
#optimizer=optim.SGD([image],lr=0.01)
optimizer=optim.LBFGS([image])
#optimizer=optim.Adam([image])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=30, verbose=True, threshold=0.00001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
epochs=400
for epoch in range(epochs+1):
    def closure():
        optimizer.zero_grad()
        loss=total_loss(image,content_image,style_image,alpha=0.000001,beta=1000)
        loss.backward(retain_graph=True)
        scheduler.step(loss)
        print('loss',loss)
        return loss

    optimizer.step(closure)


    if epoch%100==0:
        forsave=image.clone().detach()
        forsave=denormalize(forsave)
        forsave=forsave.to('cpu').numpy()
        img_save(forsave,'output-'+str(epoch))
