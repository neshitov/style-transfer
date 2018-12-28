import torch
from torch import nn, optim
import numpy as np
from torchvision import datasets, transforms, models
import cv2 as cv
import torch.nn.functional as F
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau

IMG_SIZE = 448
CONTENT_IMAGE_NAME = 'dancing.jpg'
STYLE_IMAGE_NAME = 'picasso.jpg'
# Layers of the network to compute the style_cost
FEATURE_LAYERS = [1, 6, 11, 20, 22, 29]
# Layer to compute content cost
CONTENT_LAYER = 22
STYLE_WEIGHTS = {1: 0.5, 6: 1.0, 11: 1.5, 20: 3., 22: 0, 29: 4.}


if torch.cuda.is_available():
    device = 'cuda'
    print('GPU availabale, device=cuda')
else:
    device = 'cpu'
    print('no GPU available, device=cpu')


def img_show(array):
    '''
    Input:
        array: np.array of shape (3,IMG_SIZE,IMG_SIZE) or (1,3,IMG_SIZE,IMG_SIZE) in the range (0,1)
    Otuput:
        shows the image
    '''
    assert (
        array.shape == (
            3, IMG_SIZE, IMG_SIZE) or array.shape == (
            1, 3, IMG_SIZE, IMG_SIZE)), 'got wrong input shape'
    array = np.squeeze(array)
    array = np.moveaxis(array, 0, -1)
    cv.imshow(' ', array)
    cv.waitKey(0)
    cv.destroyAllWindows()


def img_save(array, name):
    '''
    Input:
        array: np.array of shape (3,IMG_SIZE,IMG_SIZE) or (1,3,IMG_SIZE,IMG_SIZE) in the range (0,1)
        name: name of the file to save image
    Otuput:
        saves image to file 'name.png'
    '''
    assert (
        array.shape == (
            3, IMG_SIZE, IMG_SIZE) or array.shape == (
            1, 3, IMG_SIZE, IMG_SIZE)), 'got wrong input shape'
    array = np.squeeze(array)
    array = np.moveaxis(array, 0, -1) * 255
    cv.imwrite(name + '.png', array)


#  Load pretrained VGG model
vgg = models.vgg19(pretrained=True)
for param in vgg.parameters():
    param.requires_grad = False
vgg.eval()


def intermediate_features(x):
    '''
    Input:
        x: torch tensor of size (3,n_H,n_W), the input image
    Output:
        feat: a dictionary {i:output_i} where i is the layer number from
        FEATURE_LAYERS, output_i is the output of the i-th layer
    '''
    n_C, n_H, n_W = x.shape
    x = x.view(1, n_C, n_H, n_W)
    feat = {}
    for i, layer in enumerate(vgg.features):
        x = layer(x)
        if i in FEATURE_LAYERS:
            feat[i] = x
    return feat


def content_loss(content_image, image):
    '''
    Inputs:
        content_image, image: torch tensors representing content image and target image
    Outputs:
        content_loss: loss computed using the CONTENT_LAYER
    '''
    content_image_features = torch.squeeze(
        intermediate_features(content_image)[CONTENT_LAYER])
    image_features = torch.squeeze(intermediate_features(image)[CONTENT_LAYER])
    n_C, n_H, n_W = image_features.shape
    return (1 / (4 * n_C * n_H * n_W) *
            torch.sum(torch.pow(content_image_features - image_features, 2)))


def gram_matrix(x):
    '''
    Input:
        x: tensor x of shape (n_C,n_H,n_W)
    Output:
        gram_matrix: tensor of shape (n_C,n_C), the gram matrix of x
    '''
    n_C, n_H, n_W = x.shape
    x = x.view(n_C, n_H * n_W)
    xT = torch.transpose(x, 0, 1)
    return torch.mm(x, xT)


def style_cost(style_image, image, n):
    '''
    Inputs:
        style_image,image: tensors representint style_image and image
        n: layer number from FEATURE_LAYERS
    Outputs:
        style_cost: component of style cost computed on layer n
    '''
    style_image_features = torch.squeeze(intermediate_features(style_image)[n])
    image_features = torch.squeeze(intermediate_features(image)[n])
    n_C, n_H, n_W = image_features.shape
    style_image_gram = gram_matrix(style_image_features)
    image_gram = gram_matrix(image_features)
    return (1 / (4 * (n_C * n_H * n_W)**2) *
            torch.sum(torch.pow(style_image_gram - image_gram, 2)))


def total_style_cost(style_image, image):
    '''
    Inputs:
        style_image,image: tensors representint style_image and image
    Outputs:
        total_style_cost: weighted sum of style costs computed on FEATURE_LAYERS
        with weights given by STYLE_WEIGHTS
    '''
    ans = 0
    for i in FEATURE_LAYERS:
        ans += style_cost(style_image, image, i) * STYLE_WEIGHTS[i]
    return ans


def total_loss(image, content_image, style_image, alpha=1, beta=1):
    '''weighted sum of content loss and style loss'''
    return alpha * content_loss(content_image, image) + \
        beta * total_style_cost(style_image, image)


# read the images, resize and transform to torch tensors of shape
# (3,IMG_SIZE,IMG_SIZE) in range(0,1)
content_image = cv.imread(CONTENT_IMAGE_NAME)
content_image = cv.resize(content_image, (IMG_SIZE, IMG_SIZE))
style_image = cv.imread(STYLE_IMAGE_NAME)
style_image = cv.resize(style_image, (IMG_SIZE, IMG_SIZE))
content_image = torch.from_numpy(content_image).float() / 255
style_image = torch.from_numpy(style_image).float() / 255
content_image = content_image.permute(2, 0, 1)
style_image = style_image.permute(2, 0, 1)

# transformation needed to normalize images for pretrained network
# https://pytorch.org/docs/stable/torchvision/models.html
# and its inverse transformation
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
denormalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

# normalize images
content_image = normalize(content_image)
style_image = normalize(style_image)

# move to gpu if possible
content_image = content_image.to(device)
style_image = style_image.to(device)

# start with a copy of the content image, one for Adam optimizer and one
# for LBFG
image_LBFG = content_image.clone().detach()
image_LBFG.requires_grad = True
image_Adam = content_image.clone().detach()
image_Adam.requires_grad = True
vgg = vgg.to(device)

# set up optimizers
optimizer_LBFG = optim.LBFGS([image_LBFG])
optimizer_Adam = optim.Adam([image_Adam], lr=0.1)
scheduler = ReduceLROnPlateau(
    optimizer_Adam,
    mode='min',
    factor=0.1,
    patience=30,
    verbose=True,
    threshold=0.00001,
    threshold_mode='rel',
    cooldown=0,
    min_lr=0,
    eps=1e-08)

# adjusting image with LBFG optimizer
for epoch in range(6):
    save_loss = 0

    def closure():
        optimizer_LBFG.zero_grad()
        loss = total_loss(
            image_LBFG,
            content_image,
            style_image,
            alpha=0.0001,
            beta=1000)
        loss.backward(retain_graph=True)
        print('epoch:', epoch, ', loss_LBFG:', loss)
        save_loss = loss.item()
        return loss
    optimizer_LBFG.step(closure)
    # if epoch%10==0:
    forsave = image_LBFG.clone().detach()
    forsave = denormalize(forsave)
    forsave = forsave.to('cpu').numpy()
    img_save(forsave,'dancing-LBFG-output-' +str(epoch) +
        'epochs-loss' + str(save_loss)[0:5])

# adjusting image with Adam optimizer
for epoch in range(401):
    optimizer_Adam.zero_grad()
    loss = total_loss(
        image_Adam,
        content_image,
        style_image,
        alpha=0.0001,
        beta=1000)
    print('epoch:', epoch, ', loss_Adam:', loss)
    loss.backward(retain_graph=True)
    scheduler.step(loss)
    optimizer_Adam.step()
    if epoch % 50 == 0:
        forsave = image_Adam.clone().detach()
        forsave = denormalize(forsave)
        forsave = forsave.to('cpu').numpy()
        img_save(forsave, 'dancing-Adam-output-' + str(epoch) +
                 'epochs-loss' + str(loss.item())[0:5])
