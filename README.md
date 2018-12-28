# Neural style transfer

This is a simple pytorch implementation of [neural style transfer algorithm by Gatis, Ecker and Bethge](https://arxiv.org/abs/1508.06576). This implementation uses pretrained [VGG-19 network](https://pytorch.org/docs/stable/torchvision/models.html#id2). It uses intermediate layers
(1, 6, 11, 20, 29) with weights (0.5, 1, 1.5, 3, 4) to compute the style loss and layer 22 to compute content loss. It starts with the initial image equal content image as the origin and minimize the

    total loss = 0.0001 * content_loss+1000 * style_loss

It is [often suggested](https://discuss.pytorch.org/t/pytorch-tutorial-for-neural-transfert-of-artistic-style/336/20?u=alexis-jacq) to use LBFGS algorithm for optimization. An excellent comparison of the optimization speed for different optimizers can be found [here](https://blog.slavv.com/picking-an-optimizer-for-style-transfer-86e7b8cba84b). In the next section we will see that even when optimizers achieve the same value of total loss, the results may differ significantly.

## Comparison of optimizers
#### 1. Content and style image have similar palette
    Content image:                        Style image:

<img style="float: left;" src="https://raw.githubusercontent.com/neshitov/style-transfer/master/church.jpg" width="20%">      <img src="https://raw.githubusercontent.com/neshitov/style-transfer/master/vangogh.jpg" width="20%">

Results obtained after training with different optimizers and similar total loss:

    Adam optimizer, loss=36:              LBFGS optimizer, loss=33:

Although two algorithms achieve similar values of loss function, the image adjusted with Adam optimizer seems closer to the sytle image. The LBFGS algorithm shows a similar result when it achieves loss function equal to 22.

<img style="float: left;" src="https://raw.githubusercontent.com/neshitov/style-transfer/master/Adam-output-350epochs-loss36.62.png" width="40%"> <img src="https://raw.githubusercontent.com/neshitov/style-transfer/master/LBFG-output-4epochs-loss33.png" width="40%">

#### 2. Content image and style image have different palettes (images taken from [pytorch tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)):

<img style="float: left;" src="https://raw.githubusercontent.com/neshitov/style-transfer/master/dancing.jpg" width="20%"> <img src="https://raw.githubusercontent.com/neshitov/style-transfer/master/picasso.jpg" width="20%">

    Adam optimizer, loss=15:              LBFGS optimizer, loss=15:

<img style="float: left;" src="https://raw.githubusercontent.com/neshitov/style-transfer/master/dancing-Adam-output-400epochs-loss14.46.png" width="40%"> <img src="https://raw.githubusercontent.com/neshitov/style-transfer/master/dancing-LBFG-output-5epochs-loss15.png" width="40%">
Here we see that Adam optimizer picks more color from the style image.
