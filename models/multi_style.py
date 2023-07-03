from architectures import *
import torch.nn as nn

""""
Models for stylization: Multimodal Style Transfer

style_subnet:

    RGB block

    L-Block

enhance_subnet

refine_subnet

"""


class StyleSubnet(nn.Module):
    def __init__(self):
        super(StyleSubnet, self).__init__()

        """
        Style_subnet:
        RGB block
        L-block
        conv_block
        """
        # Bilinear downsampling
        #self.downsample = nn.Upsample(size=256, mode='bilinear')

        # Transform to Grayscale
        self.togray = nn.Conv2d(3, 1, kernel_size=1, stride=1)
        w = torch.nn.Parameter(torch.tensor([[[[0.299]],
                                              [[0.587]],
                                              [[0.114]]]]))
        self.togray.weight = w

        # RGB Block
        self.rgb_conv1 = ConvLayer(3, 16, kernel_size=9, stride=1)
        self.rgb_in1 = InstanceNormalization(16)
        self.rgb_conv2 = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.rgb_in2 = InstanceNormalization(32)
        self.rgb_conv3 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.rgb_in3 = InstanceNormalization(64)
        self.rgb_res1 = ResidualBlock(64)
        self.rgb_res2 = ResidualBlock(64)
        self.rgb_res3 = ResidualBlock(64)

        # L Block
        self.l_conv1 = ConvLayer(1, 16, kernel_size=9, stride=1) #Because it takes the gray scaled image
        self.l_in1 = InstanceNormalization(16)
        self.l_conv2 = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.l_in2 = InstanceNormalization(32)
        self.l_conv3 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.l_in3 = InstanceNormalization(64)
        self.l_res1 = ResidualBlock(64)
        self.l_res2 = ResidualBlock(64)
        self.l_res3 = ResidualBlock(64)

        # Residual layers
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.res6 = ResidualBlock(128)

        # Upsampling Layers
        self.rezconv1 = ResizeConvLayer(128, 64, kernel_size=3, stride=1)
        self.in4 = InstanceNormalization(64)
        self.rezconv2 = ResizeConvLayer(64, 32, kernel_size=3, stride=1)
        self.in5 = InstanceNormalization(32)
        self.rezconv3 = ConvLayer(32, 3, kernel_size=3, stride=1)

        # Non-linearities
        self.relu = nn.ReLU()


    def forward(self, x):

        # Bilinear downsampling
        #x = self.downsample(x)

        # Resized input image is the content target
        resized_input_img = x.clone()

        # Get RGB and L image
        x_rgb = x
        with torch.no_grad(): x_l = self.togray(x.clone())

        # RGB Block

        ## RGB-Block comprises three strided convolutional layers (9 × 9, 3 × 3, 3 × 3 respectively, the latter two are used
        ## for downsampling) and three residual blocks [10]
        ##All non-residual convolutional layers are followed by instance normalization and ReLU nonlinearity
        y_rgb = self.relu(self.rgb_in1(self.rgb_conv1(x_rgb)))
        y_rgb = self.relu(self.rgb_in2(self.rgb_conv2(y_rgb)))
        y_rgb = self.relu(self.rgb_in3(self.rgb_conv3(y_rgb)))
        #Residual Blocks
        y_rgb = self.rgb_res1(y_rgb)
        y_rgb = self.rgb_res2(y_rgb)
        y_rgb = self.rgb_res3(y_rgb)

        # L Block

        y_l = self.relu(self.l_in1(self.l_conv1(x_l)))
        y_l = self.relu(self.l_in2(self.l_conv2(y_l)))
        y_l = self.relu(self.l_in3(self.l_conv3(y_l)))

        y_l = self.l_res1(y_l)
        y_l = self.l_res2(y_l)
        y_l = self.l_res3(y_l)

        #conv_block
        """
        Conv-Block is composed of three residual
        blocks, two resize-convolution layers for upsampling and
        the last 3 × 3 convolutional layer to obtain the output RGB image ˆy
        """

        # Concatenate blocks along the depth dimension
        y = torch.cat((y_rgb, y_l), 1)

        # Residuals
        y = self.res4(y)
        y = self.res5(y)
        y = self.res6(y)

        # Decoder conv_block
        y = self.relu(self.in4(self.rezconv1(y)))
        y = self.relu(self.in5(self.rezconv2(y)))
        y = self.rezconv3(y)


        # Clamp image to be in range [0,1] after denormalization
        y[0][0].clamp_((0-0.485)/0.299, (1-0.485)/0.299)
        y[0][1].clamp_((0-0.456)/0.224, (1-0.456)/0.224)
        y[0][2].clamp_((0-0.406)/0.225, (1-0.406)/0.225)

        return y, resized_input_img

class EnhanceSubnet(nn.Module):
    def __init__(self):
        super(EnhanceSubnet, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear') #outputs a interpolated image of 512x512

        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)   # size = 512
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)   # size = 256
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)   # size = 128
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        self.conv4 = ConvLayer(128, 256, kernel_size=3, stride=2)   # size = 64
        self.in4 = nn.InstanceNorm2d(256, affine=True)

        # Residual layers
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)
        self.res3 = ResidualBlock(256)
        self.res4 = ResidualBlock(256)
        self.res5 = ResidualBlock(256)
        self.res6 = ResidualBlock(256)

        # Upsampling Layers
        self.rezconv1 = ResizeConvLayer(256, 128, kernel_size=3, stride=1) 
        self.in5 = nn.InstanceNorm2d(128, affine=True)
        self.rezconv2 = ResizeConvLayer(128, 64, kernel_size=3, stride=1)
        self.in6 = nn.InstanceNorm2d(64, affine=True)
        self.rezconv3 = ResizeConvLayer(64, 32, kernel_size=3, stride=1)
        self.in7 = nn.InstanceNorm2d(32, affine=True)
        self.rezconv4 = ConvLayer(32, 3, kernel_size=9, stride=1)

        ##Just outputs the stylized image in 512

        # Non-linearities
        self.relu = nn.ReLU()

    def forward(self, X):
        X = self.upsample(X)
        # resized input image is the content target
        resized_input_img = X.clone()

        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.relu(self.in4(self.conv4(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.res6(y)
        y = self.relu(self.in5(self.rezconv1(y)))
        y = self.relu(self.in6(self.rezconv2(y)))
        y = self.relu(self.in7(self.rezconv3(y)))
        y = self.rezconv4(y)

        # Clamp image to be in range [0,1] after denormalization
        y[0][0].clamp_((0-0.485)/0.299, (1-0.485)/0.299)
        y[0][1].clamp_((0-0.456)/0.224, (1-0.456)/0.224)
        y[0][2].clamp_((0-0.406)/0.225, (1-0.406)/0.225)

        return y, resized_input_img
    

"""
the refine net consists of three convolutional layers, three residual blocks, two resize-convolution layers and
one last convolutional layer to obtain the final output, which
is much shallower than the style and enhance subnet.
"""    
class RefineSubnet(nn.Module):
    def __init__(self):
        super(RefineSubnet, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear') #outputs a interpolated image of 1024x1024

        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1) 
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)

        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)

        # Upsampling Layers
        self.rezconv1 = ResizeConvLayer(128, 64, kernel_size=3, stride=1)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.rezconv2 = ResizeConvLayer(64, 32, kernel_size=3, stride=1)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.rezconv3 = ConvLayer(32, 3, kernel_size=3, stride=1)

        # Non-linearities
        self.relu = nn.ReLU()

    def forward(self, X):
        in_X = X
        #if self.training == False: in_X = self.upsample(in_X)   # Only apply upsampling during test

        # resized input image is the content target
        resized_input_img = in_X.clone()

        y = self.relu(self.in1(self.conv1(in_X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.relu(self.in4(self.rezconv1(y)))
        y = self.relu(self.in5(self.rezconv2(y)))
        y = self.rezconv3(y)
        y = y + resized_input_img

        # Clamp image to be in range [0,1] after denormalization
        y[0][0].clamp_((0-0.485)/0.299, (1-0.485)/0.299)
        y[0][1].clamp_((0-0.456)/0.224, (1-0.456)/0.224)
        y[0][2].clamp_((0-0.406)/0.225, (1-0.406)/0.225)

        return y, resized_input_img