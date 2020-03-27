import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

##############################################################

#                           VAE

##############################################################
class Down_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bn=True, dropout=0.0, conv8 = True):
        super(Down_Block,self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))

        if conv8:
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False))
            if bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
            if dropout:
                layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)

class Up_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout=0.0, conv=False):
        super(Up_Block, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        if conv:
            layers.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))

        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)

class Flatten(nn.Module):
    def forward(self,x):
        return x.view(x.shape[0],-1)
class UnFlatten(nn.Module):
    def forward(self,x):
        return x.view(x.shape[0],256,1,1)

class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()
        #encoder
        self.down1 = Down_Block(3,32)
        self.down2 = Down_Block(32,64)
        self.down3 = Down_Block(64,128)
        self.down4 = Down_Block(128,256)
        self.down5 = Down_Block(256,256,dropout=0.5)
        self.down6 = Down_Block(256,256,dropout=0.5)
        self.down7 = Down_Block(256,256,dropout=0.5)
        self.down8 = Down_Block(256,256,kernel_size=2, stride=1, padding=0, conv8=False)

        self.Flatten = Flatten()
        self.mu = nn.Linear(256,32)
        self.log_var = nn.Linear(256,32)

        self.fc = nn.Linear(32,256)
        self.UnFlatten = UnFlatten()
        self.up1 = Up_Block(256,256)
        self.up2 = Up_Block(256,256, dropout=0.5)
        self.up3 = Up_Block(256,256, dropout=0.5)
        self.up4 = Up_Block(256,256, dropout=0.5)
        self.up5 = Up_Block(256,128, conv=True)
        self.up6 = Up_Block(128,64, conv=True)
        self.up7 = Up_Block(64,32, conv=True)
        self.up8 = Up_Block(32,32, conv=True)

        self.final_conv = nn.Conv2d(32,3,kernel_size=1,stride=1,padding=0)

    def encode(self,x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.down6(x)
        x = self.down7(x)
        x = self.down8(x)
        x = self.Flatten(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        std = std.cuda()
        esp = torch.randn(*mu.size())
        esp = esp.cuda()
        z = mu + std * esp
        return z

    def decode(self, z):
        z = self.UnFlatten(z)
        z = self.up1(z)
        z = self.up2(z)
        z = self.up3(z)
        z = self.up4(z)
        z = self.up5(z)
        z = self.up6(z)
        z = self.up7(z)
        z = self.up8(z)
        z = self.final_conv(z)
        out = F.sigmoid(z)

        return out

    def forward(self,x):
        mu,log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z = self.fc(z)

        return self.decode(z), mu, log_var

    def loss_fuction(self, recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

        KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        return BCE + KLD, BCE, KLD


##############################################################

#                    pix2pix - Generator

##############################################################
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)



##############################################################

#                    pix2pix - Discriminator

##############################################################r


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
