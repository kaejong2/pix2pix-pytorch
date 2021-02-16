import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, stride=1, norm=None, act=None, drop=None):
        super(Encoder, self).__init__()

        block = []
        block += [nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]

        if not norm is False:
            if norm == "Batch":
                block += [nn.BatchNorm2d(out_features)]
            elif norm == "Instance":
                block += [nn.InstanceNorm2d(out_features)]
        if not act is False:
            if act == "ReLU":
                block += [nn.ReLU(inplace=True)]
            elif act == "LeakyReLU":
                block += [nn.LeakyReLU(0.2, inplace=True)]
            elif act == "Tanh":
                block += [nn.Tanh()]
            elif act == "Sigmoid":
                block += [nn.Sigmoid()]
        if not drop is False:
            block += [nn.Dropout2d(0.5)]
        
        self.layer = nn.Sequential(*block)

    def forward(self, x):
        return self.layer(x)


class Decoder(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, stride=1, norm=None, act=None, drop=False):
        super(Decoder, self).__init__()

        block = []
        block += [nn.ConvTranspose2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]

        if not norm is False:
            if norm == "Batch":
                block += [nn.BatchNorm2d(out_features)]
            elif norm == "Instance":
                block += [nn.InstanceNorm2d(out_features)]
        if not act is False:
            if act == "ReLU":
                block += [nn.ReLU(inplace=True)]
            elif act == "LeakyReLU":
                block += [nn.LeakyReLU(0.2,inplace=True)]
            elif act == "Tanh":
                block += [nn.Tanh()]
            elif act == "Sigmoid":
                block += [nn.Sigmoid()]
        if not drop is False:
            block += [nn.Dropout2d(0.5)]
        
        self.layer = nn.Sequential(*block)

    def forward(self,x):
        return self.layer(x)

class Generator(nn.Module):
    def __init__(self, input_channels):
        super(Generator, self).__init__()
        self.enc1 = Encoder(in_features=input_channels, out_features=64, kernel_size=4, padding=1, stride=2,
                               norm=False, act="LeakyReLU", drop=False)
        self.enc2 = Encoder(in_features=64, out_features=128, kernel_size=4, padding=1, stride=2,
                               norm="Instance", act="LeakyReLU", drop=False)
        self.enc3 = Encoder(in_features=128, out_features=256, kernel_size=4, padding=1, stride=2,
                               norm="Instance", act="LeakyReLU", drop=False)
        self.enc4 = Encoder(in_features=256, out_features=512, kernel_size=4, padding=1, stride=2,
                               norm="Instance", act="LeakyReLU", drop=False)
        self.enc5 = Encoder(in_features=512, out_features=512, kernel_size=4, padding=1, stride=2,
                               norm="Instance", act="LeakyReLU", drop=False)
        self.enc6 = Encoder(in_features=512, out_features=512, kernel_size=4, padding=1, stride=2,
                               norm="Instance", act="LeakyReLU", drop=False)
        self.enc7 = Encoder(in_features=512, out_features=512, kernel_size=4, padding=1, stride=2,
                               norm="Instance", act="LeakyReLU", drop=False)
        self.enc8 = Encoder(in_features=512, out_features=512, kernel_size=4, padding=1, stride=2,
                               norm="Instance", act="ReLU", drop=False)
       
        self.dec1 = Decoder(in_features=512, out_features=512, kernel_size=4, padding=1, stride=2,
                               norm="Instance", act="ReLU", drop=True)
        self.dec2 = Decoder(in_features=2*512, out_features=512, kernel_size=4, padding=1, stride=2,
                               norm="Instance", act="ReLU", drop=True)
        self.dec3 = Decoder(in_features=2*512, out_features=512, kernel_size=4, padding=1, stride=2,
                               norm="Instance", act="ReLU", drop=True)
        self.dec4 = Decoder(in_features=2*512, out_features=512, kernel_size=4, padding=1, stride=2,
                               norm="Instance", act="ReLU", drop=False)
        self.dec5 = Decoder(in_features=2*512, out_features=256, kernel_size=4, padding=1, stride=2,
                               norm="Instance", act="ReLU", drop=False)
        self.dec6 = Decoder(in_features=2*256, out_features=128, kernel_size=4, padding=1, stride=2,
                               norm="Instance", act="ReLU", drop=False)
        self.dec7 = Decoder(in_features=2*128, out_features=64, kernel_size=4, padding=1, stride=2,
                               norm="Instance", act="ReLU", drop=False)
        self.dec8 = Decoder(in_features=2*64, out_features=3, kernel_size=4, padding=1, stride=2,
                               norm=False, act='Tanh', drop=False)


    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)
        x8 = self.enc8(x7)

        x_8 = self.dec1(x8)
        x_7 = self.dec2(torch.cat((x7,x_8), dim=1))
        x_6 = self.dec3(torch.cat((x6,x_7), dim=1))
        x_5 = self.dec4(torch.cat((x5,x_6), dim=1))
        x_4 = self.dec5(torch.cat((x4,x_5), dim=1))
        x_3 = self.dec6(torch.cat((x3,x_4), dim=1))
        x_2 = self.dec7(torch.cat((x2,x_3), dim=1))
        x_1 = self.dec8(torch.cat((x1,x_2), dim=1))
        
        x = torch.tanh(x_1)
        return x

class Discriminator(nn.Module):
    def __init__(self, output_channels):
        super(Discriminator, self).__init__()
        self.layer1 = Encoder(in_features=3*2, out_features=64, kernel_size=4, padding=1, stride=2,
                               norm=None, act="LeakyReLU", drop=None)
        self.layer2 = Encoder(in_features=64, out_features=128, kernel_size=4, padding=1, stride=2,
                               norm="Instance", act="LeakyReLU", drop=None)
        self.layer3 = Encoder(in_features=128, out_features=256, kernel_size=4, padding=1, stride=2,
                               norm="Instance", act="LeakyReLU", drop=None)
        self.layer4 = Encoder(in_features=256, out_features=512, kernel_size=4, padding=1, stride=2,
                               norm="Instance", act="LeakyReLU", drop=None)
        self.RF_pad = nn.ZeroPad2d((1,0,1,0))
        self.final_layer = nn.Conv2d(512, 1, 4, padding=1, bias=False)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.RF_pad(x)
        x = self.final_layer(x)
     
        return x        

