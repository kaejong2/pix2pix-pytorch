import random
import time
import datetime
import sys
import os
from torch.nn import init
from torch.autograd import Variable
import torch

import numpy as np
from torch.optim import lr_scheduler



def init_weight(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def pix2pix_save(ckpt_dir, G, D, optimizerG, optimizerD, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    torch.save({'G' : G.state_dict(), 'D': D.state_dict(), 'optimG':optimizerG.state_dict(), 'optimD': optimizerD.state_dict()},
    "%s/model_epoch%d.pth" % (ckpt_dir,epoch))


def pix2pix_load(ckpt_path, device):
    ckpt_lst = os.listdir(ckpt_path)
    ckpt_lst.sort()
    dict_model = torch.load('%s/%s'% (ckpt_path, ckpt_lst[-1]), map_location=device)
    print('Loading checkpoint from %s/%s succeed' % (ckpt_path, ckpt_lst[-1]))
    return dict_model

def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


def save_image(image_tensor):
    img = image_tensor.to('cpu').detach().numpy().transpose(0,2,3,1)
    img = img/2.0 *255.0
    img = img.clip(0,255)
    img = img.astype(np.uint8)
    
    return img


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad