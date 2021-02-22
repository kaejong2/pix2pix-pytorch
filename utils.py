import random
import time
import datetime
import sys
import os
from torch.nn import init
from torch.autograd import Variable
import torch
import torchvision
import numpy as np
from torch.optim import lr_scheduler
from torchvision.utils import make_grid


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


def pix2pix_save(args, G, D, optimizerG, optimizerD, epoch):
    save_path = os.path.join(args.root_path, args.ckpt_path)
    
    torch.save({'G' : G.state_dict(), 'D': D.state_dict(), 'optimG':optimizerG.state_dict(), 'optimD': optimizerD.state_dict()},
    "%s/model_epoch%d.pth" % (save_path,epoch))


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



def sample_images(args, batches_done, G, dataloader):
    save_path = os.path.join(args.root_path, args.result_path+"_test")

    imgs = next(iter(dataloader))
    data = imgs['data_img'].to(device=args.device)
    label = imgs['label_img'].to(device=args.device)
    fake = G(data)
    data = make_grid(data, nrow=7, normalize=True)
    fake = make_grid(fake, nrow=7, normalize=True)
    label = make_grid(label, nrow=7, normalize=True)

    result = (torch.cat((data.data, fake.data, label.data),1))
    torchvision.utils.save_image(result, save_path+'/sample'+str(batches_done)+'.jpg', nrow=3, normalize=False)
