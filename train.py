
import os, sys
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import itertools

from dataloader import *
from utils import *
from model import Generator, Discriminator
from dataloader import data_loader

class pix2pix():
    def __init__(self, args):
        self.args = args
        
        self.G = Generator(3).to(device=self.args.device)
        self.D = Discriminator(3).to(device=self.args.device)
        
        init_weight(self.G, init_type=self.args.init_weight, init_gain=0.02)
        init_weight(self.D, init_type=self.args.init_weight, init_gain=0.02)
        
        self.optimizerG = torch.optim.Adam(self.G.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        self.optimizerD = torch.optim.Adam(self.D.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        
        self.criterion_GAN = torch.nn.MSELoss().to(device=args.device)
        self.criterion_L1  = torch.nn.L1Loss().to(device=args.device)

        self.dataloader = data_loader(self.args, mode=args.mode)
        self.val_dataloader = data_loader(args, mode='test')

    def run(self, ckpt_path=None, result_path=None):
        self.G.train()
        self.D.train()
        for epoch in range(self.args.epoch+1, self.args.num_epochs):
            for _iter, imgs in enumerate(self.dataloader, 1):
                data = imgs['data_img'].to(device=self.args.device)
                label = imgs['label_img'].to(device=self.args.device)
                output = self.G(data)

                
                #################################################
                #              Train Generator
                #################################################
                self.optimizerG.zero_grad()

                fake = torch.cat((output, data), dim=1)
                pred_fake = self.D(fake.detach())

                loss_G_GAN = self.criterion_GAN(pred_fake, torch.ones_like(pred_fake))
                loss_G_l1 = self.criterion_L1(output, label)
                loss_G = loss_G_GAN + (100*loss_G_l1)

                loss_G.backward()
                self.optimizerG.step()

                #################################################
                #              Train Discriminator
                #################################################
                self.optimizerD.zero_grad()

                real = torch.cat((label, data), dim=1)
                fake = torch.cat((output, data), dim=1)

                pred_real = self.D(real)
                pred_fake = self.D(fake.detach())

                loss_D_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))
                loss_D_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
                loss_D = 0.5 * (loss_D_fake + loss_D_real)

                loss_D.backward()
                self.optimizerD.step()
                
                #################################################
                #              Training Process
                ################################################# 
                # sys.stdout.write("\r")            
                batches_done = epoch * len(self.dataloader) + _iter
                print("Train : Epoch %02d/ %02d | Batch %03d / %03d | "
                       "Generator GAN %.4f | "
                       "Generator L1 %.4f | "
                       "Discriminator Real %.4f | "
                       "Discriminator Fake %.4f | " %
                       (epoch, self.args.num_epochs, _iter, len(self.dataloader),
                       loss_G_GAN, loss_G_l1,
                       loss_D_real, loss_D_fake))
                if batches_done % self.args.sample_save ==0:
                    print("sample save")
                    sample_images(self.args, batches_done, self.G, self.val_dataloader)
            
            if epoch % 10==0:
                pix2pix_save(self.args, self.G, self.D, self.optimizerG, self.optimizerD, epoch)



