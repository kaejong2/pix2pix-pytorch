
import os, sys
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import itertools

from utils import *
from model import Generator, Discriminator
from dataloader import data_loader


class pix2pix():
    def __init__(self, args):
        self.args = args
        
        self.G = Generator(3).to(device=self.args.device)
        self.D = Discriminator(3).to(device=self.args.device)
        
        init_weight(self.G, init_type="normal", init_gain=0.02)
        init_weight(self.D, init_type="normal", init_gain=0.02)


        self.optimizerG = optim.Adam(self.G.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        self.optimizerD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        
        self.criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device=args.device)
        self.criterion_L1 =  torch.nn.L1Loss().to(device=args.device)

        transform_train = transforms.Compose([Normalize(), RandomFlip(), Rescale((286, 286)), RandomCrop((256, 256)), ToTensor()])
        transform_val = transforms.Compose([Normalize(), ToTensor()])
        self.transform_inv = transforms.Compose([ToNumpy(), Denomalize()])

        dataset_train = data_loader(args.data_path+"train", transform=transform_train)
        dataset_val = data_loader(args.data_path+"val", transform=transform_val)

        self.data_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
        self.data_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)
        
        self.writer_train = SummaryWriter(log_dir=args.log_path+"train")
        self.writer_test = SummaryWriter(log_dir=args.log_path+"test")


    def train(self, ckpt_path=None, result_path=None):
        for epoch in range(self.args.epoch, self.args.num_epochs):
            self.G.train()
            self.D.train()
            for _iter, input in enumerate(self.data_train,1):
                data = input['dataA'].to(device=self.args.device)
                label = input['dataB'].to(device=self.args.device)
                output = self.G(data)

                #################################################
                #              Train Discriminator
                #################################################
                set_requires_grad(self.D, True)
                self.optimizerD.zero_grad()

                real = torch.cat((data, label), dim=1)
                fake = torch.cat((data, output), dim=1)

                pred_real = self.D(real)
                pred_fake = self.D(fake.detach())

                loss_D_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))
                loss_D_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
                loss_D = 0.5 * (loss_D_fake + loss_D_real)

                loss_D.backward()
                self.optimizerD.step()
                
                fake = torch.cat((data, output), dim=1)
                set_requires_grad(self.D, False)
                #################################################
                #              Train Generator
                #################################################
                self.optimizerG.zero_grad()

                pred_fake = self.D(fake)

                loss_G_GAN = self.criterion_GAN(pred_fake, torch.ones_like(pred_real))
                loss_G_l1 = self.criterion_L1(output, label)
                loss_G = loss_G_GAN + (100*loss_G_l1)

                loss_G.backward()
                self.optimizerG.step()

                #################################################
                #              Training Process
                #################################################               
                print("Train : Epoch %02d/ %02d | Batch %03d / %03d | "
                       "Generator GAN %.4f | "
                       "Generator L1 %.4f | "
                       "Discriminator Real %.4f | "
                       "Discriminator Fake %.4f | " %
                       (epoch, self.args.num_epochs, _iter, len(self.data_train),
                       loss_G_GAN, loss_G_l1,
                       loss_D_real, loss_D_fake))
        
            if epoch % 100==0:
                print("Tensorboard")
                pix2pix_save(ckpt_path, self.G, self.D, self.optimizerG, self.optimizerD, epoch)
                data = self.transform_inv(data)
                output = self.transform_inv(output)
                label = self.transform_inv(label)
                self.writer_train.add_images('input', data, self.args.num_epochs * (epoch - 1) + _iter, dataformats='NHWC')
                self.writer_train.add_images('output', output, self.args.num_epochs * (epoch - 1) + _iter, dataformats='NHWC')
                self.writer_train.add_images('label', label, self.args.num_epochs * (epoch - 1) + _iter, dataformats='NHWC')       

    def test(self, ckpt_path=None, result_path=None):
        pix2pix_load(ckpt_path, self.G, self.D, self.optimizerG, self.optimizerD, epoch=1)
        with torch.no_grad():
            self.G.eval()
            for _iter, input in enumerate(self.data_train):
                data = input['dataA'].to(device=self.args.device)
                label = input['dataB'].to(device=self.args.device)
                output = self.G(data)

                data = transform_inv(data)
                output = transform_inv(output)
                label = transform_inv(label)

                for j in range(data.shape[0]):
                    name = args.batch_size * (_iter - 1) + j
                    fileset = {'name': name,
                                'input': "%04d-input.png" % name,
                                'output': "%04d-output.png" % name,
                                'label': "%04d-label.png" % name}

                    plt.imsave(os.path.join(result_path, fileset['input']), input[j, :, :, :].squeeze())
                    plt.imsave(os.path.join(result_path, fileset['output']), output[j, :, :, :].squeeze())
                    plt.imsave(os.path.join(result_path, fileset['label']), label[j, :, :, :].squeeze())


if __name__ == '__main__':
    args = Arguments().parser().parse_args()

    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
        os.makedirs(args.log_path+"train")
        os.makedirs(args.log_path+"test")
                
    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
    model = pix2pix(args)
    
    model.train(ckpt_path=args.ckpt_path, result_path=args.result_path)

