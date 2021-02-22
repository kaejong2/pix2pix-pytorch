
import torchvision.transforms as transforms
from PIL import Image
from utils import *
from dataloader import *
import os
from dataloader import data_loader
import torch
import torchvision
from model import Generator, Discriminator

def test(args):
    G = Generator(3).to(device=args.device)

    try:
        ckpt = pix2pix_load(args.ckpt_path, args.device)
        G.load_state_dict(ckpt['G'])
    except:
        print('Failed to load checkpoint')

    dataloader = data_loader(args, mode = 'test')
    for i in range(len(dataloader)):
       sample_images(args, i, G, dataloader)