
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
    

    dataloader = data_loader(args, mode='test')
    
    with torch.no_grad():
        G.eval()
        for _iter, input in enumerate(dataloader):
            data = input['data_img'].to(device=args.device)
            label = input['label_img'].to(device=args.device)
            fake = G(data)

            result = (torch.cat([data, fake, label], dim=0).data + 1)/ 2.0

            torchvision.utils.save_image(result, args.result_path+'sample'+str(_iter)+'.jpg', nrow=4)