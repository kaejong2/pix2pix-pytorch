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

from PIL import Image

from model import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] ="3"

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default = 0, help="dsa")
parser.add_argument("--n_epochs", type=int, default = 10000, help="dsa")
parser.add_argument("--dataset_name", type=str, default = "retinal1", help="dsa")
parser.add_argument("--batch_size", type=int, default = 20, help="dsa")
parser.add_argument("--lr", type=float, default = 1e-3, help="dsa")
parser.add_argument("--b1", type=float, default = 0.5, help="adam : decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default = 0.999, help="adam : decay of second order momentum of gradient")
parser.add_argument("--decay", type=int, default = 100, help="epoch from which to start lr decay")
parser.add_argument("--img_height", type=int, default = 256, help="dsa")
parser.add_argument("--img_width", type=int, default = 256, help="dsa")
parser.add_argument("--channels", type=int, default = 3, help="dsa")
parser.add_argument(
    "--sample_interval", type=int, default=10, help="interval between sampling of images from generators"
)

cuda = True if torch.cuda.is_available() else False
opt = parser.parse_args()
print(opt)


model = VAE()
model.load_state_dict(torch.load('model/VAE_0325.pth'))

if cuda:
    model.cuda()

dataset_A = datasets.ImageFolder(root='data/A', transform=transforms.Compose([transforms.ToTensor()]))
dataloader_A = torch.utils.data.DataLoader(dataset_A, batch_size=4, shuffle=False)


a, _ = next(iter(dataloader_A))
print(a)

save_image(a,'real_image.png')

a = a.cuda()
recon, mu, log_var = model(a)
print(recon)
save_image(recon,'fake_image.png')
img_real = Image.open('real_image.png')
img_fake = Image.open('fake_image.png')

img_real.show()
img_fake.show()