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

from model import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] ="3,4"

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default = 0, help="dsa")
parser.add_argument("--n_epochs", type=int, default = 10000, help="dsa")
parser.add_argument("--dataset_name", type=str, default = "retinal1", help="dsa")
parser.add_argument("--batch_size", type=int, default = 20, help="dsa")
parser.add_argument("--lr", type=float, default = 1e-2, help="dsa")
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)




def train(epoch, model, optimizer, train_loader):
    model.train()
    train_loss =0.0

    for idx, (x, _) in enumerate(train_loader):
        if cuda:
            x = x.cuda()

        recon, mu, log_var = model(x)
        loss, bce, kld = model.loss_fuction(recon, x, mu, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if idx % opt.sample_interval == 0:
            to_print = "Train Epoch: [{}/{} ({:.0f}%)]\t Loss: {:.6f} BCE: {:.3f} KLD: {:.3f}, Learning_rate: {}".format(epoch+1, opt.n_epochs, 100. * opt.batch_size / len(train_loader), loss.item()/opt.batch_size, bce.item()/opt.batch_size, kld.item()/opt.batch_size, scheduler.get_lr())

    print(to_print)




if __name__ == "__main__":
    dataset_A = datasets.ImageFolder(root='data/A', transform=transforms.Compose([transforms.ToTensor()]))
    dataloader_A = torch.utils.data.DataLoader(dataset_A, batch_size=opt.batch_size, shuffle=False)

    model = VAE()
    if cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=2000,gamma=0.1)

    for epoch in range(opt.n_epochs):
        scheduler.step(epoch)
        train(epoch, model, optimizer, dataloader_A)

        if epoch % 999 == 0:
            torch.save(model.state_dict(), 'VAE_0325.pth')
