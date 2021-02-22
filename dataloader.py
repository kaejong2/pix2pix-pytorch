import numpy as np
import torch
from PIL import Image
import random
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import glob

class ImageDataset(Dataset):
    def __init__(self, data_path, transforms_=None, mode='train'):
        self.transform = transforms_
        self.data_path = data_path
        self.files = sorted(glob.glob(os.path.join(data_path,'%s' % mode)+'/*'))
        self.transform = transforms_

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])

        label_img = img.crop((0, 0, img.width/2, img.height))
        data_img = img.crop((img.width/2, 0, img.width, img.height))

        data_img = self.transform(data_img)
        label_img = self.transform(label_img)
        
        return {'data_img': data_img, 'label_img': label_img}

    def __len__(self):
        return len(self.files)

def data_loader(args, mode="train"):
    # Dataset loader
    data_path = os.path.join(args.root_path,args.data_path)
    if mode=='train':
        transforms_ = transforms.Compose(
            [transforms.Resize((256,256), Image.BICUBIC), 
            # transforms.RandomCrop((256, 256)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
        dataset = ImageDataset(data_path, transforms_=transforms_, mode = mode)

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        return dataloader
    else: 
        transforms_ = transforms.Compose([transforms.Resize((256, 256), Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = ImageDataset(data_path, transforms_=transforms_, mode = mode)

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
        return dataloader
