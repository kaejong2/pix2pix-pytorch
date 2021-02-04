import numpy as np
import torch
from PIL import Image
import random
import os
from skimage import transform
import matplotlib.pyplot as plt

class data_loader(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=[]):
        self.data_dir = data_dir
        self.transform = transform
        self.data_list = os.listdir(data_dir)

    def __getitem__(self, index):
        data = plt.imread(os.path.join(self.data_dir, self.data_list[index]))[:, :, :3]

        if data.dtype == np.uint8:
            data = data / 255.0

        boundary = int(data.shape[1]/2)

        dataA = data[:, :boundary, :]
        dataB = data[:, boundary:, :]

        data = {'dataA': dataB, 'dataB': dataA}
        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.data_list)


class ToTensor(object):
    def __call__(self, data):
        dataA, dataB = data['dataA'], data['dataB']
        dataA = dataA.transpose((2, 0, 1)).astype(np.float32)
        dataB = dataB.transpose((2, 0, 1)).astype(np.float32)
        return {'dataA': torch.from_numpy(dataA), 'dataB': torch.from_numpy(dataB)}

class Normalize(object):
    def __call__(self, data):
        dataA, dataB = data['dataA'], data['dataB']
        dataA = 2 * dataA - 1
        dataB = 2 * dataB - 1
        return {'dataA': dataA, 'dataB': dataB}

class RandomFlip(object):
    def __call__(self, data):
        dataA, dataB = data['dataA'], data['dataB']

        if np.random.rand() > 0.5:
            dataA = np.fliplr(dataA)
            dataB = np.fliplr(dataB)

        return {'dataA': dataA, 'dataB': dataB}

class Rescale(object):

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size

  def __call__(self, data):
    dataA, dataB = data['dataA'], data['dataB']

    h, w = dataA.shape[:2]

    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)

    dataA = transform.resize(dataA, (new_h, new_w))
    dataB = transform.resize(dataB, (new_h, new_w))

    return {'dataA': dataA, 'dataB': dataB}


class RandomCrop(object):
  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size)
    else:
      assert len(output_size) == 2
      self.output_size = output_size

  def __call__(self, data):
    dataA, dataB = data['dataA'], data['dataB']

    h, w = dataA.shape[:2]
    new_h, new_w = self.output_size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    dataA = dataA[top: top + new_h, left: left + new_w]
    dataB = dataB[top: top + new_h, left: left + new_w]

    return {'dataA': dataA, 'dataB': dataB}


class ToNumpy(object):
    def __call__(self, data):
        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

class Denomalize(object):
    def __call__(self, data):
        return (data + 1) / 2