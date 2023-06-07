import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class augmentation_dataset(Dataset):
  def __init__(self, data, transform):
    self.data = data
    self.transform = transform

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self):
    list = []
    for i in range(len(self.data)):
      pixels_array = np.array(self.data[i], dtype=np.uint8)
      image = Image.fromarray(pixels_array, mode='L')
      item = self.transform(image)
      item = np.asarray(item)
      list.append(item.astype('float32'))
    
    list = np.asarray(list)
    list = np.expand_dims(list, -1)
    list /= 255.0
    return torch.tensor(list, dtype=torch.float32)


def get_transform():
    return transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(48, scale=(0.8, 1.0)),  # Randomly crop and resize the image
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, saturation, and hue
        transforms.GaussianBlur(kernel_size=3),  # Apply random Gaussian blur
    ])