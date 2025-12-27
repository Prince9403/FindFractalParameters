import os
import json
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

model_mean = [0.485, 0.456, 0.406]
model_std = [0.229, 0.224, 0.225]


class FractalImageDataset(Dataset):
    def __init__(self, annotations_filename, images_folder):
        with open(annotations_filename, "r") as fd:
            self.dict_fractals = json.load(fd)
        self.list_fnames = list(self.dict_fractals.keys())
        self.img_dir = images_folder
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(model_mean, model_std)
        ])

    def __len__(self):
        return len(self.list_fnames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.list_fnames[idx])
        image = Image.open(img_path)
        image_torch = self.transform(image)

        c0, c1 = self.dict_fractals[self.list_fnames[idx]]
        return image_torch, torch.from_numpy(np.array([c0, c1], dtype=np.float32))
