import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class FractalImageadDataset(Dataset):
    def __init__(self, annotations_filename, images_folder):
        with open(annotations_filename, "r") as fd:
            self.dict_fractals = json.load(fd)
        self.list_fnames = list(self.dict_fractals.keys())
        self.img_dir = images_folder

    def __len__(self):
        return len(self.list_fnames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.list_fnames[idx])
        image = plt.imread(img_path)
        c0, c1 = self.dict_fractals[self.list_fnames[idx]]
        return image, torch.from_numpy(np.array([c0, c1]))