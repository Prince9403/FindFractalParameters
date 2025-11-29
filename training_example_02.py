import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler

from fractals_dataset import FractalImageadDataset
from train_neural_network import train_nn


class ParamPredictor(torch.nn.Module):
    '''
    Critic Class
    Values:
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, hidden_dim=4):
        super().__init__()

        img = torch.from_numpy(np.zeros((1, 3, 224, 224), dtype=np.float32))

        self.paramPredictor1 = torch.nn.Sequential(
            torch.nn.Sequential(
                torch.nn.Conv2d(3, hidden_dim, 4, 2),
                torch.nn.BatchNorm2d(hidden_dim),
                torch.nn.LeakyReLU(0.2, inplace=True),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(hidden_dim, 2 * hidden_dim, 4, 2),
                torch.nn.BatchNorm2d(2 * hidden_dim),
                torch.nn.LeakyReLU(0.2, inplace=True),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(2 *  hidden_dim, 2, 4, 2),
            )
        )

        out1 = self.paramPredictor1(img)
        out1 = out1.view(out1.size(0), -1)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(out1.size(1), 2)
        )


    def forward(self, image):
        '''
        Function for completing a forward pass of the critic: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        '''
        pred1 = self.paramPredictor1(image)
        pred1 = pred1.view(pred1.size(0), -1)
        return self.fc(pred1)


if __name__ == "__main__":
    model = ParamPredictor()

    images_folder = "dataset_preparation/fractals_2_colors"
    train_anno_fpath = "dataset_preparation/fractals_2_colors/params_train_6.json"
    val_anno_fpath = "dataset_preparation/fractals_2_colors/params_val_6.json"
    train_dataset = FractalImageadDataset(train_anno_fpath, images_folder)
    val_dataset = FractalImageadDataset(val_anno_fpath, images_folder)
    optimizer = torch.optim.Adam(model.parameters(), lr=5.0e-4)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    print(f"{datetime.datetime.now()} Started training")
    train_losses, val_losses = train_nn(model, train_dataset, val_dataset, optimizer, scheduler, batch_size=8, num_epochs=20, saving_freq=5,
             save_model_name="custom_model", save_model_folder="train_custom_02")
    print(f"{datetime.datetime.now()} Finished training")

    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.title("Loss")
    plt.legend()
    plt.grid()
    plt.show()