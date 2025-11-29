import datetime

import matplotlib.pyplot as plt
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.models import resnet18

from fractals_dataset import FractalImageadDataset
from train_neural_network import train_nn


if __name__ == "__main__":
    model = resnet18(weights='DEFAULT')

    # modify last layer
    num_feats = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_feats, 2)
    )

    images_folder = "dataset_preparation/fractals_2_colors"
    train_anno_fpath = "dataset_preparation/fractals_2_colors/params_train_6.json"
    val_anno_fpath = "dataset_preparation/fractals_2_colors/params_val_6.json"
    train_dataset = FractalImageadDataset(train_anno_fpath, images_folder)
    val_dataset = FractalImageadDataset(val_anno_fpath, images_folder)
    optimizer = torch.optim.Adam(model.parameters(), lr=5.0e-4)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    print(f"{datetime.datetime.now()} Started training")
    train_losses, val_losses = train_nn(model, train_dataset, val_dataset, optimizer, scheduler, batch_size=8, num_epochs=20, saving_freq=5,
             save_model_name="resnet18", save_model_folder="train_resnet18_01")
    print(f"{datetime.datetime.now()} Finished training")

    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.title("Loss")
    plt.legend()
    plt.grid()
    plt.show()