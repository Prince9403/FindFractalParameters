import datetime
import os

import torch
from torch.utils.data import DataLoader


def train_nn(neural_network, train_dataset, validation_dataset, optimizer, batch_size, num_epochs, saving_freq, save_model_name, save_model_folder):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    os.makedirs(save_model_folder, exist_ok=True)

    mse_loss = torch.nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch_num in range(1, num_epochs + 1):
        total_train_loss = 0.0
        total_val_loss = 0.0
        num_train_batches = 0.0
        num_val_batches = 0.0

        neural_network.train()
        for image_batch in train_dataloader:
            images, coeffs = image_batch
            out_tensor = neural_network(images)
            batch_loss = mse_loss(out_tensor, coeffs)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_train_loss += batch_loss.item()
            num_train_batches += 1
        avg_train_loss = total_train_loss / num_train_batches
        train_losses.append(avg_train_loss)

        if epoch_num % saving_freq == 0:
            save_model_path = os.path.join(save_model_folder, save_model_name + f"_epoch__{epoch_num}.ckpt")
            torch.save(neural_network.state_dict(), save_model_path)

        neural_network.eval()
        for image_batch in val_dataloader:
            images, coeffs = image_batch
            out_tensor = neural_network(images)
            batch_loss = mse_loss(out_tensor, coeffs)

            total_val_loss += batch_loss.item()
            num_val_batches += 1
        avg_val_loss = total_val_loss / num_val_batches
        val_losses.append(avg_val_loss)

        print(f"{datetime.datetime.now()} Epoch {epoch_num}, train loss {avg_train_loss}, val loss {avg_val_loss}")

    return train_losses, val_losses





